# qa_system.py
# --------------------------------------------------------------------
# Handles Gemini-based text Q&A and integrates with VideoProcessor for
# multi-object visual highlighting, distance display, retry logic,
# and Drive auto-save for captured highlights.
# --------------------------------------------------------------------

import os
import re
import time
import shutil
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

try:
    from google.colab import userdata
except ImportError:
    userdata = None

import config
from config import GEMINI_API_KEY as CONFIG_API_KEY


class QASystem:
    def __init__(self, video_processor=None):
        self.video_processor = video_processor

        # ---------------- Gemini API Setup ----------------
        colab_key = None
        if userdata is not None:
            try:
                colab_key = userdata.get("gemini")
            except Exception:
                pass

        env_key = os.getenv("GEMINI_API_KEY")
        api_key = colab_key or env_key or CONFIG_API_KEY
        if not api_key:
            raise ValueError("No Gemini API key found.")

        genai.configure(api_key=api_key)
        self.api_key = api_key

        self.model = genai.GenerativeModel(
            model_name=getattr(config, "GEMINI_MODEL", "gemini-pro-latest"),
            generation_config={
                "temperature": 0.2,
                "top_p": 1,
                "max_output_tokens": 2048
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        print("✅ Gemini model initialized with multi-object visual support.")

    # ------------------------------------------------------------
    # Helper: Retry Gemini requests safely (with 429 fallback)
    # ------------------------------------------------------------
    def _generate_with_retry(self, prompt_parts):
        """Generate content with retry and fallback handling."""
        for attempt in range(3):
            try:
                return self.model.generate_content(prompt_parts).text
            except google_exceptions.ResourceExhausted:
                print(f"⚠️ Rate limit, retrying ({attempt+1}/3)...")
                time.sleep(2 ** (attempt + 1))
            except google_exceptions.ServiceUnavailable:
                print(f"⚠️ Gemini API temporarily unavailable, retrying ({attempt+1}/3)...")
                time.sleep(1)
            except google_exceptions.GoogleAPIError as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print("⚠️ Gemini quota exceeded — using fallback text mode.")
                    return (
                        "Gemini quota limit reached. Visual analysis completed, "
                        "but text reasoning is temporarily unavailable."
                    )
                raise
            except Exception as e:
                print(f"Unexpected Gemini error: {e}")
                return "An unexpected error occurred."
        return "Gemini API unavailable after multiple retries."

    # ------------------------------------------------------------
    # Extract multiple labels from user query
    # ------------------------------------------------------------
    def _extract_object_labels(self, query):
        """Find all YOLO class names mentioned in the user's question."""
        if not self.video_processor:
            return []
        yolo_labels = [n.lower() for n in self.video_processor.model.names.values()]
        found = [lbl for lbl in yolo_labels if re.search(rf"\b{re.escape(lbl)}\b", query.lower())]
        return list(set(found))

    # ------------------------------------------------------------
    # Main Q&A pipeline
    # ------------------------------------------------------------
    def get_answer(self, query, context_texts_with_scores):
        """
        Generate both text and image-based answer.
        Includes highlight retry and Drive auto-save.
        """
        relevant_memories = [
            text for text, score in context_texts_with_scores if score >= 0.2
        ]

        labels = self._extract_object_labels(query)
        image_path = None

        # 1️⃣ Capture visual highlight (with retry)
        if labels and self.video_processor:
            try:
                image_path, err = self.video_processor.capture_highlight(labels)
                if err or image_path is None:
                    print("[Visual Q&A] Highlight unavailable yet. Retrying in 1s...")
                    time.sleep(1)
                    image_path, err = self.video_processor.capture_highlight(labels)
                    if err:
                        print(f"[Visual Q&A] Retry highlight skipped: {err}")
                        image_path = None
                    elif image_path:
                        print("[Visual Q&A] Highlight capture successful on retry.")
            except Exception as e:
                print(f"[Visual Q&A] Highlight failed: {e}")
                image_path = None

        # 2️⃣ Auto-save highlight to Google Drive
        if image_path and os.path.exists(image_path):
            drive_dir = "/content/drive/MyDrive/ObjectTrackingCaptures"
            try:
                if os.path.exists("/content/drive/MyDrive"):
                    os.makedirs(drive_dir, exist_ok=True)
                    dest = os.path.join(drive_dir, os.path.basename(image_path))
                    shutil.copy(image_path, dest)
                    print(f"📂 Highlight also saved to Drive: {dest}")
            except Exception as e:
                print(f"⚠️ Drive sync failed: {e}")

        # 3️⃣ Generate Gemini text answer
        if not relevant_memories:
            text_answer = "I don't have enough context about that yet."
        else:
            prompt = (
                "You are an intelligent video assistant.\n"
                "Use the memory logs to answer the question accurately.\n"
                "If distances are relevant, include them in your explanation.\n\n"
                f"--- CONTEXT ---\n{chr(10).join(relevant_memories)}\n"
                f"--- QUESTION ---\n{query}\n"
            )
            text_answer = self._generate_with_retry([prompt])

        # 4️⃣ Package results
        return {
            "text_answer": text_answer,
            "image_path": image_path,
            "object_labels": labels,
        }
