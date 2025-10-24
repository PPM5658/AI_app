# qa_system.py
# Handles interaction with the Gemini LLM
# --- UPDATED WITH RESILIENT RETRY LOGIC ---

import google.generativeai as genai
from google.colab import userdata
import config
from config import GEMINI_API_KEY as CONFIG_API_KEY
import os
import time  # <-- 1. ADDED
import random  # <-- 2. ADDED
from google.api_core import exceptions as google_exceptions  # <-- 3. ADDED

class QASystem:
    def __init__(self):
        # 1. Retrieve API key
        api_key = userdata.get('gemini') or CONFIG_API_KEY

        # ... (your placeholder check logic can stay here) ...
        
        # 3. Assign to instance
        self.api_key = api_key

        # 4. Configure Gemini client
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini client. Verify API key. Error: {e}")

        # 5. Generation config
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        # 6. Safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # 7. Load model
        model_name = getattr(config, "GEMINI_MODEL", "gemini-pro-latest")
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )

        print(f"✅ Gemini model '{model_name}' loaded successfully.")

    # --- 4. NEW PRIVATE METHOD WITH RETRY LOGIC ---
    def _generate_with_retry(self, prompt_parts):
        """
        Calls the Gemini API with exponential backoff and retry logic.
        """
        max_retries = 3
        base_wait_time = 2  # Start with 2 seconds
        
        for i in range(max_retries):
            try:
                # --- This is your API call ---
                response = self.model.generate_content(prompt_parts)
                return response.text  # Success!
                # -------------------------------

            except google_exceptions.ResourceExhausted as e:  # This is the 429 Error
                print(f"WARNING: Rate limit hit (429). Attempt {i+1}/{max_retries}.")
                if i == max_retries - 1:
                    print("ERROR: Max retries exceeded for 429 error.")
                    # This is the "human-like" response
                    return "I'm currently experiencing a high volume of requests. Please try again in a moment."
                
                # Exponential backoff: 2s, 4s, 8s... + random "jitter"
                wait_time = (base_wait_time ** (i + 1)) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            except (google_exceptions.InternalServerError, google_exceptions.ServiceUnavailable) as e:  # These are 500/503 Errors
                print(f"WARNING: Gemini server error ({e}). Attempt {i+1}/{max_retries}.")
                if i == max_retries - 1:
                    print("ERROR: Max retries exceeded for 5xx error.")
                    # This is the "human-like" response
                    return "I'm sorry, the generative AI service I rely on seems to be having a temporary issue. Please try again in a few minutes."
                
                # Wait and retry
                wait_time = (base_wait_time ** i) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            except Exception as e:
                # Catch all other unexpected errors
                print(f"ERROR: An unexpected error occurred: {e}")
                return "I'm sorry, an unexpected error occurred while processing your request."

        # Fallback in case loop finishes without returning
        return "I'm sorry, I had a problem generating an answer after several attempts."

    # --- 5. UPDATED get_answer METHOD ---

    def get_answer(self, query, context_texts_with_scores):
        """Generates an answer using Gemini based on the query and retrieved context."""
        
        # --- THIS IS THE NEW LOGIC ---
        # We set a threshold for L2 distance. You MUST tune this value.
        # Start with something like 1.0.
        MIN_SIMILARITY_THRESHOLD = 0.2
        
        relevant_memories = []
        for text, score in context_texts_with_scores:
            # --- INVERT THE CHECK ---
            if score >= MIN_SIMILARITY_THRESHOLD:
                relevant_memories.append(text)
            else:
                # This log is crucial for debugging your threshold!
                print(f"DEBUG: Discarding memory (score {score:.2f} < {MIN_SIMILARITY_THRESHOLD}): '{text}'")
        # --- END OF NEW LOGIC ---

        if not relevant_memories:
            return "I've scanned the video, but I haven't found any information related to that specific question yet."

        context_prompt = "\n".join(relevant_memories)

        context_prompt = "\n".join(relevant_memories) # <-- Use the filtered list
        prompt_parts = [
            "You are a video analysis assistant. Your memory consists of the following events from a video. "
            "Answer the user's question based *only* on these memories. "
            "If the memories don't provide an answer, say so.\n\n"
            "--- MEMORY CONTEXT ---",
            context_prompt,
            "--- END OF MEMORY ---",
            f"\nUser Question: \"{query}\"",
        ]

        return self._generate_with_retry(prompt_parts)