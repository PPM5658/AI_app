# live_video.py
# ------------------------------------------------------------
# Runs real-time video processing and visual Q&A in parallel.
# Integrated with YOLO-based highlight capture for visual answers.
# Includes Colab image display and Drive auto-save.
# ------------------------------------------------------------

import config
import threading
from ultralytics import YOLO
from video_processor import VideoProcessor
from memory_bank import MemoryBank
from qa_system import QASystem
import cv2
import os

# ------------------------------------------------------------
# Thread: Video Processing
# ------------------------------------------------------------
def run_video_processing(
    processor,
    memory,
    model,
    stop_event,
    video_done_event,
    video_path=None,
    progress_callback=None,
):
    """Runs YOLO-based tracking and updates memory bank continuously."""
    active_video_path = video_path or config.VIDEO_PATH
    print(f"🎥 Video thread started: Processing '{active_video_path}'...")

    try:
        for frame, timestamp, results in processor.process_video(
            active_video_path, stop_event=stop_event
        ):
            new_memories = memory.add_frame_memories(results, timestamp, model)

            if progress_callback and new_memories:
                try:
                    progress_callback(timestamp, new_memories)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")

            if stop_event.is_set():
                print("🛑 Stop signal received. Stopping video thread.")
                break

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Error in video thread: {e}")
    finally:
        video_done_event.set()
        if not stop_event.is_set():
            print("\n✅ Video processing finished. Memory bank ready.")
            print("You can continue asking questions. Type 'quit' to exit.")


# ------------------------------------------------------------
# Thread: Q&A Interactive Loop
# ------------------------------------------------------------
def run_qa_loop(qa, memory, stop_event, video_done_event):
    """Handles user input, performs memory retrieval and Gemini Q&A."""
    print("\n--- 🧠 Visual Q&A Session Started ---")
    print("Video is processing in background. Ask questions anytime.")
    print("Type 'quit' to exit.\n")

    while not stop_event.is_set():
        try:
            if video_done_event.is_set():
                pass  # Video fully processed

            user_question = input("\n🤔 Your Question: ")

            if user_question.lower() == "quit":
                print("👋 Quitting session and stopping threads...")
                stop_event.set()
                break

            print("🔍 Searching memories...")
            context = memory.search_memories(user_question, k=5)

            if not context and not video_done_event.is_set():
                print("⏳ No relevant memories yet (video still running).")
                continue

            print("🧩 Relevant memories found:")
            for mem in context:
                print(f"  - {mem[0]} (score={mem[1]:.2f})")

            print("\n🤖 Generating Gemini answer...")
            result = qa.get_answer(user_question, context)

            # ------------------- Output Section -------------------------
            text_answer = result.get("text_answer", "")
            image_path = result.get("image_path", None)
            object_labels = result.get("object_labels", [])

            print("\n======================")
            print("📜 TEXT ANSWER:")
            print(text_answer)
            print("======================")

            # Display highlighted image (if available)
            if image_path:
                print(f"🖼️ Highlighted image (with distances) saved: {image_path}")
                # 🧠 Optional auto-save to Drive
                drive_path = "/content/drive/MyDrive/ObjectTrackingCaptures"
                try:
                    if os.path.exists("/content/drive/MyDrive"):
                        os.makedirs(drive_path, exist_ok=True)
                        dest = os.path.join(drive_path, os.path.basename(image_path))
                        import shutil
                        shutil.copy(image_path, dest)
                        print(f"📂 Also saved to Drive: {dest}")
                except Exception as e:
                    print(f"⚠️ Drive sync failed: {e}")

                # Try to display the image inline if running in Colab
                try:
                    from google.colab.patches import cv2_imshow
                    img = cv2.imread(image_path)
                    if img is not None:
                        cv2_imshow(img)
                        print("👁️  Displayed highlighted frame above.")
                    else:
                        print("⚠️ Could not load image for display.")
                except ImportError:
                    # Not in Colab → show locally
                    try:
                        img = cv2.imread(image_path)
                        if img is not None:
                            cv2.imshow("Highlighted Objects", img)
                            print("Press any key to close the image window...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            print("⚠️ Could not load image for display.")
                    except Exception as e:
                        print(f"⚠️ Image display unavailable: {e}")

            else:
                print("📷 No visual highlight available for this question.")

            # Display all detected object labels
            if object_labels:
                print(f"🎯 Objects highlighted: {', '.join(object_labels)}")

        except EOFError:
            print("Input error or EOF detected. Exiting...")
            stop_event.set()
            break
        except Exception as e:
            print(f"⚠️ Error in Q&A loop: {e}")

    print("🧹 Exiting Q&A session.")


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Initializing components...")

    # 1️⃣ Load models
    model = YOLO(config.YOLO_MODEL)
    processor = VideoProcessor()
    memory = MemoryBank()
    qa = QASystem(video_processor=processor)  # ✅ pass processor reference

    print("✅ All components loaded successfully.")

    # 2️⃣ Thread control flags
    stop_signal = threading.Event()
    video_done_signal = threading.Event()

    # 3️⃣ Launch video thread
    video_thread = threading.Thread(
        target=run_video_processing,
        args=(processor, memory, model, stop_signal, video_done_signal),
        daemon=True,
    )
    video_thread.start()

    # 4️⃣ Run interactive Q&A loop
    run_qa_loop(qa, memory, stop_signal, video_done_signal)

    # 5️⃣ Graceful cleanup
    print("🧭 Waiting for video thread to finish...")
    video_thread.join()
    print("✅ Program exited cleanly.")
