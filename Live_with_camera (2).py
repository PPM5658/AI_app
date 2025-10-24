# live_main.py
# ------------------------------------------------------------------
# Main entry point for the live video + Q&A system.
# This script connects all system components:
#   - YOLO (object detection and tracking)
#   - VideoProcessor (frame processing)
#   - MemoryBank (semantic storage of detections)
#   - QASystem (Gemini model for answering questions)
#
# It runs video capture and user Q&A in parallel threads,
# allowing real-time object memory generation and questioning.
# ------------------------------------------------------------------

import config
from VideoProcessor import VideoProcessor
from memory_bank import MemoryBank
from ultralytics import YOLO
from qa_system import QASystem
import threading
import cv2
import sys


def run_video_processing(processor, memory, model, stop_event):
    """
    Runs the video feed in a separate background thread.

    Continuously captures frames (from webcam or file),
    performs YOLO detection/tracking, and sends the results
    to the MemoryBank for storage.

    Args:
        processor (VideoProcessor): Handles YOLO tracking per frame.
        memory (MemoryBank): Stores textual memories for each detected object.
        model (YOLO): YOLO model instance used for class name lookup.
        stop_event (threading.Event): Shared signal to stop processing safely.
    """
    print("Video thread started: Opening webcam...")

    try:
        # Choose video source:
        #   - 0 → default webcam (live mode)
        #   - config.VIDEO_PATH → test video (if argument passed)
        video_source = 0 if len(sys.argv) < 2 else config.VIDEO_PATH

        # Process video frame-by-frame
        for frame, timestamp, results in processor.process_video(video_source, stop_event):
            # Add new object detections to memory
            memory.add_frame_memories(results, timestamp, model)

            # --- Optional: Visualize detections here if desired ---
            cv2.imshow("Live Feed - (Q&A in console)", frame)

            # Stop feed when user presses 'q'
            if cv2.waitKey(1) == ord('q'):
                stop_event.set()
                break

    except Exception as e:
        # Catch unexpected video processing errors
        if not stop_event.is_set():
            print(f"Error in video thread: {e}")
    finally:
        print("Video feed stopping...")
        # Ensure OpenCV windows are properly closed
        cv2.destroyAllWindows()


def run_qa_loop(qa, memory, stop_event):
    """
    Runs in the main thread to handle user interactions (Q&A).

    Allows the user to ask natural-language questions about
    what has been detected so far in the live video feed.

    Args:
        qa (QASystem): Handles communication with the Gemini model.
        memory (MemoryBank): Provides search capability for relevant memories.
        stop_event (threading.Event): Shared signal to stop both loops.
    """
    print("\n--- Starting Live Q&A Session ---")
    print("Video feed is running. Ask questions in this console.")
    print("Type 'quit' to exit both the Q&A and the video feed.")

    while not stop_event.is_set():
        try:
            # Take user input from the console
            user_question = input("\n🤔 User Question: ")

            # Graceful exit command
            if user_question.lower() == 'quit':
                print("Quit command received. Shutting down...")
                stop_event.set()
                break

            print("Looking through memories...")
            context = memory.search_memories(user_question, k=5)

            # Handle case where no detections yet
            if not context:
                print("Found no memories yet. (Feed is still running...)")
                continue

            # Display matched memories with similarity scores
            print("Found relevant memories:")
            for mem in context:
                print(f"  - {mem}")

            # Generate an answer using the Gemini model
            print("\n🤖 Generating final answer with Gemini...")
            answer = qa.get_answer(user_question, context)
            print(f"\nFinal Answer:\n{answer}")

        except EOFError:
            # Handles accidental input interruption
            print("Input error. Shutting down...")
            stop_event.set()
            break
        except Exception as e:
            print(f"Error in Q&A loop: {e}")

    print("Exiting Q&A session.")


# ------------------------------------------------------------------
# Program Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Load all required components
    print("Loading models...")
    model = YOLO(config.YOLO_MODEL)      # Load YOLOv8 model
    processor = VideoProcessor()         # Frame-by-frame video processor
    memory = MemoryBank()                # Semantic storage for object memories
    qa = QASystem()                      # Gemini-based question-answering model
    print("All components loaded successfully.")

    # Shared flag used to signal all threads to stop safely
    stop_signal = threading.Event()

    # Start video feed in background thread
    video_thread = threading.Thread(
        target=run_video_processing,
        args=(processor, memory, model, stop_signal),
        daemon=True
    )
    video_thread.start()

    # Run interactive Q&A in the main thread
    run_qa_loop(qa, memory, stop_signal)

    # Wait for background video thread to exit
    print("Waiting for video thread to finish...")
    video_thread.join()

    print("Program exited cleanly.")
