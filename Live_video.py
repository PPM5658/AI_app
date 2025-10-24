# parallel_batch_main.py
# Runs file processing and Q&A in parallel threads.

import config
from video_processor import VideoProcessor
from memory_bank import MemoryBank
from ultralytics import YOLO
from qa_system import QASystem
import threading

def run_video_processing(
    processor,
    memory,
    model,
    stop_event,
    video_done_event,
    video_path=None,
    progress_callback=None,
):
    """
    This function runs in a separate thread.
    It processes the video file and streams new memories to an optional callback.
    """
    active_video_path = video_path or config.VIDEO_PATH
    print(f"Video thread started: Processing video file '{active_video_path}'...")

    try:
        # We loop through the generator and feed the results to the memory bank
        for frame, timestamp, results in processor.process_video(
            active_video_path, stop_event=stop_event
        ):
            new_memories = memory.add_frame_memories(results, timestamp, model)

            if progress_callback and new_memories:
                try:
                    progress_callback(timestamp, new_memories)
                except Exception as callback_error:
                    print(f"Warning: progress callback raised an error: {callback_error}")

            if stop_event.is_set():
                print("Stop signal received. Exiting video processing thread loop.")
                break

    except Exception as e:
        if not stop_event.is_set(): # Only log error if it wasn't a manual quit
            print(f"Error in video thread: {e}")
    finally:
        # Signal that the video is done processing
        video_done_event.set()
        if not stop_event.is_set():
            print("\n--- Video processing finished. Memory bank is complete. ---")
            print("You can continue asking questions. Type 'quit' to exit.")


def run_qa_loop(qa, memory, stop_event, video_done_event):
    """
    This function runs in the main thread.
    It handles user questions.
    """
    print("\n--- Starting Parallel Q&A Session ---")
    print("Video processing is running in the background.")
    print("You can start asking questions now (type 'quit' to exit).")

    while not stop_event.is_set():
        try:
            if video_done_event.is_set():
                # Check if the video thread has quit
                pass # Video is done, just continue the Q&A loop
            
            user_question = input("\n🤔 User Question: ")
            
            if user_question.lower() == 'quit':
                print("Quit command received. Shutting down all threads...")
                stop_event.set() # Signal the video thread to stop
                break
            
            print("looking through memories...")
            # Search is now thread-safe
            context = memory.search_memories(user_question, k=5)
            
            if not context and not video_done_event.is_set():
                print("Found no memories yet. (Video is still processing...)")
                continue
            
            print("Found relevant memories:")
            for mem in context:
                print(f"  - {mem}")
            
            print("\n🤖 Generating final answer with Gemini...")
            answer = qa.get_answer(user_question, context)
            print(f"\nFinal Answer:\n{answer}")

        except EOFError:
            # Handle user pressing Ctrl+D or Colab input issues
            print("Input error. Shutting down...")
            stop_event.set()
            break
        except Exception as e:
            print(f"Error in Q&A loop: {e}")

    print("Exiting Q&A session.")

# This makes the script runnable from the command line
if __name__ == "__main__":
    # 1. Initialize all components ONCE
    print("Loading models...")
    model = YOLO(config.YOLO_MODEL) # Load model once
    processor = VideoProcessor()
    memory = MemoryBank()
    qa = QASystem()
    print("All components loaded.")

    # 2. Create "stop signals" for the threads
    stop_signal = threading.Event()
    video_done_signal = threading.Event() # To signal when video is finished

    # 3. Configure and start the video thread
    video_thread = threading.Thread(
        target=run_video_processing,
        args=(processor, memory, model, stop_signal, video_done_signal),
        daemon=True # A daemon thread will exit when the main program exits
    )
    video_thread.start()

    # 4. Run the Q&A loop in the main thread
    run_qa_loop(qa, memory, stop_signal, video_done_signal)

    # 5. Clean up
    print("Waiting for video thread to finish...")
    video_thread.join() # Wait for the video thread to clean up
    print("Program exited cleanly.")