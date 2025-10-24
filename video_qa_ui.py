"""Graphical interface for running the video QA pipeline."""

from __future__ import annotations

import importlib
import os
import queue
import threading
from datetime import datetime
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:  # Optional dependency used for the video preview.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # Optional dependency used for Tk-compatible image objects.
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


def _safe_import(module: str, attr: Optional[str] = None):
    """Import a module or attribute, capturing errors for user-friendly reporting."""

    try:
        imported = importlib.import_module(module)
        if attr:
            return getattr(imported, attr)
        return imported
    except Exception as exc:  # pragma: no cover - defensive guard
        return exc


config = _safe_import("config")
run_video_processing = _safe_import("Live_video", "run_video_processing")
MemoryBank = _safe_import("memory_bank", "MemoryBank")
QASystem = _safe_import("qa_system", "QASystem")
format_time = _safe_import("utils", "format_time")
VideoProcessor = _safe_import("video_processor", "VideoProcessor")


class VideoQAApp:
    """Tkinter-based UI that orchestrates the video QA workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video QA Assistant")
        self.root.geometry("1000x780")

        # State
        self.processor = None
        self.memory_bank = None
        self.qa_system = None
        self.yolo_model = None
        self.models_loaded = False

        self.stop_event = threading.Event()
        self.video_done_event = threading.Event()
        self.video_thread = None
        self.qa_thread = None

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._video_finished_logged = False

        self.progress_var = tk.DoubleVar(value=0.0)
        self._video_duration_seconds: Optional[float] = None
        self._preview_capture: Optional["cv2.VideoCapture"] = None
        self._preview_image = None

        # UI variables
        default_path = getattr(config, "VIDEO_PATH", "") if not isinstance(config, Exception) else ""
        self.video_path_var = tk.StringVar(value=default_path)
        self.question_var = tk.StringVar()

        self._build_layout()

        self._report_import_errors()

        self.root.after(200, self._process_log_queue)
        self.root.after(500, self._poll_video_completion)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video selection controls
        path_frame = ttk.LabelFrame(main_frame, text="Video Source")
        path_frame.pack(fill=tk.X, pady=(0, 10))

        path_entry = ttk.Entry(path_frame, textvariable=self.video_path_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=10)

        ttk.Button(path_frame, text="Browse", command=self._browse_video).pack(
            side=tk.LEFT, padx=(0, 10), pady=10
        )

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        self.init_button = ttk.Button(
            button_frame, text="Initialize Models", command=self.initialize_models
        )
        self.init_button.pack(side=tk.LEFT, padx=(0, 10))

        self.start_button = ttk.Button(
            button_frame,
            text="Start Video Processing",
            command=self.start_video_processing,
            state=tk.DISABLED,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_processing,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT)

        # Preview & progress widgets
        preview_frame = ttk.LabelFrame(main_frame, text="Video Preview")
        preview_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.preview_label = ttk.Label(
            preview_frame,
            text="Preview unavailable until processing starts.",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        progress_frame = ttk.Frame(preview_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=(10, 10))

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=1.0
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        self.progress_status = ttk.Label(progress_frame, text="0%")
        self.progress_status.pack(side=tk.LEFT, padx=(10, 0))

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_widget = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Question/answer panel
        qa_frame = ttk.LabelFrame(main_frame, text="Ask a Question")
        qa_frame.pack(fill=tk.BOTH, expand=False)

        question_entry = ttk.Entry(qa_frame, textvariable=self.question_var)
        question_entry.pack(fill=tk.X, padx=10, pady=(10, 5))
        question_entry.bind("<Return>", lambda event: self.ask_question())

        self.ask_button = ttk.Button(
            qa_frame, text="Ask", command=self.ask_question, state=tk.DISABLED
        )
        self.ask_button.pack(padx=10, pady=(0, 10), anchor=tk.W)

        self.answer_widget = scrolledtext.ScrolledText(
            qa_frame, wrap=tk.WORD, height=8, state=tk.DISABLED
        )
        self.answer_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")],
        )
        if selected:
            self.video_path_var.set(selected)

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _report_import_errors(self) -> None:
        errors = [
            (name, dependency)
            for name, dependency in (
                ("config", config),
                ("Live_video.run_video_processing", run_video_processing),
                ("memory_bank.MemoryBank", MemoryBank),
                ("qa_system.QASystem", QASystem),
                ("utils.format_time", format_time),
                ("video_processor.VideoProcessor", VideoProcessor),
            )
            if isinstance(dependency, Exception)
        ]

        if errors:
            warning_lines = [
                "Some modules could not be imported. Certain features will be disabled:",
                *[f" - {name}: {exc}" for name, exc in errors],
            ]
            warning = "\n".join(warning_lines)
            self._log(warning)
            self.root.after(
                0,
                lambda: messagebox.showwarning(
                    "Missing dependencies",
                    warning,
                ),
            )

    def _append_to_widget(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text + "\n")
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _process_log_queue(self) -> None:
        while not self.log_queue.empty():
            self._append_to_widget(self.log_widget, self.log_queue.get())
        self.root.after(200, self._process_log_queue)

    def _poll_video_completion(self) -> None:
        if (
            self.video_done_event.is_set()
            and not self._video_finished_logged
        ):
            self._log("Video processing finished. You can continue asking questions.")
            self.stop_button.configure(state=tk.DISABLED)
            if self._video_duration_seconds:
                self._update_progress(1.0)
            self._video_finished_logged = True
        self.root.after(500, self._poll_video_completion)

    def _on_close(self) -> None:
        self.stop_processing()
        self.root.after(200, self.root.destroy)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    def initialize_models(self) -> None:
        if self.models_loaded:
            messagebox.showinfo("Models Ready", "Models are already initialised.")
            return

        self.init_button.configure(state=tk.DISABLED)

        if isinstance(VideoProcessor, Exception) or isinstance(MemoryBank, Exception) or isinstance(QASystem, Exception):
            messagebox.showerror(
                "Missing modules",
                "Cannot initialise models because required modules failed to import."
                " Check the logs for details.",
            )
            self.init_button.configure(state=tk.NORMAL)
            return

        def _load():
            try:
                self._log("Loading models. This may take a moment...")

                processor = VideoProcessor()
                memory_bank = MemoryBank()
                qa_system = QASystem()

            except Exception as exc:  # pragma: no cover - UI feedback
                self._log(f"Failed to initialise models: {exc}")

                def _handle_error() -> None:
                    self.init_button.configure(state=tk.NORMAL)
                    messagebox.showerror(
                        "Initialisation Error",
                        f"Could not initialise the models.\n{exc}",
                    )

                self.root.after(0, _handle_error)
                return

            def _handle_success() -> None:
                self.processor = processor
                self.memory_bank = memory_bank
                self.qa_system = qa_system
                self.yolo_model = processor.model

                self.models_loaded = True
                self._log("Models loaded successfully.")

                self.start_button.configure(state=tk.NORMAL)
                self.ask_button.configure(state=tk.NORMAL)

            self.root.after(0, _handle_success)

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------
    # Video processing controls
    # ------------------------------------------------------------------
    def start_video_processing(self) -> None:
        if isinstance(run_video_processing, Exception):
            messagebox.showerror(
                "Missing function",
                "Video processing cannot start because the pipeline entry point"
                " failed to import. Check the logs for more details.",
            )
            return

        if not self.models_loaded:
            messagebox.showwarning(
                "Models not ready", "Please initialise the models first."
            )
            return

        if self.video_thread and self.video_thread.is_alive():
            messagebox.showinfo(
                "Processing", "Video processing is already running."
            )
            return

        video_path = self.video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Missing path", "Select a video file first.")
            return

        if not os.path.isfile(video_path):
            messagebox.showerror("Invalid file", "The selected video file was not found.")
            return

        self.stop_event.clear()
        self.video_done_event.clear()
        self._video_finished_logged = False

        self.stop_button.configure(state=tk.NORMAL)
        self._update_progress(0.0)

        self._log(f"Starting video processing for '{video_path}'.")

        self._prepare_video_metadata(video_path)
        self._start_preview(video_path)

        def _run():
            try:
                run_video_processing(
                    self.processor,
                    self.memory_bank,
                    self.yolo_model,
                    self.stop_event,
                    self.video_done_event,
                    video_path=video_path,
                    progress_callback=self._on_new_memories,
                )
            finally:
                self.root.after(
                    0, lambda: self.stop_button.configure(state=tk.DISABLED)
                )

        self.video_thread = threading.Thread(target=_run, daemon=True)
        self.video_thread.start()

    def stop_processing(self) -> None:
        if self.stop_event.is_set():
            return

        self.stop_event.set()
        self._log("Stop signal sent. Waiting for background threads to exit...")
        self._stop_preview()

    def _on_new_memories(self, timestamp: float, memories: list[str]) -> None:
        if isinstance(format_time, Exception):
            time_str = f"{timestamp:.2f}s"
        else:
            time_str = format_time(timestamp)
        for memory_text in memories:
            self._log(f"{time_str} - {memory_text}")

        if self._video_duration_seconds:
            progress = min(max(timestamp / self._video_duration_seconds, 0.0), 1.0)
            self._update_progress(progress)

    def _prepare_video_metadata(self, video_path: str) -> None:
        if not cv2:  # pragma: no cover - optional dependency
            self._log(
                "OpenCV is not installed. Progress information will rely on logs only."
            )
            self._video_duration_seconds = None
            return

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            self._log("Could not open video for metadata. Progress bar disabled.")
            self._video_duration_seconds = None
            capture.release()
            return

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()

        if fps <= 0 or frame_count <= 0:
            self._log("Unable to determine video duration. Progress bar disabled.")
            self._video_duration_seconds = None
        else:
            self._video_duration_seconds = frame_count / fps
            self._log(
                f"Video duration detected: {self._video_duration_seconds:.1f} seconds."
            )
            self._update_progress(0.0)

    def _update_progress(self, value: float) -> None:
        self.progress_var.set(value)
        self.progress_status.configure(text=f"{int(value * 100)}%")

    def _start_preview(self, video_path: str) -> None:
        if not cv2 or not Image or not ImageTk:  # pragma: no cover - optional dependency
            self._log(
                "Preview disabled. Install OpenCV and Pillow for live frame display."
            )
            self.preview_label.configure(
                image="",
                text="Preview requires OpenCV and Pillow.",
            )
            return

        self._stop_preview()

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            self._log("Could not open video for preview display.")
            return

        self._preview_capture = capture
        self._log("Video preview initialised.")
        self._schedule_preview_update()

    def _schedule_preview_update(self) -> None:
        if not self._preview_capture:
            return

        ret, frame = self._preview_capture.read()
        if not ret:
            self._log("Reached end of preview stream.")
            self._stop_preview(reset_progress=False)
            self._update_progress(1.0)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((640, 360))
        photo = ImageTk.PhotoImage(image=image)

        self._preview_image = photo  # keep reference alive
        self.preview_label.configure(image=photo, text="")

        # Schedule next frame update.
        self.root.after(33, self._schedule_preview_update)

    def _stop_preview(self, *, reset_progress: bool = True) -> None:
        if self._preview_capture:
            self._preview_capture.release()
            self._preview_capture = None
        self.preview_label.configure(
            image="",
            text="Preview stopped.",
        )
        self._preview_image = None
        if reset_progress:
            self._update_progress(0.0)

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def ask_question(self) -> None:
        if not self.models_loaded:
            messagebox.showwarning(
                "Models not ready", "Please initialise the models first."
            )
            return

        question = self.question_var.get().strip()
        if not question:
            messagebox.showinfo("Empty question", "Please type a question to ask.")
            return

        if self.qa_thread and self.qa_thread.is_alive():
            messagebox.showinfo(
                "Busy", "A question is already being processed. Please wait."
            )
            return

        self.ask_button.configure(state=tk.DISABLED)
        self._log(f"User question: {question}")
        self._set_answer_text("Thinking...")

        def _run_qa():
            try:
                context = self.memory_bank.search_memories(question, k=5)
                if not context and not self.video_done_event.is_set():
                    self._log(
                        "No relevant memories yet. Video processing may still be running."
                    )

                answer = self.qa_system.get_answer(question, context)
            except Exception as exc:  # pragma: no cover - UI feedback
                self._log(f"Failed to generate answer: {exc}")
                
                def _handle_error() -> None:
                    messagebox.showerror(
                        "Question Error",
                        f"An error occurred while generating the answer.\n{exc}",
                    )
                    self._set_answer_text(
                        "An error occurred. Check the logs for details."
                    )
                    self.ask_button.configure(state=tk.NORMAL)

                self.root.after(0, _handle_error)
                return

            self._log("Answer ready.")

            def _handle_success() -> None:
                self._set_answer_text(answer)
                self.ask_button.configure(state=tk.NORMAL)

            self.root.after(0, _handle_success)

        self.qa_thread = threading.Thread(target=_run_qa, daemon=True)
        self.qa_thread.start()

    def _set_answer_text(self, text: str) -> None:
        self.answer_widget.configure(state=tk.NORMAL)
        self.answer_widget.delete("1.0", tk.END)
        self.answer_widget.insert(tk.END, text)
        self.answer_widget.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    VideoQAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

