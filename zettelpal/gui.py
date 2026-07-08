# gui.py - Zettelpal GUI
# Tkinter interface for the Zettelpal pipeline

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from zettelpal import config, naming, pipeline
from zettelpal.log import LOGGER_NAME, get_logger
from zettelpal.vault import linking

log = get_logger(__name__)


class TkLogHandler(logging.Handler):
    """Forwards log records to a Tkinter Text widget via a queue, so worker
    threads can log safely while the Tk main loop does the widget updates."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.queue = queue.Queue()
        self.update_interval = 100
        self.widget.after(self.update_interval, self._drain)

    def emit(self, record):
        self.queue.put(self.format(record) + "\n")

    def _drain(self):
        try:
            while True:
                text = self.queue.get_nowait()
                self.widget.configure(state='normal')
                self.widget.insert(tk.END, text)
                self.widget.see(tk.END)
                self.widget.configure(state='disabled')
        except queue.Empty:
            pass
        finally:
            self.widget.after(self.update_interval, self._drain)


class StreamToLogger:
    """File-like shim so third-party writes to stdout/stderr (Whisper progress,
    tracebacks) reach the log — and don't crash under pythonw, where the real
    streams are missing."""

    def __init__(self, level: int):
        self.level = level

    def write(self, text):
        if text and text.strip():
            log.log(self.level, text.rstrip())

    def flush(self):
        pass


class ZettelpalGUI(tk.Tk):
    """Main Zettelpal GUI application."""

    def __init__(self):
        super().__init__()
        self.title("Zettelpal")
        self.geometry("900x700")
        self.minsize(700, 500)

        # Set icon
        icon_path = os.path.join(os.path.dirname(__file__), "zettelpal.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except tk.TclError:
                pass

        self.audio_files = []
        self.processing_queue = queue.Queue()
        self.processing_thread = None

        self.create_widgets()
        self.attach_logging()
        self.check_directories()

    def check_directories(self):
        """Initial directory validation."""
        log.info("Checking directories...")
        try:
            os.makedirs(config.settings.data_dir, exist_ok=True)
            os.makedirs(config.settings.raw_transcripts_dir, exist_ok=True)
            os.makedirs(config.settings.segmented_output_dir, exist_ok=True)
            os.makedirs(config.settings.resolved_archive_dir, exist_ok=True)
            os.makedirs(config.settings.clips_dir, exist_ok=True)

            if not os.path.exists(config.settings.vault_root):
                log.error(f"ERROR: Obsidian vault not found: {config.settings.vault_root}")
                messagebox.showerror(
                    "Configuration Error",
                    f"Obsidian vault not found:\n{config.settings.vault_root}"
                )
                return False

            notes_dir = os.path.join(config.settings.vault_root, config.settings.notes_subdirectory)
            os.makedirs(notes_dir, exist_ok=True)

            cache_dir = os.path.dirname(config.settings.embeddings_cache_file)
            os.makedirs(cache_dir, exist_ok=True)

            log.info("All directories ready.")
            backend = config.settings.llm_backend
            log.info(f"LLM Backend: {backend}")
            if backend == "gemini":
                log.info(f"Gemini Model: {config.settings.gemini_model}")
            else:
                log.info(f"Local LLM: {config.settings.local_llm_base_url}")
                log.info(f"Model: {config.settings.local_llm_model}")
            return True

        except Exception as e:
            log.error(f"ERROR: Directory setup failed: {e}")
            messagebox.showerror("Setup Error", str(e))
            return False

    def create_widgets(self):
        """Create the main UI."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # === PIPELINE TAB ===
        pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(pipeline_tab, text="Pipeline")
        pipeline_tab.grid_rowconfigure(1, weight=1)
        pipeline_tab.grid_columnconfigure(0, weight=1)

        # File selection frame
        file_frame = ttk.LabelFrame(pipeline_tab, text="Audio Files", padding=10)
        file_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        file_frame.grid_columnconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.EXTENDED, height=4)
        self.file_listbox.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Button(file_frame, text="Add Files", command=self.add_files).grid(
            row=1, column=0, sticky="ew", padx=2
        )
        ttk.Button(file_frame, text="Remove Selected", command=self.remove_files).grid(
            row=1, column=1, sticky="ew", padx=2
        )
        ttk.Button(file_frame, text="Clear All", command=self.clear_files).grid(
            row=1, column=2, sticky="ew", padx=2
        )

        # Console frame
        console_frame = ttk.LabelFrame(pipeline_tab, text="Console Output", padding=10)
        console_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        console_frame.grid_rowconfigure(0, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(
            console_frame, wrap=tk.WORD, height=15, state='disabled',
            font=('Consolas', 9)
        )
        self.console_text.grid(row=0, column=0, sticky="nsew")

        # Control buttons
        control_frame = ttk.Frame(pipeline_tab, padding=5)
        control_frame.grid(row=2, column=0, sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)

        self.start_button = ttk.Button(
            control_frame, text="Start Pipeline", command=self.start_pipeline
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Button(
            control_frame, text="Clear Console", command=self.clear_console
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(
            control_frame, text="Quit", command=self.quit_app
        ).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # === SETTINGS TAB ===
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Settings")
        settings_tab.grid_columnconfigure(1, weight=1)

        row = 0

        # LLM Backend selection
        ttk.Label(settings_tab, text="LLM Backend:").grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )
        backend_frame = ttk.Frame(settings_tab)
        backend_frame.grid(row=row, column=1, sticky="ew", padx=10, pady=5)

        self.backend_var = tk.StringVar(value=config.settings.llm_backend)
        self.backend_combo = ttk.Combobox(
            backend_frame, textvariable=self.backend_var,
            values=["local", "gemini"], state="readonly", width=15
        )
        self.backend_combo.pack(side=tk.LEFT)
        self.backend_combo.bind("<<ComboboxSelected>>", self.on_backend_change)

        self.backend_status = ttk.Label(backend_frame, text="", foreground="gray")
        self.backend_status.pack(side=tk.LEFT, padx=(10, 0))
        self.update_backend_status()
        row += 1

        # Manual tags
        ttk.Label(settings_tab, text="Manual Tags:").grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )
        self.tags_entry = ttk.Entry(settings_tab)
        self.tags_entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
        row += 1

        # Similarity threshold
        ttk.Label(settings_tab, text="Similarity Threshold:").grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )

        threshold_frame = ttk.Frame(settings_tab)
        threshold_frame.grid(row=row, column=1, sticky="ew", padx=10, pady=5)

        self.threshold_var = tk.DoubleVar(value=config.settings.similarity_threshold)
        self.threshold_label = ttk.Label(
            threshold_frame, text=f"{config.settings.similarity_threshold:.2f}"
        )
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 10))

        self.threshold_slider = ttk.Scale(
            threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.threshold_var, command=self.update_threshold_label
        )
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        row += 1

        # Recalculate links button
        self.relink_button = ttk.Button(
            settings_tab, text="Recalculate All Links",
            command=self.recalculate_links
        )
        self.relink_button.grid(row=row, column=0, columnspan=2, padx=10, pady=20)
        row += 1

        # Info section
        ttk.Separator(settings_tab, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )
        row += 1

        info_text = f"""Configuration:
Vault: {config.settings.vault_root}
Local LLM: {config.settings.local_llm_base_url} ({config.settings.local_llm_model})
Gemini: {config.settings.gemini_model} {"(API key set)" if config.settings.gemini_api_key else "(no API key)"}
Whisper: {config.settings.whisper_model_size}
Embeddings: {config.settings.embedding_model}"""

        info_label = ttk.Label(
            settings_tab, text=info_text, justify=tk.LEFT,
            font=('Consolas', 9), foreground='gray'
        )
        info_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def attach_logging(self):
        """Send zettelpal log output to the console widget."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)
        handler = TkLogHandler(self.console_text)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        sys.stdout = StreamToLogger(logging.INFO)
        sys.stderr = StreamToLogger(logging.ERROR)
        log.info("Zettelpal ready.")

    def update_threshold_label(self, val=None):
        """Update the threshold display."""
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

    def on_backend_change(self, event=None):
        """Handle LLM backend selection change."""
        new_backend = self.backend_var.get()
        config.settings.llm_backend = new_backend
        self.update_backend_status()
        log.info(f"[SETTINGS] LLM backend changed to: {new_backend}")

    def update_backend_status(self):
        """Update the backend status indicator."""
        backend = self.backend_var.get()
        if backend == "gemini":
            if config.settings.gemini_api_key:
                self.backend_status.config(text="(API key set)", foreground="green")
            else:
                self.backend_status.config(text="(No API key!)", foreground="red")
        else:
            self.backend_status.config(text=f"({config.settings.local_llm_model})", foreground="gray")

    def add_files(self):
        """Add audio files to the queue."""
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.ogg *.wma"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=filetypes
        )

        for filepath in files:
            filename = os.path.basename(filepath)

            # Check/rename to Zettelpal format
            if not naming.is_valid_zettelpal_filename(filename):
                log.info(f"Renaming {filename} to Zettelpal format...")
                renamed = naming.rename_audio_file_to_zettelpal_format(filepath)
                if renamed:
                    filepath = renamed
                else:
                    log.info(f"Failed to rename {filename}. Skipping.")
                    continue

            if filepath not in self.audio_files:
                self.audio_files.append(filepath)
                self.file_listbox.insert(tk.END, os.path.basename(filepath))

        if files:
            log.info(f"Added {len(files)} file(s) to queue.")

    def remove_files(self):
        """Remove selected files from queue."""
        selected = self.file_listbox.curselection()
        for i in reversed(selected):
            self.file_listbox.delete(i)
            del self.audio_files[i]

    def clear_files(self):
        """Clear all files from queue."""
        self.file_listbox.delete(0, tk.END)
        self.audio_files.clear()

    def clear_console(self):
        """Clear the console output."""
        self.console_text.configure(state='normal')
        self.console_text.delete(1.0, tk.END)
        self.console_text.configure(state='disabled')

    def set_buttons_state(self, enabled: bool):
        """Enable or disable action buttons."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.start_button.config(state=state)
        self.relink_button.config(state=state)

    def start_pipeline(self):
        """Start processing the audio queue."""
        if not self.audio_files:
            messagebox.showinfo("No Files", "Please add audio files to process.")
            return

        manual_tags = self.tags_entry.get().strip()

        for filepath in self.audio_files:
            self.processing_queue.put((filepath, manual_tags))

        self.file_listbox.delete(0, tk.END)
        self.audio_files.clear()
        self.tags_entry.delete(0, tk.END)

        self.set_buttons_state(False)
        log.info("\n" + "=" * 50)
        log.info("STARTING PIPELINE")
        log.info("=" * 50)

        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.start()

    def _process_queue(self):
        """Process files in the queue (runs in thread)."""
        while not self.processing_queue.empty():
            filepath, tags = self.processing_queue.get()
            log.info(f"\nProcessing: {os.path.basename(filepath)}")
            if tags:
                log.info(f"Tags: {tags}")

            try:
                self._run_pipeline(filepath, tags)
            except Exception:
                log.exception("Pipeline failed")

        log.info("\n" + "=" * 50)
        log.info("ALL FILES PROCESSED")
        log.info("=" * 50)
        self.after(0, lambda: self.set_buttons_state(True))

    def _run_pipeline(self, audio_filepath: str, manual_tags: str) -> bool:
        """Run the shared pipeline so GUI and CLI behavior stay identical."""
        original_threshold = config.settings.similarity_threshold
        config.settings.similarity_threshold = self.threshold_var.get()
        try:
            return pipeline.run_pipeline(audio_filepath, manual_tags)
        finally:
            config.settings.similarity_threshold = original_threshold

    def recalculate_links(self):
        """Recalculate all semantic links."""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for current processing to finish.")
            return

        self.set_buttons_state(False)
        threshold = self.threshold_var.get()
        log.info(f"\nRecalculating all links (threshold: {threshold:.2f})...")

        self.processing_thread = threading.Thread(
            target=self._relink_thread, args=(threshold,)
        )
        self.processing_thread.start()

    def _relink_thread(self, threshold: float):
        """Run linking in a thread."""
        try:
            success = linking.run_linking_process(threshold)
            if success:
                log.info("Link recalculation complete.")
            else:
                log.info("Link recalculation failed.")
        except Exception:
            log.exception("Link recalculation failed")
        finally:
            self.after(0, lambda: self.set_buttons_state(True))

    def quit_app(self):
        """Quit the application."""
        if self.processing_thread and self.processing_thread.is_alive():
            if not messagebox.askyesno(
                "Confirm Quit",
                "Processing is running. Quit anyway?"
            ):
                return
        self.destroy()


def main():
    """GUI entry point for packaging."""
    app = ZettelpalGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
