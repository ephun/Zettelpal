# gui.py - Zettelpal desktop GUI (CustomTkinter).
#
# A control panel for the pipeline: queue recordings, watch them process in a
# live log, and tune linking. The mind-map itself is viewed in Obsidian.

import logging
import os
import queue
import sys
import threading

import customtkinter as ctk
from tkinter import filedialog, messagebox

from zettelpal import config, intake, naming, pipeline
from zettelpal.log import LOGGER_NAME, get_logger
from zettelpal.vault import linking

log = get_logger(__name__)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class TkLogHandler(logging.Handler):
    """Forwards log records to a text widget via a queue, so worker threads can
    log safely while the Tk main loop does the widget updates."""

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
                self.widget.configure(state="normal")
                self.widget.insert("end", text)
                self.widget.see("end")
                self.widget.configure(state="disabled")
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


class ZettelpalGUI(ctk.CTk):
    """Main Zettelpal GUI application."""

    def __init__(self):
        super().__init__()
        self.title("Zettelpal")
        self.geometry("960x720")
        self.minsize(780, 560)

        icon_path = os.path.join(os.path.dirname(__file__), "zettelpal.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        self.audio_files: list[str] = []
        self.processing_queue: queue.Queue = queue.Queue()
        self.processing_thread: threading.Thread | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._attach_logging()
        self._check_directories()

    # ------------------------------------------------------------------ UI

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Zettelpal",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Turn recordings into a linked Obsidian mind-map.",
            text_color=("gray50", "gray60"),
        ).grid(row=1, column=0, sticky="w")

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self, corner_radius=12)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)
        self.tabs.add("Pipeline")
        self.tabs.add("Settings")
        self._build_pipeline_tab(self.tabs.tab("Pipeline"))
        self._build_settings_tab(self.tabs.tab("Settings"))

    def _build_pipeline_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # --- Audio queue ---
        files_frame = ctk.CTkFrame(tab)
        files_frame.grid(row=0, column=0, sticky="ew", pady=(4, 10))
        files_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            files_frame, text="Audio queue",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        self.file_list = ctk.CTkScrollableFrame(files_frame, height=110)
        self.file_list.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=4)
        self.file_list.grid_columnconfigure(0, weight=1)
        self._empty_hint = ctk.CTkLabel(
            self.file_list, text="No files queued. Click “Add files…”.",
            text_color=("gray55", "gray55"),
        )
        self._empty_hint.grid(row=0, column=0, sticky="w", padx=4, pady=6)

        btns = ctk.CTkFrame(files_frame, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        ctk.CTkButton(btns, text="Add files…", width=110, command=self.add_files).pack(side="left")
        ctk.CTkButton(
            btns, text="Add folder…", width=110, command=self.add_folder,
            fg_color="transparent", border_width=1,
        ).pack(side="left", padx=(8, 0))
        ctk.CTkButton(
            btns, text="Clear", width=80, command=self.clear_files,
            fg_color="transparent", border_width=1,
        ).pack(side="left", padx=(8, 0))

        # --- Console ---
        console_frame = ctk.CTkFrame(tab)
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            console_frame, text="Activity",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        self.console = ctk.CTkTextbox(
            console_frame, state="disabled", wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        # --- Controls ---
        controls = ctk.CTkFrame(tab, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls.grid_columnconfigure(0, weight=1)

        self.start_button = ctk.CTkButton(
            controls, text="Start pipeline", height=40,
            font=ctk.CTkFont(size=14, weight="bold"), command=self.start_pipeline,
        )
        self.start_button.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            controls, text="Clear log", width=110, command=self.clear_console,
            fg_color="transparent", border_width=1,
        ).grid(row=0, column=1, padx=(10, 0))

    def _build_settings_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)
        pad = {"padx": 14, "pady": 8}
        row = 0

        ctk.CTkLabel(tab, text="LLM backend").grid(row=row, column=0, sticky="w", **pad)
        backend_row = ctk.CTkFrame(tab, fg_color="transparent")
        backend_row.grid(row=row, column=1, sticky="w", **pad)
        self.backend_var = ctk.StringVar(value=config.settings.llm_backend)
        ctk.CTkOptionMenu(
            backend_row, values=["local", "gemini"], variable=self.backend_var,
            width=140, command=self.on_backend_change,
        ).pack(side="left")
        self.backend_status = ctk.CTkLabel(backend_row, text="", text_color=("gray45", "gray60"))
        self.backend_status.pack(side="left", padx=(10, 0))
        row += 1

        ctk.CTkLabel(tab, text="Manual tags").grid(row=row, column=0, sticky="w", **pad)
        self.tags_entry = ctk.CTkEntry(tab, placeholder_text="comma,separated,tags")
        self.tags_entry.grid(row=row, column=1, sticky="ew", **pad)
        row += 1

        ctk.CTkLabel(tab, text="Similarity threshold").grid(row=row, column=0, sticky="w", **pad)
        threshold_row = ctk.CTkFrame(tab, fg_color="transparent")
        threshold_row.grid(row=row, column=1, sticky="ew", **pad)
        threshold_row.grid_columnconfigure(1, weight=1)
        self.threshold_var = ctk.DoubleVar(value=config.settings.similarity_threshold)
        self.threshold_label = ctk.CTkLabel(threshold_row, text=f"{self.threshold_var.get():.2f}", width=40)
        self.threshold_label.grid(row=0, column=0, padx=(0, 10))
        ctk.CTkSlider(
            threshold_row, from_=0.0, to=1.0, variable=self.threshold_var,
            command=self.update_threshold_label,
        ).grid(row=0, column=1, sticky="ew")
        row += 1

        ctk.CTkLabel(tab, text="Appearance").grid(row=row, column=0, sticky="w", **pad)
        ctk.CTkOptionMenu(
            tab, values=["System", "Light", "Dark"], width=140,
            command=ctk.set_appearance_mode,
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        self.relink_button = ctk.CTkButton(
            tab, text="Recalculate all links", command=self.recalculate_links,
        )
        self.relink_button.grid(row=row, column=0, columnspan=2, sticky="w", padx=14, pady=(18, 8))
        row += 1

        self.info_label = ctk.CTkLabel(
            tab, justify="left", text_color=("gray40", "gray60"),
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.info_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=14, pady=(16, 8))
        self._refresh_info()
        self.update_backend_status()

    def _refresh_info(self):
        gemini = "(API key set)" if config.settings.gemini_api_key else "(no API key)"
        self.info_label.configure(text="\n".join([
            f"Vault:      {config.settings.vault_root}",
            f"Local LLM:  {config.settings.local_llm_base_url} ({config.settings.local_llm_model})",
            f"Gemini:     {config.settings.gemini_model} {gemini}",
            f"Whisper:    {config.settings.whisper_model_size}",
            f"Embeddings: {config.settings.embedding_model}",
        ]))

    # -------------------------------------------------------------- logging

    def _attach_logging(self):
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)
        handler = TkLogHandler(self.console)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        sys.stdout = StreamToLogger(logging.INFO)
        sys.stderr = StreamToLogger(logging.ERROR)
        log.info("Zettelpal ready.")

    def _check_directories(self):
        try:
            for path in (
                config.settings.data_dir,
                config.settings.raw_transcripts_dir,
                config.settings.segmented_output_dir,
                config.settings.resolved_archive_dir,
                config.settings.clips_dir,
            ):
                os.makedirs(path, exist_ok=True)

            if not os.path.exists(config.settings.vault_root):
                log.error(f"Obsidian vault not found: {config.settings.vault_root}")
                messagebox.showerror(
                    "Configuration Error",
                    f"Obsidian vault not found:\n{config.settings.vault_root}\n\n"
                    "Set vault_root in zettelpal.toml or the ZETTELPAL_VAULT_ROOT "
                    "environment variable.",
                )
                return
            os.makedirs(config.settings.notes_dir, exist_ok=True)
            os.makedirs(os.path.dirname(config.settings.embeddings_cache_file), exist_ok=True)
            log.info(f"Ready. Backend: {config.settings.llm_backend}.")
        except OSError as e:
            log.error(f"Directory setup failed: {e}")
            messagebox.showerror("Setup Error", str(e))

    # ------------------------------------------------------------- queue UI

    def _render_file_list(self):
        for widget in self.file_list.winfo_children():
            widget.destroy()
        if not self.audio_files:
            self._empty_hint = ctk.CTkLabel(
                self.file_list, text="No files queued. Click “Add files…”.",
                text_color=("gray55", "gray55"),
            )
            self._empty_hint.grid(row=0, column=0, sticky="w", padx=4, pady=6)
            return
        for index, path in enumerate(self.audio_files):
            row = ctk.CTkFrame(self.file_list, fg_color="transparent")
            row.grid(row=index, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=os.path.basename(path), anchor="w").grid(
                row=0, column=0, sticky="ew", padx=(4, 0)
            )
            ctk.CTkButton(
                row, text="✕", width=28, height=24, fg_color="transparent",
                border_width=1, command=lambda p=path: self.remove_file(p),
            ).grid(row=0, column=1, padx=4)

    def _queue_path(self, filepath: str):
        filename = os.path.basename(filepath)
        if not naming.is_valid_zettelpal_filename(filename):
            log.info(f"Renaming {filename} to Zettelpal format...")
            renamed = naming.rename_audio_file_to_zettelpal_format(filepath)
            if not renamed:
                log.error(f"Failed to rename {filename}. Skipping.")
                return
            filepath = renamed
        if filepath not in self.audio_files:
            self.audio_files.append(filepath)

    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.m4a *.ogg *.wma *.aac *.webm *.mp4"),
                ("All files", "*.*"),
            ],
        )
        for filepath in files:
            self._queue_path(filepath)
        if files:
            self._render_file_list()
            log.info(f"Queued {len(files)} file(s).")

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of recordings")
        if not folder:
            return
        found = intake.scan_folder(folder)
        for filepath in found:
            self._queue_path(filepath)
        self._render_file_list()
        log.info(f"Queued {len(found)} file(s) from folder.")

    def remove_file(self, path: str):
        if path in self.audio_files:
            self.audio_files.remove(path)
            self._render_file_list()

    def clear_files(self):
        self.audio_files.clear()
        self._render_file_list()

    def clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # --------------------------------------------------------------- events

    def update_threshold_label(self, _=None):
        self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}")

    def on_backend_change(self, _=None):
        config.settings.llm_backend = self.backend_var.get()
        self.update_backend_status()
        self._refresh_info()
        log.info(f"[SETTINGS] LLM backend changed to: {self.backend_var.get()}")

    def update_backend_status(self):
        if self.backend_var.get() == "gemini":
            if config.settings.gemini_api_key:
                self.backend_status.configure(text="API key set", text_color=("green", "#4ade80"))
            else:
                self.backend_status.configure(text="No API key!", text_color=("#b91c1c", "#f87171"))
        else:
            self.backend_status.configure(
                text=config.settings.local_llm_model, text_color=("gray45", "gray60")
            )

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.start_button.configure(state=state)
        self.relink_button.configure(state=state)

    # -------------------------------------------------------------- actions

    def start_pipeline(self):
        if not self.audio_files:
            messagebox.showinfo("No files", "Add audio files to process first.")
            return
        manual_tags = self.tags_entry.get().strip()
        for filepath in self.audio_files:
            self.processing_queue.put((filepath, manual_tags))
        self.clear_files()
        self.tags_entry.delete(0, "end")

        self._set_busy(True)
        log.info("\n" + "=" * 50 + "\nSTARTING PIPELINE\n" + "=" * 50)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

    def _process_queue(self):
        while not self.processing_queue.empty():
            filepath, tags = self.processing_queue.get()
            log.info(f"\nProcessing: {os.path.basename(filepath)}")
            try:
                self._run_pipeline(filepath, tags)
            except Exception:
                log.exception("Pipeline failed")
        log.info("\n" + "=" * 50 + "\nALL FILES PROCESSED\n" + "=" * 50)
        self.after(0, lambda: self._set_busy(False))

    def _run_pipeline(self, audio_filepath: str, manual_tags: str) -> bool:
        """Run the shared pipeline so GUI and CLI behavior stay identical."""
        original = config.settings.similarity_threshold
        config.settings.similarity_threshold = self.threshold_var.get()
        try:
            return pipeline.run_pipeline(audio_filepath, manual_tags)
        finally:
            config.settings.similarity_threshold = original

    def recalculate_links(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "Wait for the current run to finish.")
            return
        self._set_busy(True)
        threshold = self.threshold_var.get()
        log.info(f"\nRecalculating all links (threshold: {threshold:.2f})...")
        self.processing_thread = threading.Thread(
            target=self._relink_thread, args=(threshold,), daemon=True
        )
        self.processing_thread.start()

    def _relink_thread(self, threshold: float):
        try:
            if linking.run_linking_process(threshold):
                log.info("Link recalculation complete.")
            else:
                log.info("Link recalculation failed.")
        except Exception:
            log.exception("Link recalculation failed")
        finally:
            self.after(0, lambda: self._set_busy(False))

    def quit_app(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if not messagebox.askyesno("Confirm quit", "Processing is running. Quit anyway?"):
                return
        self.destroy()


def main():
    """GUI entry point for packaging."""
    app = ZettelpalGUI()
    app.protocol("WM_DELETE_WINDOW", app.quit_app)
    app.mainloop()


if __name__ == "__main__":
    main()
