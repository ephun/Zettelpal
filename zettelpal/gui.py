# gui.py - Zettelpal desktop GUI (CustomTkinter).
#
# A control panel for the pipeline: queue recordings, watch them process in a
# live log, and edit settings. The mind-map itself is viewed in Obsidian.

import io
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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Palette as (light, dark) tuples so both themes are deliberate, not inverted.
ACCENT = ("#4f46e5", "#7d79ff")
ACCENT_HOVER = ("#4338ca", "#6b66f2")
WINDOW_BG = ("#eceef2", "#0f1116")
SIDEBAR_BG = ("#ffffff", "#171a21")
CARD_BG = ("#ffffff", "#1b1e26")
FIELD_BG = ("#f4f5f8", "#242834")
FIELD_HOVER = ("#eceef3", "#2b303c")
TEXT = ("#1b1e26", "#e7e9ef")
MUTED = ("#6b7280", "#949bab")
BORDER = ("#e3e6eb", "#2b2f3a")

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


class TkLogHandler(logging.Handler):
    """Forwards log records to a text widget via a queue, so worker threads can
    log safely while the Tk main loop does the widget updates."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.update_interval = 100
        self.queue: queue.Queue = queue.Queue()
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


class StreamToLogger(io.TextIOBase):
    """File-like shim so third-party writes to stdout/stderr (Whisper progress,
    tracebacks) reach the log — and don't crash under pythonw, where the real
    streams are missing.

    Subclasses TextIOBase so libraries that probe the stream get real answers
    instead of AttributeError: tqdm (via transformers/sentence-transformers)
    calls isatty(), others call fileno() or check encoding.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, level: int):
        self.level = level

    def write(self, text):
        if text and text.strip():
            log.log(self.level, text.rstrip())
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False

    def writable(self):
        return True


class ZettelpalGUI(ctk.CTk):
    """Main Zettelpal GUI application."""

    def __init__(self):
        super().__init__()
        self.title("Zettelpal")
        self.geometry("1040x720")
        self.minsize(900, 600)
        self.configure(fg_color=WINDOW_BG)

        icon_path = os.path.join(os.path.dirname(__file__), "zettelpal.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        self.audio_files: list[str] = []
        self.processing_queue: queue.Queue = queue.Queue()
        self.processing_thread: threading.Thread | None = None
        self.insights_thread: threading.Thread | None = None
        self.threshold_var = ctk.DoubleVar(value=config.settings.similarity_threshold)
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self._setting_vars: dict[str, ctk.Variable] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_pipeline_view()
        self._build_insights_view()
        self._build_settings_view()
        self._attach_logging()
        self.show_view("pipeline")
        self._check_directories()

    # -- helpers ---------------------------------------------------------

    def _card(self, parent) -> ctk.CTkFrame:
        return ctk.CTkFrame(parent, fg_color=CARD_BG, border_width=1,
                            border_color=BORDER, corner_radius=12)

    def _section_label(self, parent, text) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=text, text_color=MUTED, anchor="w",
                            font=ctk.CTkFont(size=12, weight="bold"))

    def _secondary_button(self, parent, text, command, width=120) -> ctk.CTkButton:
        # Always visible at rest: filled field color + border, not transparent.
        return ctk.CTkButton(
            parent, text=text, command=command, width=width, height=34,
            fg_color=FIELD_BG, hover_color=FIELD_HOVER, text_color=TEXT,
            border_width=1, border_color=BORDER, corner_radius=8,
        )

    # -- sidebar ---------------------------------------------------------

    def _build_sidebar(self):
        bar = ctk.CTkFrame(self, width=212, corner_radius=0, fg_color=SIDEBAR_BG)
        bar.grid(row=0, column=0, sticky="nsw")
        bar.grid_propagate(False)
        bar.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            bar, text="Zettelpal", anchor="w",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="ew", padx=20, pady=(22, 0))
        ctk.CTkLabel(
            bar, text="audio → mind-map", anchor="w",
            font=ctk.CTkFont(size=12), text_color=MUTED,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))

        nav = ctk.CTkFrame(bar, fg_color="transparent")
        nav.grid(row=2, column=0, sticky="ew", padx=12)
        nav_items = (("pipeline", "  Pipeline"), ("insights", "  Insights"), ("settings", "  Settings"))
        for i, (key, label) in enumerate(nav_items):
            btn = ctk.CTkButton(
                nav, text=label, anchor="w", height=40, corner_radius=8,
                fg_color="transparent", text_color=TEXT, hover_color=FIELD_HOVER,
                font=ctk.CTkFont(size=14), command=lambda k=key: self.show_view(k),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=3)
            nav.grid_columnconfigure(0, weight=1)
            self.nav_buttons[key] = btn

        footer = ctk.CTkFrame(bar, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", padx=20, pady=18)
        ctk.CTkLabel(footer, text="Appearance", text_color=MUTED,
                     font=ctk.CTkFont(size=11), anchor="w").pack(anchor="w")
        ctk.CTkOptionMenu(
            footer, values=["Dark", "Light", "System"], width=160, height=30,
            fg_color=FIELD_BG, button_color=FIELD_BG, button_hover_color=FIELD_HOVER,
            text_color=TEXT, command=ctk.set_appearance_mode,
        ).pack(anchor="w", pady=(4, 0))

    # -- pipeline view ---------------------------------------------------

    def _build_pipeline_view(self):
        view = ctk.CTkFrame(self, fg_color="transparent")
        view.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(2, weight=1)
        self.pipeline_view = view

        ctk.CTkLabel(view, text="Pipeline", anchor="w", text_color=TEXT,
                     font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 16))

        # Queue card
        queue_card = self._card(view)
        queue_card.grid(row=1, column=0, sticky="ew")
        queue_card.grid_columnconfigure(0, weight=1)
        self._section_label(queue_card, "AUDIO QUEUE").grid(
            row=0, column=0, sticky="w", padx=18, pady=(14, 6))
        self.file_list = ctk.CTkScrollableFrame(
            queue_card, height=118, fg_color=FIELD_BG, corner_radius=8)
        self.file_list.grid(row=1, column=0, sticky="ew", padx=14, pady=2)
        self.file_list.grid_columnconfigure(0, weight=1)

        actions = ctk.CTkFrame(queue_card, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=14, pady=(10, 14))
        ctk.CTkButton(actions, text="Add files", width=110, height=34, corner_radius=8,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.add_files).pack(side="left")
        self._secondary_button(actions, "Add folder", self.add_folder).pack(side="left", padx=8)
        self._secondary_button(actions, "Clear", self.clear_files, width=90).pack(side="left")

        # Activity card
        log_card = self._card(view)
        log_card.grid(row=2, column=0, sticky="nsew", pady=(16, 0))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)
        self._section_label(log_card, "ACTIVITY").grid(
            row=0, column=0, sticky="w", padx=18, pady=(14, 6))
        self.console = ctk.CTkTextbox(
            log_card, state="disabled", wrap="word", fg_color=FIELD_BG,
            corner_radius=8, font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        # Controls
        controls = ctk.CTkFrame(view, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        controls.grid_columnconfigure(0, weight=1)
        self.start_button = ctk.CTkButton(
            controls, text="Start pipeline", height=44, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=15, weight="bold"), command=self.start_pipeline)
        self.start_button.grid(row=0, column=0, sticky="ew")
        self._secondary_button(controls, "Clear log", self.clear_console, width=110).grid(
            row=0, column=1, padx=(12, 0))

    # -- insights view ---------------------------------------------------

    def _build_insights_view(self):
        view = ctk.CTkFrame(self, fg_color="transparent")
        view.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)
        self.insights_view = view
        self.insight_buttons: list[ctk.CTkButton] = []

        ctk.CTkLabel(view, text="Insights", anchor="w", text_color=TEXT,
                     font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 16))

        # Ask-yourself card
        ask_card = self._card(view)
        ask_card.grid(row=1, column=0, sticky="nsew")
        ask_card.grid_columnconfigure(0, weight=1)
        ask_card.grid_rowconfigure(2, weight=1)
        self._section_label(ask_card, "ASK YOURSELF").grid(
            row=0, column=0, sticky="w", padx=18, pady=(14, 6))

        ask_row = ctk.CTkFrame(ask_card, fg_color="transparent")
        ask_row.grid(row=1, column=0, sticky="ew", padx=14)
        ask_row.grid_columnconfigure(0, weight=1)
        self.ask_entry = self._make_entry(ask_row, "")
        self.ask_entry.grid(row=0, column=0, sticky="ew")
        self.ask_entry.bind("<Return>", lambda _e: self.ask_question())
        self.ask_button = ctk.CTkButton(
            ask_row, text="Ask", width=90, height=34, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.ask_question)
        self.ask_button.grid(row=0, column=1, padx=(8, 0))
        self.insight_buttons.append(self.ask_button)

        self.answer_box = ctk.CTkTextbox(
            ask_card, state="disabled", wrap="word", fg_color=FIELD_BG,
            corner_radius=8, font=ctk.CTkFont(size=13))
        self.answer_box.grid(row=2, column=0, sticky="nsew", padx=14, pady=(10, 14))
        self._set_answer("Ask a question and it's answered from your own notes, "
                         "with the notes it drew on listed underneath.")

        # Generate card
        gen_card = self._card(view)
        gen_card.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        gen_card.grid_columnconfigure(0, weight=1)
        self._section_label(gen_card, "GENERATE").grid(
            row=0, column=0, sticky="w", padx=18, pady=(14, 4))
        ctk.CTkLabel(gen_card,
                     text=f"Written to your vault's “{config.settings.insights_subdirectory}” "
                          "folder — kept apart from your own notes.",
                     text_color=MUTED, anchor="w", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="w", padx=18, pady=(0, 8))

        btn_row = ctk.CTkFrame(gen_card, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        for text, command in (("Themes", self.gen_themes),
                              ("Weekly digest", self.gen_digest),
                              ("Resurface", self.gen_resurface)):
            btn = self._secondary_button(btn_row, text, command, width=150)
            btn.pack(side="left", padx=(0, 8))
            self.insight_buttons.append(btn)

        self.insight_status = ctk.CTkLabel(gen_card, text="", text_color=MUTED,
                                           font=ctk.CTkFont(size=12), anchor="w")
        self.insight_status.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 14))

    def _set_answer(self, text: str):
        self.answer_box.configure(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.insert("1.0", text)
        self.answer_box.configure(state="disabled")

    def _insight_status(self, text: str, kind: str = "info"):
        colors = {
            "info": MUTED,
            "ok": ("#15803d", "#4ade80"),
            "warn": ("#b45309", "#fbbf24"),
            "error": ("#b91c1c", "#f87171"),
        }
        self.insight_status.configure(text=text, text_color=colors.get(kind, MUTED))

    def _set_insights_busy(self, busy: bool):
        for btn in self.insight_buttons:
            btn.configure(state="disabled" if busy else "normal")

    def ask_question(self):
        question = self.ask_entry.get().strip()
        if not question:
            return
        if self._busy():
            messagebox.showinfo("Busy", "Wait for the current task to finish.")
            return
        self._set_insights_busy(True)
        self._insight_status("Thinking…")
        self._set_answer("Searching your notes…")

        def work():
            try:
                from zettelpal.insights import rag

                result = rag.ask(question)
                text = result["answer"]
                if result["sources"]:
                    text += "\n\nSources:\n" + "\n".join(
                        f"  • {note['title']}" for note in result["sources"])
                self.after(0, lambda: (self._set_answer(text), self._insight_status("")))
            except Exception:
                log.exception("Ask failed")
                self.after(0, lambda: (
                    self._set_answer("Something went wrong — see the Pipeline activity log."),
                    self._insight_status("Ask failed.", "error")))
            finally:
                self.after(0, lambda: self._set_insights_busy(False))

        self.insights_thread = threading.Thread(target=work, daemon=True)
        self.insights_thread.start()

    def gen_themes(self):
        from zettelpal.insights import themes
        self._run_generate(themes.generate_themes, "themes")

    def gen_digest(self):
        from zettelpal.insights import digest
        self._run_generate(digest.generate_digest, "weekly digest")

    def gen_resurface(self):
        from zettelpal.insights import resurfacing
        self._run_generate(resurfacing.resurface, "resurfaced notes")

    def _run_generate(self, fn, label: str):
        if self._busy():
            messagebox.showinfo("Busy", "Wait for the current task to finish.")
            return
        self._set_insights_busy(True)
        self._insight_status(f"Generating {label}…")

        def work():
            try:
                path = fn()
                if path:
                    self.after(0, lambda: self._insight_status(
                        f"Wrote {os.path.basename(path)}.", "ok"))
                else:
                    self.after(0, lambda: self._insight_status(
                        f"No {label} yet — you may need more notes.", "warn"))
            except Exception:
                log.exception("%s generation failed", label)
                self.after(0, lambda: self._insight_status(
                    f"{label.capitalize()} failed — see the Pipeline activity log.", "error"))
            finally:
                self.after(0, lambda: self._set_insights_busy(False))

        self.insights_thread = threading.Thread(target=work, daemon=True)
        self.insights_thread.start()

    # -- settings view ---------------------------------------------------

    def _build_settings_view(self):
        view = ctk.CTkFrame(self, fg_color="transparent")
        view.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)
        self.settings_view = view

        ctk.CTkLabel(view, text="Settings", anchor="w", text_color=TEXT,
                     font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 16))

        form = ctk.CTkScrollableFrame(view, fg_color="transparent")
        form.grid(row=1, column=0, sticky="nsew")
        form.grid_columnconfigure(0, weight=1)

        # --- Vault group ---
        vault_card = self._card(form)
        vault_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        vault_card.grid_columnconfigure(0, weight=1)
        self._section_label(vault_card, "VAULT").grid(
            row=0, column=0, sticky="w", padx=18, pady=(14, 8))
        vault_row = ctk.CTkFrame(vault_card, fg_color="transparent")
        vault_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        vault_row.grid_columnconfigure(0, weight=1)
        self.vault_entry = self._make_entry(vault_row, config.settings.vault_root)
        self.vault_entry.grid(row=0, column=0, sticky="ew")
        self._setting_vars["vault_root"] = self.vault_entry
        self._secondary_button(vault_row, "Browse", self._browse_vault, width=90).grid(
            row=0, column=1, padx=(8, 0))
        self._labeled_entry(vault_card, 2, "Notes subfolder (blank = vault root)",
                            "notes_subdirectory", config.settings.notes_subdirectory)

        # --- Models group ---
        models_card = self._card(form)
        models_card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        models_card.grid_columnconfigure(1, weight=1)
        self._section_label(models_card, "MODELS & BACKEND").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 8))

        self._form_label(models_card, 1, "LLM backend")
        self.backend_var = ctk.StringVar(value=config.settings.llm_backend)
        ctk.CTkOptionMenu(
            models_card, values=["local", "gemini"], variable=self.backend_var,
            width=160, fg_color=FIELD_BG, button_color=FIELD_BG,
            button_hover_color=FIELD_HOVER, text_color=TEXT,
            command=self.on_backend_change,
        ).grid(row=1, column=1, sticky="w", padx=(0, 18), pady=6)
        self._setting_vars["llm_backend"] = self.backend_var

        self._grid_entry(models_card, 2, "Local LLM URL", "local_llm_base_url",
                         config.settings.local_llm_base_url)
        self._grid_entry(models_card, 3, "Local LLM model", "local_llm_model",
                         config.settings.local_llm_model)
        self._grid_entry(models_card, 4, "Gemini model", "gemini_model",
                         config.settings.gemini_model)

        self._form_label(models_card, 5, "Whisper model")
        self.whisper_var = ctk.StringVar(value=config.settings.whisper_model_size)
        ctk.CTkOptionMenu(
            models_card, values=WHISPER_SIZES, variable=self.whisper_var, width=160,
            fg_color=FIELD_BG, button_color=FIELD_BG, button_hover_color=FIELD_HOVER,
            text_color=TEXT,
        ).grid(row=5, column=1, sticky="w", padx=(0, 18), pady=6)
        self._setting_vars["whisper_model_size"] = self.whisper_var

        self._grid_entry(models_card, 6, "Embedding model", "embedding_model",
                         config.settings.embedding_model)

        self.gemini_status = ctk.CTkLabel(models_card, text="", text_color=MUTED,
                                          font=ctk.CTkFont(size=12), anchor="w")
        self.gemini_status.grid(row=7, column=0, columnspan=2, sticky="w",
                                padx=18, pady=(2, 14))

        # --- Linking group ---
        link_card = self._card(form)
        link_card.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        link_card.grid_columnconfigure(1, weight=1)
        self._section_label(link_card, "LINKING").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 8))

        self._form_label(link_card, 1, "Similarity threshold")
        thresh_row = ctk.CTkFrame(link_card, fg_color="transparent")
        thresh_row.grid(row=1, column=1, sticky="ew", padx=(0, 18), pady=6)
        thresh_row.grid_columnconfigure(1, weight=1)
        self.threshold_label = ctk.CTkLabel(thresh_row, text=f"{self.threshold_var.get():.2f}",
                                            width=42, text_color=TEXT)
        self.threshold_label.grid(row=0, column=0, padx=(0, 10))
        ctk.CTkSlider(thresh_row, from_=0.0, to=1.0, variable=self.threshold_var,
                      button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                      progress_color=ACCENT, command=self.update_threshold_label).grid(
            row=0, column=1, sticky="ew")

        self._grid_entry(link_card, 2, "Max semantic links / note",
                         "max_semantic_links_per_note",
                         str(config.settings.max_semantic_links_per_note))
        self.relink_button = self._secondary_button(
            link_card, "Recalculate all links", self.recalculate_links, width=190)
        self.relink_button.grid(row=3, column=0, columnspan=2, sticky="w",
                                padx=18, pady=(4, 14))

        # --- Save bar ---
        save_bar = ctk.CTkFrame(form, fg_color="transparent")
        save_bar.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ctk.CTkButton(save_bar, text="Save settings", height=40, corner_radius=10,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER, width=150,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      command=self.save_settings).pack(side="left")
        self.save_status = ctk.CTkLabel(save_bar, text="", text_color=MUTED,
                                        font=ctk.CTkFont(size=12))
        self.save_status.pack(side="left", padx=14)

        self.update_backend_status()

    # small form builders

    def _make_entry(self, parent, value) -> ctk.CTkEntry:
        entry = ctk.CTkEntry(parent, height=34, corner_radius=8, fg_color=FIELD_BG,
                             border_color=BORDER, text_color=TEXT)
        if value:
            entry.insert(0, str(value))
        return entry

    def _form_label(self, parent, row, text):
        ctk.CTkLabel(parent, text=text, text_color=TEXT, anchor="w").grid(
            row=row, column=0, sticky="w", padx=18, pady=6)

    def _grid_entry(self, parent, row, label, key, value):
        self._form_label(parent, row, label)
        entry = self._make_entry(parent, value)
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=6)
        self._setting_vars[key] = entry

    def _labeled_entry(self, parent, row, label, key, value):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 12))
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(wrap, text=label, text_color=MUTED, anchor="w",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=(0, 2))
        entry = self._make_entry(wrap, value)
        entry.grid(row=1, column=0, sticky="ew")
        self._setting_vars[key] = entry

    # -- view switching --------------------------------------------------

    def show_view(self, name: str):
        views = {
            "settings": self.settings_view,
            "insights": self.insights_view,
            "pipeline": self.pipeline_view,
        }
        views.get(name, self.pipeline_view).tkraise()
        for key, btn in self.nav_buttons.items():
            if key == name:
                btn.configure(fg_color=ACCENT, text_color=("#ffffff", "#ffffff"))
            else:
                btn.configure(fg_color="transparent", text_color=TEXT)

    # -- logging & setup -------------------------------------------------

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
                messagebox.showwarning(
                    "Vault not found",
                    f"Obsidian vault not found:\n{config.settings.vault_root}\n\n"
                    "Set it in the Settings tab and click Save.",
                )
                self.show_view("settings")
                return
            os.makedirs(config.settings.notes_dir, exist_ok=True)
            os.makedirs(os.path.dirname(config.settings.embeddings_cache_file), exist_ok=True)
            log.info(f"Ready. Backend: {config.settings.llm_backend}.")
        except OSError as e:
            log.error(f"Directory setup failed: {e}")

    # -- queue UI --------------------------------------------------------

    def _render_file_list(self):
        for widget in self.file_list.winfo_children():
            widget.destroy()
        if not self.audio_files:
            ctk.CTkLabel(self.file_list, text="No files queued — add files or a folder.",
                         text_color=MUTED, anchor="w").grid(
                row=0, column=0, sticky="w", padx=6, pady=8)
            return
        for index, path in enumerate(self.audio_files):
            row = ctk.CTkFrame(self.file_list, fg_color="transparent")
            row.grid(row=index, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=os.path.basename(path), anchor="w",
                         text_color=TEXT).grid(row=0, column=0, sticky="ew", padx=(6, 0))
            ctk.CTkButton(row, text="✕", width=28, height=26, corner_radius=6,
                          fg_color=FIELD_HOVER, hover_color=("#e5484d", "#e5484d"),
                          text_color=TEXT, command=lambda p=path: self.remove_file(p)).grid(
                row=0, column=1, padx=4)

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
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.m4a *.ogg *.wma *.aac *.webm *.mp4"),
                       ("All files", "*.*")],
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

    # -- settings events -------------------------------------------------

    def update_threshold_label(self, _=None):
        self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}")

    def on_backend_change(self, _=None):
        config.settings.llm_backend = self.backend_var.get()
        self.update_backend_status()

    def update_backend_status(self):
        if self.backend_var.get() == "gemini":
            if config.settings.gemini_api_key:
                self.gemini_status.configure(
                    text="Gemini API key detected in environment.",
                    text_color=("#15803d", "#4ade80"))
            else:
                self.gemini_status.configure(
                    text="No Gemini API key — set GEMINI_API_KEY in your environment.",
                    text_color=("#b91c1c", "#f87171"))
        else:
            self.gemini_status.configure(
                text="Local backend — nothing leaves your machines.", text_color=MUTED)

    def save_settings(self):
        values = {}
        for key, var in self._setting_vars.items():
            raw = var.get().strip() if hasattr(var, "get") else var
            values[key] = raw
        values["similarity_threshold"] = round(self.threshold_var.get(), 3)
        try:
            values["max_semantic_links_per_note"] = int(
                values.get("max_semantic_links_per_note", 10))
        except (TypeError, ValueError):
            self.save_status.configure(text="Max links must be a whole number.",
                                       text_color=("#b91c1c", "#f87171"))
            return
        try:
            path = config.write_user_settings(values)
        except OSError as e:
            self.save_status.configure(text=f"Could not save: {e}",
                                       text_color=("#b91c1c", "#f87171"))
            return
        self.update_backend_status()
        self.save_status.configure(text=f"Saved to {os.path.basename(path)}.",
                                   text_color=("#15803d", "#4ade80"))
        log.info(f"[SETTINGS] Saved to {path}")

    def _browse_vault(self):
        folder = filedialog.askdirectory(title="Select your Obsidian vault")
        if folder:
            self.vault_entry.delete(0, "end")
            self.vault_entry.insert(0, folder)

    # -- actions ---------------------------------------------------------

    def _busy(self) -> bool:
        for thread in (self.processing_thread, self.insights_thread):
            if thread and thread.is_alive():
                return True
        return False

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.start_button.configure(state=state)
        self.relink_button.configure(state=state)

    def start_pipeline(self):
        if not self.audio_files:
            messagebox.showinfo("No files", "Add audio files to process first.")
            return
        if self._busy():
            messagebox.showinfo("Busy", "Wait for the current task to finish.")
            return
        manual_tags = ""  # manual tags removed from this view; tag in Obsidian
        for filepath in self.audio_files:
            self.processing_queue.put((filepath, manual_tags))
        self.clear_files()
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
        if self._busy():
            messagebox.showinfo("Busy", "Wait for the current task to finish.")
            return
        self._set_busy(True)
        threshold = self.threshold_var.get()
        log.info(f"\nRecalculating all links (threshold: {threshold:.2f})...")
        self.processing_thread = threading.Thread(
            target=self._relink_thread, args=(threshold,), daemon=True)
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
        if self._busy():
            if not messagebox.askyesno("Confirm quit", "A task is running. Quit anyway?"):
                return
        self.destroy()


def main():
    """GUI entry point for packaging."""
    app = ZettelpalGUI()
    app.protocol("WM_DELETE_WINDOW", app.quit_app)
    app.mainloop()


if __name__ == "__main__":
    main()
