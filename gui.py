# gui.py - Zettelpal GUI
# Modern tkinter interface for the Zettelpal pipeline

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import queue
import os
import sys
import datetime
import re
import shutil
import json

import config
import transcribe
import segment
import create_notes
import link_notes
import utils
import classify


class TextRedirector:
    """Redirects stdout/stderr to a Tkinter Text widget."""

    def __init__(self, widget):
        self.widget = widget
        self.queue = queue.Queue()
        self.update_interval = 100
        self.widget.after(self.update_interval, self.update_text)

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

    def update_text(self):
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
            self.widget.after(self.update_interval, self.update_text)


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
        self.redirect_stdout()
        self.check_directories()

    def check_directories(self):
        """Initial directory validation."""
        print("Checking directories...")
        try:
            os.makedirs(config.ZETTELPAL_ROOT, exist_ok=True)
            os.makedirs(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, exist_ok=True)
            os.makedirs(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, exist_ok=True)
            os.makedirs(config.ARCHIVE_DIR, exist_ok=True)

            if not os.path.exists(config.OBSIDIAN_VAULT_ROOT):
                print(f"ERROR: Obsidian vault not found: {config.OBSIDIAN_VAULT_ROOT}")
                messagebox.showerror(
                    "Configuration Error",
                    f"Obsidian vault not found:\n{config.OBSIDIAN_VAULT_ROOT}"
                )
                return False

            notes_dir = os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)
            os.makedirs(notes_dir, exist_ok=True)

            cache_dir = os.path.dirname(config.EMBEDDINGS_CACHE_FILE)
            os.makedirs(cache_dir, exist_ok=True)

            print("All directories ready.")
            print(f"LLM: {config.LOCAL_LLM_BASE_URL}")
            print(f"Model: {config.LOCAL_LLM_MODEL}")
            return True

        except Exception as e:
            print(f"ERROR: Directory setup failed: {e}")
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

        self.threshold_var = tk.DoubleVar(value=config.SIMILARITY_THRESHOLD)
        self.threshold_label = ttk.Label(
            threshold_frame, text=f"{config.SIMILARITY_THRESHOLD:.2f}"
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
Vault: {config.OBSIDIAN_VAULT_ROOT}
LLM: {config.LOCAL_LLM_BASE_URL}
Model: {config.LOCAL_LLM_MODEL}
Whisper: {config.LOCAL_WHISPER_MODEL_SIZE}
Embeddings: {config.LOCAL_EMBEDDING_MODEL}"""

        info_label = ttk.Label(
            settings_tab, text=info_text, justify=tk.LEFT,
            font=('Consolas', 9), foreground='gray'
        )
        info_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def redirect_stdout(self):
        """Redirect stdout and stderr to console."""
        self.console_text.configure(state='normal')
        sys.stdout = TextRedirector(self.console_text)
        sys.stderr = TextRedirector(self.console_text)
        print("Zettelpal ready.")

    def update_threshold_label(self, val=None):
        """Update the threshold display."""
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

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
            if not utils.is_valid_zettelpal_filename(filename):
                print(f"Renaming {filename} to Zettelpal format...")
                renamed = utils.rename_audio_file_to_zettelpal_format(filepath)
                if renamed:
                    filepath = renamed
                else:
                    print(f"Failed to rename {filename}. Skipping.")
                    continue

            if filepath not in self.audio_files:
                self.audio_files.append(filepath)
                self.file_listbox.insert(tk.END, os.path.basename(filepath))

        if files:
            print(f"Added {len(files)} file(s) to queue.")

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
        print("\n" + "=" * 50)
        print("STARTING PIPELINE")
        print("=" * 50)

        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.start()

    def _process_queue(self):
        """Process files in the queue (runs in thread)."""
        while not self.processing_queue.empty():
            filepath, tags = self.processing_queue.get()
            print(f"\nProcessing: {os.path.basename(filepath)}")
            if tags:
                print(f"Tags: {tags}")

            try:
                self._run_pipeline(filepath, tags)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 50)
        print("ALL FILES PROCESSED")
        print("=" * 50)
        self.after(0, lambda: self.set_buttons_state(True))

    def _run_pipeline(self, audio_filepath: str, manual_tags: str) -> bool:
        """Run the full pipeline for one audio file."""
        audio_basename = os.path.basename(audio_filepath)
        audio_stem = os.path.splitext(audio_basename)[0]
        audio_ext = os.path.splitext(audio_basename)[1].lstrip('.')

        if not re.match(r"^\d{6,8}$", audio_stem):
            print(f"ERROR: Invalid filename format: {audio_stem}")
            return False

        recording_id = audio_stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        raw_path = os.path.join(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, f"{recording_id}.txt")
        seg_path = os.path.join(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, f"{recording_id}_segmented.json")
        arch_audio = os.path.join(config.ARCHIVE_DIR, f"{recording_id}_{timestamp}.{audio_ext}")
        arch_transcript = os.path.join(config.ARCHIVE_DIR, f"{recording_id}_{timestamp}.txt")

        # Step 1: Transcribe
        print("\n--- Step 1: Transcribe ---")
        result = transcribe.transcribe_audio_to_file(audio_filepath, raw_path)
        if result is None:
            print("ERROR: Transcription failed.")
            return False

        # Step 2: Classify
        print("\n--- Step 2: Classify ---")
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_transcript = f.read()
        except Exception as e:
            print(f"ERROR: Could not read transcript: {e}")
            return False

        transcript_type = classify.classify_transcript(raw_transcript)
        if transcript_type is None:
            print("ERROR: Classification failed.")
            self._cleanup(raw_path)
            return False
        print(f"Type: {transcript_type}")

        # Step 3: Segment
        print("\n--- Step 3: Segment ---")
        segments = segment.segment_transcript(raw_transcript)
        if not segments:
            print("ERROR: Segmentation failed.")
            self._cleanup(raw_path)
            return False

        # Save segments
        try:
            with open(seg_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)
        except:
            pass

        # Step 4: Create Notes
        print("\n--- Step 4: Create Notes ---")
        notes_info = create_notes.create_notes_from_segments(
            segments,
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            config.NOTES_SUBDIRECTORY_IN_VAULT,
            recording_id,
            transcript_type,
            manual_tags
        )
        print(f"Created {len(notes_info)} notes.")

        # Step 5: Link
        print("\n--- Step 5: Link Notes ---")
        threshold = self.threshold_var.get()
        link_notes.run_linking_process(threshold, notes_info)

        # Step 6: Archive
        print("\n--- Step 6: Archive ---")
        try:
            shutil.move(audio_filepath, arch_audio)
            print(f"Archived: {os.path.basename(arch_audio)}")
        except Exception as e:
            print(f"Warning: {e}")

        try:
            shutil.move(raw_path, arch_transcript)
            print(f"Archived: {os.path.basename(arch_transcript)}")
        except Exception as e:
            print(f"Warning: {e}")

        self._cleanup(seg_path)

        print(f"\nCompleted: {recording_id}")
        return True

    def _cleanup(self, filepath: str):
        """Remove a temporary file."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

    def recalculate_links(self):
        """Recalculate all semantic links."""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "Please wait for current processing to finish.")
            return

        self.set_buttons_state(False)
        threshold = self.threshold_var.get()
        print(f"\nRecalculating all links (threshold: {threshold:.2f})...")

        self.processing_thread = threading.Thread(
            target=self._relink_thread, args=(threshold,)
        )
        self.processing_thread.start()

    def _relink_thread(self, threshold: float):
        """Run linking in a thread."""
        try:
            success = link_notes.run_linking_process(threshold)
            if success:
                print("Link recalculation complete.")
            else:
                print("Link recalculation failed.")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
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


if __name__ == "__main__":
    app = ZettelpalGUI()
    app.mainloop()
