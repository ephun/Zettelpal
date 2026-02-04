# gui.py

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk # Import ttk for Notebook
import threading
import queue
import os
import sys
import datetime
import re
import shutil
import json
import inspect # For inspecting module to get configurable values

# Import core Zettelpal functions and config
import config
import transcribe
import segment
import create_notes
import link_notes
import utils
import classify

# --- Configuration Mapping (Defines all configurable items and their types) ---
# This list is used to dynamically generate the settings GUI.
CONFIGURABLE_ITEMS = [
    # Directories (Set to 'path' type for validation/description)
    ("OBSIDIAN_VAULT_ROOT", "path", "Absolute path to your Obsidian vault root."),
    ("ZETTELPAL_ROOT", "path", "Root directory for Zettelpal intermediate/archive files."),
    ("NOTES_SUBDIRECTORY_IN_VAULT", "str", "Subdirectory within the vault for generated notes."),
    ("ARCHIVE_DIR", "path", "Directory where original audio and raw transcripts are archived."),
    # Models & API
    ("GOOGLE_API_KEY", "str", "Gemini API Key (preferably set via environment variable)."),
    ("LOCAL_WHISPER_MODEL_SIZE", "str", "Local Whisper model size (e.g., 'medium', 'large-v3')."),
    ("LOCAL_EMBEDDING_MODEL", "str", "Sentence Transformer model for embeddings (e.g., 'all-MiniLM-L6-v2')."),
    ("SEGMENTATION_LLM_MODEL", "str", "Gemini model for segmentation and tagging (e.g., 'gemini-2.5-pro')."),
    ("SEGMENTATION_LLM_TEMPERATURE", "float", "LLM creativity level (0.0 to 1.0)."),
    ("SEGMENTATION_MAX_TOKENS", "int", "Max output tokens for LLM segmentation (prevent JSON cutoff)."),
    # Linking
    ("SIMILARITY_THRESHOLD", "float", "Default Cosine similarity cutoff for semantic linking."),
    ("SOURCE_METADATA_KEY", "str", "YAML key for source metadata (e.g., 'source')."),
    # LLM Prompts (Set to 'multiline' for Text widget)
    ("CLASSIFICATION_PROMPT_TEMPLATE", "multiline", "Prompt for classifying transcript type."),
    ("SEGMENTATION_PROMPT_TEMPLATE", "multiline", "Prompt for breaking transcript into segments and titling."),
    ("GRANULAR_TAGGING_PROMPT_TEMPLATE", "multiline", "Prompt for generating fine-grained tags for each segment."),
    # Hardcoded/Runtime (Include these for completeness in settings, though they live in other files)
    ("max_chars_per_chunk (segment.py)", "int", "Max character length for LLM input chunks (in segment.py)."),
    ("max_output_tokens (classify.py)", "int", "Max output tokens for classification prompt (in classify.py)."),
]


class TextRedirector:
    """A helper class to redirect stdout to a Tkinter Text widget."""
    def __init__(self, widget):
        self.widget = widget
        self.queue = queue.Queue()
        self.update_interval = 100 # ms, how often to check the queue
        self.widget.after(self.update_interval, self.update_text)

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        # Tkinter's Text widget doesn't need flushing for append operations
        pass

    def update_text(self):
        try:
            while True:
                text = self.queue.get_nowait()
                self.widget.insert(tk.END, text)
                self.widget.see(tk.END) # Auto-scroll to the end
        except queue.Empty:
            pass
        finally:
            self.widget.after(self.update_interval, self.update_text)

class ZettelpalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zettelpal - Audio to Obsidian Notes")
        self.geometry("900x700")

        icon_path = os.path.join(os.path.dirname(__file__), "zettelpal.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except tk.TclError as e:
                print(f"Warning: Could not set icon. Error: {e}")

        self.audio_files = []
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.config_vars = {} # Dictionary to hold Tkinter variables for settings

        # Configure grid weights (removed, now managed by Notebook)

        self.create_widgets()
        self.redirect_stdout_to_console()
        self.ensure_directories_gui_check()

    def ensure_directories_gui_check(self):
        # ... (Same as original implementation)
        """Performs initial directory checks and prints to console."""
        print("Performing initial directory checks...")
        try:
            # Zettelpal Root and its subdirectories (intermediate and archive)
            os.makedirs(config.ZETTELPAL_ROOT, exist_ok=True)
            os.makedirs(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, exist_ok=True)
            os.makedirs(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, exist_ok=True)
            os.makedirs(config.ARCHIVE_DIR, exist_ok=True)

            # Obsidian Vault related directories
            if not os.path.exists(config.OBSIDIAN_VAULT_ROOT):
                 print(f"ERROR: Obsidian vault root not found at {config.OBSIDIAN_VAULT_ROOT}")
                 print("Please check your config.py and ensure OBSIDIAN_VAULT_ROOT is correct.")
                 messagebox.showerror("Configuration Error",
                                      f"Obsidian vault root not found at {config.OBSIDIAN_VAULT_ROOT}. "
                                      "Please check your config.py.")
                 self.destroy() # Close the app if critical config is missing
                 return False

            notes_subdir_path = os.path.join(config.OBSIDIAN_VAULT_ROOT, config.NOTES_SUBDIRECTORY_IN_VAULT)
            os.makedirs(notes_subdir_path, exist_ok=True)

            cache_dir = os.path.dirname(config.EMBEDDINGS_CACHE_FILE)
            os.makedirs(cache_dir, exist_ok=True)

            print("All necessary directories checked/created.")
            return True
        except Exception as e:
            print(f"CRITICAL ERROR during directory setup: {e}")
            messagebox.showerror("Setup Error", f"Failed to set up directories: {e}. See console for details.")
            self.destroy()
            return False

    def create_widgets(self):
        # --- Top-Level Notebook (Tabs) ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # --- 1. Pipeline Tab (Main Operational View) ---
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="Pipeline")
        self.pipeline_tab.grid_rowconfigure(1, weight=1)
        self.pipeline_tab.grid_columnconfigure(0, weight=1)

        # File Selection Frame (on Pipeline Tab)
        file_frame = tk.LabelFrame(self.pipeline_tab, text="Select Audio Files", padx=10, pady=10)
        file_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        file_frame.grid_columnconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(file_frame, selectmode=tk.EXTENDED, height=5)
        self.file_listbox.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        add_button = tk.Button(file_frame, text="Add Files", command=self.add_files)
        add_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        remove_button = tk.Button(file_frame, text="Remove Selected", command=self.remove_selected_files)
        remove_button.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        # Console Output Frame (on Pipeline Tab)
        console_frame = tk.LabelFrame(self.pipeline_tab, text="Console Output", padx=10, pady=10)
        console_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        console_frame.grid_rowconfigure(0, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=15, state='disabled')
        self.console_text.grid(row=0, column=0, sticky="nsew")

        # Pipeline Controls Frame (on Pipeline Tab, only Start/Quit)
        pipeline_controls_frame = tk.Frame(self.pipeline_tab, padx=5, pady=5)
        pipeline_controls_frame.grid(row=2, column=0, sticky="ew")
        pipeline_controls_frame.grid_columnconfigure(0, weight=1)
        pipeline_controls_frame.grid_columnconfigure(1, weight=1)
        pipeline_controls_frame.grid_columnconfigure(2, weight=1)

        self.start_pipeline_button = tk.Button(pipeline_controls_frame, text="Start Pipeline (Process Files)", command=self.start_pipeline)
        self.start_pipeline_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.clear_console_button = tk.Button(pipeline_controls_frame, text="Clear Console", command=self.clear_console)
        self.clear_console_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.quit_button = tk.Button(pipeline_controls_frame, text="Quit", command=self.quit_application)
        self.quit_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")


        # --- 2. Settings Tab (Configuration View) ---
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        self.settings_tab.grid_rowconfigure(0, weight=1)
        self.settings_tab.grid_columnconfigure(0, weight=1)

        # Scrollable Area for Settings
        canvas = tk.Canvas(self.settings_tab)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

        self.settings_content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.settings_content_frame, anchor="nw")

        self.settings_content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self.settings_content_frame.grid_columnconfigure(0, weight=0) # Labels
        self.settings_content_frame.grid_columnconfigure(1, weight=1) # Inputs

        # General Controls Frame (Manual Tags, Slider, Recalculate)
        general_controls_frame = tk.LabelFrame(self.settings_content_frame, text="Runtime Controls & Linking", padx=10, pady=10)
        general_controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        general_controls_frame.grid_columnconfigure(1, weight=1)
        general_controls_frame_row = 0

        # Semantic Similarity Slider
        self.threshold_label = tk.Label(general_controls_frame, text=f"Similarity Threshold: {config.SIMILARITY_THRESHOLD:.2f}")
        self.threshold_label.grid(row=general_controls_frame_row, column=0, columnspan=2, pady=(0, 5), sticky="w")
        general_controls_frame_row += 1

        self.threshold_slider = tk.Scale(general_controls_frame, from_=0.0, to=1.0, resolution=0.01,
                                         orient=tk.HORIZONTAL, length=300,
                                         command=self.update_threshold_label)
        self.threshold_slider.set(config.SIMILARITY_THRESHOLD) # Set initial value
        self.threshold_slider.grid(row=general_controls_frame_row, column=0, columnspan=2, padx=5, sticky="ew")
        general_controls_frame_row += 1

        # Manual Tags Input
        tk.Label(general_controls_frame, text="Manual Tags (comma separated):").grid(row=general_controls_frame_row, column=0, sticky="w", pady=(10, 2))
        self.manual_tags_entry = tk.Entry(general_controls_frame)
        self.manual_tags_entry.grid(row=general_controls_frame_row, column=1, sticky="ew", padx=5, pady=(10, 2))
        general_controls_frame_row += 1
        
        # Recalculate Button
        self.recalculate_links_button = tk.Button(general_controls_frame, text="Recalculate ALL Links (Applies Current Threshold)", command=self.recalculate_links)
        self.recalculate_links_button.grid(row=general_controls_frame_row, column=0, columnspan=2, padx=5, pady=10, sticky="ew")


        # Dynamic Configuration Editor
        self.create_config_editor(self.settings_content_frame)

        # Save Button (outside the dynamic editor)
        save_button = tk.Button(self.settings_content_frame, text="Save Settings to config.py", command=self.save_config)
        save_button.grid(row=self.settings_content_row, column=0, columnspan=2, pady=20)


    def create_config_editor(self, parent_frame):
        """Dynamically creates input fields for all configurable items."""
        
        config_editor_frame = tk.LabelFrame(parent_frame, text="LLM and System Configuration (config.py)", padx=10, pady=10)
        config_editor_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        config_editor_frame.grid_columnconfigure(1, weight=1)

        self.settings_content_row = 0

        for name, data_type, description in CONFIGURABLE_ITEMS:
            current_row = self.settings_content_row

            # Determine the module and variable name
            if "(" in name and name.endswith(")"): # Hardcoded/Runtime variable from other files
                # Example name: "max_chars_per_chunk (segment.py)"
                
                # Split the full name into "variable_name" and "(module.py)"
                parts = name.split("(")
                var_name = parts[0].strip() # "max_chars_per_chunk"
                module_part = parts[1].strip() # "segment.py)"

                # Extract module name from "(module.py)"
                module_name = module_part.replace(").py", "").replace(")", "").replace(".py", "") # Cleans up segment.py or classify.py
                
                # Use the extracted variable name (var_name) for logic
                if var_name == "max_chars_per_chunk":
                    # This is hardcoded in segment.py (line ~77)
                    initial_value = 10000 
                elif var_name == "max_output_tokens":
                    # This is hardcoded in classify.py (line ~43)
                    initial_value = 50 
                else:
                    initial_value = "N/A"
                source = f"({module_name}.py)"
            else: # Variable from config.py
                try:
                    initial_value = getattr(config, name)
                except AttributeError:
                    initial_value = "NOT FOUND"
                source = "(config.py)"

            # Label for variable name and description
            label_text = f"{name} {source}:\n{description}"
            tk.Label(config_editor_frame, text=label_text, anchor="w", justify=tk.LEFT,
                     wraplength=400).grid(row=current_row, column=0, sticky="w", padx=5, pady=(5, 0))

            # Input Widget
            if data_type == "multiline":
                var = tk.StringVar(value=initial_value if initial_value else "") # Text widgets use insert/get, not StringVar
                text_widget = scrolledtext.ScrolledText(config_editor_frame, height=8, width=50, wrap=tk.WORD)
                text_widget.insert(tk.END, initial_value)
                text_widget.grid(row=current_row, column=1, sticky="ew", padx=5, pady=5)
                self.config_vars[name] = (data_type, text_widget)

            else: # Simple Entry fields
                # Use StringVar to hold the value and enable easy reading
                var = tk.StringVar(value=str(initial_value))
                entry = tk.Entry(config_editor_frame, textvariable=var)
                entry.grid(row=current_row, column=1, sticky="ew", padx=5, pady=5)
                self.config_vars[name] = (data_type, var)

            self.settings_content_row += 1


    def save_config(self):
        """Reads all configuration fields and attempts to save back to config.py."""
        updates = {}
        
        # 1. Gather all values and validate
        for name, (data_type, widget_or_var) in self.config_vars.items():
            try:
                if data_type == "multiline":
                    value = widget_or_var.get("1.0", tk.END).strip()
                else:
                    value = widget_or_var.get().strip()

                if "(" in name and name.endswith(")"):
                     # Skip writing hardcoded runtime variables to config.py, but validate type
                     if data_type == "int": int(value)
                     elif data_type == "float": float(value)
                     continue 
                     
                # Validate and format for writing to config.py
                if data_type == "int":
                    updates[name] = int(value)
                elif data_type == "float":
                    updates[name] = float(value)
                elif data_type == "path":
                    # Paths are just strings, but ensure they aren't empty
                    if not value: raise ValueError("Path cannot be empty.")
                    updates[name] = value
                else: # 'str' type
                    updates[name] = value

            except ValueError as e:
                messagebox.showerror("Validation Error", f"Invalid value for {name} ({data_type}): {e}")
                return # Stop save process

        # 2. Rewrite config.py
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            
            # Simple line-by-line substitution is used for simplicity.
            for line in lines:
                updated = False
                for name, value in updates.items():
                    if line.strip().startswith(f"{name} ="):
                        if isinstance(value, str) and not name.endswith("TEMPLATE"):
                            # Wrap strings (including paths) in quotes
                            new_lines.append(f'{name} = "{value}"\n')
                        elif isinstance(value, str) and name.endswith("TEMPLATE"):
                            # Multiline prompts are already structured correctly; write as is
                            new_lines.append(line) 
                        else:
                            # Write numbers directly
                            new_lines.append(f'{name} = {value}\n')
                        updated = True
                        break
                
                if not updated:
                    new_lines.append(line)

            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            
            messagebox.showinfo("Success", "Configuration saved successfully. Restart the application for changes to fully take effect!")
            
            # Optionally, update runtime values (e.g., SIMILARITY_THRESHOLD) without restart
            if "SIMILARITY_THRESHOLD" in updates:
                new_threshold = updates["SIMILARITY_THRESHOLD"]
                config.SIMILARITY_THRESHOLD = new_threshold # Update module variable
                self.threshold_slider.set(new_threshold) # Update GUI slider
                self.update_threshold_label(new_threshold)

        except Exception as e:
            messagebox.showerror("File Error", f"Failed to save config.py: {e}")
            import traceback
            traceback.print_exc()


    # --- Methods moved from Controls Frame ---

    def add_files(self):
        # ... (Same as original implementation)
        filetypes = [("Audio files", "*.mp3 *.wav *.flac *.m4a"), ("All files", "*.*")]
        new_files_selected = filedialog.askopenfilenames(
            title="Select Audio Files for Zettelpal",
            filetypes=filetypes
        )
        if new_files_selected:
            processed_count = 0
            for original_filepath in new_files_selected:
                filename = os.path.basename(original_filepath)

                # Check if the file already has the correct naming schema
                if not utils.is_valid_zettelpal_filename(filename):
                    print(f"File '{filename}' does not match MMDDYYNN.ext format. Attempting to rename...")
                    renamed_filepath = utils.rename_audio_file_to_zettelpal_format(original_filepath)
                    if renamed_filepath:
                        filepath_to_add = renamed_filepath
                        print(f"Successfully renamed to '{os.path.basename(renamed_filepath)}'.")
                    else:
                        print(f"Failed to rename '{filename}'. This file will be skipped.")
                        messagebox.showwarning("Renaming Failed", f"Could not rename '{filename}'. It will be skipped. Please check console for details.")
                        continue # Skip this file if renaming failed
                else:
                    filepath_to_add = original_filepath
                    print(f"File '{filename}' already matches MMDDYYNN.ext format. No renaming needed.")

                # Add to listbox and internal list only if it's not already there and wasn't skipped
                if filepath_to_add not in self.audio_files:
                    self.audio_files.append(filepath_to_add)
                    self.file_listbox.insert(tk.END, os.path.basename(filepath_to_add))
                    processed_count += 1
                else:
                    print(f"File '{os.path.basename(filepath_to_add)}' already in queue.")

            if processed_count > 0:
                print(f"Added {processed_count} new files to the queue.")
            else:
                print("No new files added (all either skipped or already in queue).")

    def remove_selected_files(self):
        # ... (Same as original implementation)
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select files to remove.")
            return

        # Delete from listbox in reverse order to avoid index issues
        for i in selected_indices[::-1]:
            self.file_listbox.delete(i)
            del self.audio_files[i] # Remove from our internal list too
        print("Removed selected files.")

    def clear_console(self):
        # ... (Same as original implementation)
        self.console_text.configure(state='normal')
        self.console_text.delete(1.0, tk.END)
        self.console_text.configure(state='disabled')
        print("Console cleared.")

    def redirect_stdout_to_console(self):
        # ... (Same as original implementation)
        self.console_text.configure(state='normal') # Enable writing
        sys.stdout = TextRedirector(self.console_text)
        sys.stderr = TextRedirector(self.console_text) # Also redirect stderr
        print("Console output redirected.")

    def update_threshold_label(self, val):
        self.threshold_label.config(text=f"Similarity Threshold: {float(val):.2f}")

    def disable_buttons(self):
        # Disable all major action buttons and input fields
        self.start_pipeline_button.config(state=tk.DISABLED)
        self.recalculate_links_button.config(state=tk.DISABLED)
        self.threshold_slider.config(state=tk.DISABLED)
        self.manual_tags_entry.config(state=tk.DISABLED)

    def enable_buttons(self):
        # Enable all major action buttons and input fields
        self.start_pipeline_button.config(state=tk.NORMAL)
        self.recalculate_links_button.config(state=tk.NORMAL)
        self.threshold_slider.config(state=tk.NORMAL)
        self.manual_tags_entry.config(state=tk.NORMAL)

    def start_pipeline(self):
        # ... (Same as original implementation, using updated flow)
        if not self.audio_files:
            messagebox.showinfo("No Files", "Please add audio files to process.")
            return

        # Get manual tags from the Settings Tab widget
        manual_tags_input = self.manual_tags_entry.get().strip()
        
        # Put all selected files and the tags into the processing queue
        for filepath in self.audio_files:
            self.processing_queue.put((filepath, manual_tags_input))

        # Clear the listbox and internal list
        self.file_listbox.delete(0, tk.END)
        self.audio_files = []
        
        # Clear the manual tags input box after reading it
        self.manual_tags_entry.delete(0, tk.END)

        self.disable_buttons()
        print("\n--- Starting Zettelpal Pipeline ---")
        self.processing_thread = threading.Thread(target=self._process_files_in_queue)
        self.processing_thread.start()

    def _process_files_in_queue(self):
        # ... (Same as original implementation)
        while not self.processing_queue.empty():
            current_audio_filepath, manual_tags = self.processing_queue.get() 
            print(f"\nProcessing file: {os.path.basename(current_audio_filepath)}")
            print(f"  Manual tags applied: '{manual_tags}'")
            try:
                pipeline_success = self._run_single_pipeline_for_gui(current_audio_filepath, manual_tags)
                if not pipeline_success:
                     print(f"Pipeline failed for {os.path.basename(current_audio_filepath)}. Skipping.")
            except Exception as e:
                print(f"Error processing {os.path.basename(current_audio_filepath)}: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Processing Error", f"Error processing {os.path.basename(current_audio_filepath)}. See console for details.")

        print("\n--- All files processed. ---")
        self.after(0, self.enable_buttons)


    def _run_single_pipeline_for_gui(self, audio_filepath: str, manual_tags_str: str) -> bool:
        # ... (Same as original implementation)
        print(f"--- Zettelpal Pipeline Started for {os.path.basename(audio_filepath)} ---")

        audio_basename = os.path.basename(audio_filepath)
        audio_filename_stem = os.path.splitext(audio_basename)[0]
        audio_extension = os.path.splitext(audio_basename)[1].lstrip('.')

        if not re.match(r"^\d{6,8}$", audio_filename_stem):
            print(f"ERROR: Audio filename stem '{audio_filename_stem}' does not match expected MMDDYY(NN) format (6 to 8 digits).")
            print("Please ensure files are renamed correctly before adding.")
            return False

        recording_id = audio_filename_stem
        timestamp_full = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        raw_transcript_filepath_intermediate = os.path.join(config.RAW_TRANSCRIPTS_INTERMEDIATE_DIR, f"{recording_id}.txt")
        segmented_filepath_intermediate = os.path.join(config.SEGMENTED_OUTPUT_INTERMEDIATE_DIR, f"{recording_id}_segmented.json")
        archived_audio_filepath = os.path.join(config.ARCHIVE_DIR, f"{recording_id}_{timestamp_full}.{audio_extension}")
        archived_raw_transcript_filepath = os.path.join(config.ARCHIVE_DIR, f"{recording_id}_{timestamp_full}.txt")

        print(f"Processing Recording ID: {recording_id}")

        # --- Step 1: Transcribe ---
        print("\n--- Step 1: Transcribing Audio ---")
        transcription_output_path = transcribe.transcribe_audio_to_file(audio_filepath, raw_transcript_filepath_intermediate)
        if transcription_output_path is None or not os.path.exists(transcription_output_path):
            print("ERROR: Transcription step failed. Skipping pipeline for this file.")
            return False

        print("Step 1 Complete.")

        # --- Step 2: Classify ---
        print("\n--- Step 2: Classifying Transcript ---")
        try:
            with open(transcription_output_path, "r", encoding="utf-8") as f:
                raw_transcript_content = f.read()
        except Exception as e:
            print(f"ERROR: Could not re-read raw transcript file for classification: {e}")
            print("Skipping pipeline for this file.")
            return False
        
        transcript_type_tag = classify.classify_transcript(raw_transcript_content)
        
        if transcript_type_tag is None:
            print("ERROR: Transcript could not be classified. **Every note needs a type tag.** Skipping pipeline for this file.")
            if os.path.exists(transcription_output_path):
                try:
                    os.remove(transcription_output_path)
                    print(f"  Cleaned up intermediate raw transcript: {transcription_output_path}")
                except Exception as e:
                    print(f"Warning: Could not clean up intermediate raw transcript {transcription_output_path}: {e}")
            return False
            
        print(f"Transcript classified as: {transcript_type_tag}")
        print("Step 2 Complete.")

        # --- Step 3: Segment ---
        print("\n--- Step 3: Segmenting Transcript ---")
        if config.GOOGLE_API_KEY is None:
             print("ERROR: GOOGLE_API_KEY environment variable is not set. Cannot perform segmentation.")
             if os.path.exists(transcription_output_path):
                 try:
                     os.remove(transcription_output_path)
                     print(f"  Cleaned up intermediate raw transcript: {transcription_output_path}")
                 except Exception as e:
                     print(f"Warning: Could not clean up intermediate raw transcript {transcription_output_path}: {e}")
             return False

        segmented_data = segment.segment_and_title_transcript_gemini(raw_transcript_content)

        if not segmented_data:
            print("ERROR: Segmentation step failed or produced no segments. Skipping pipeline for this file.")
            if os.path.exists(transcription_output_path):
                try:
                    os.remove(transcription_output_path)
                    print(f"  Cleaned up intermediate raw transcript: {transcription_output_path}")
                except Exception as e:
                    print(f"Warning: Could not clean up intermediate raw transcript {transcription_output_path}: {e}")

            return False

        try:
            os.makedirs(os.path.dirname(segmented_filepath_intermediate), exist_ok=True)
            with open(segmented_filepath_intermediate, "w", encoding="utf-8") as f:
                json.dump(segmented_data, f, indent=4)
            print(f"Segmented data saved to {segmented_filepath_intermediate}")
        except Exception as e:
            print(f"Warning: Could not save segmented data to intermediate file {segmented_filepath_intermediate}: {e}")

        print("Step 3 Complete.")

        # --- Step 4: Create Obsidian Notes & Embed ---
        print("\n--- Step 4: Creating Obsidian Notes ---")
        notes_info_for_linking = create_notes.create_notes_from_segments(
            segmented_data,
            os.path.abspath(config.OBSIDIAN_VAULT_ROOT),
            config.NOTES_SUBDIRECTORY_IN_VAULT,
            recording_id,
            transcript_type_tag,
            manual_tags_str
        )

        if not notes_info_for_linking:
            print("Warning: Note creation completed, but no notes were successfully created with valid embeddings. Linking step will be skipped for these new notes.")
            print("Step 4 Complete (No new notes with embeddings).")
        else:
             print(f"Step 4 Complete. {len(notes_info_for_linking)} notes created and initial embeddings calculated.")


        # --- Step 5: Perform Semantic Linking (for ALL notes in vault) ---
        print("\n--- Step 5: Performing Semantic Linking for ALL notes ---")
        # Get threshold directly from slider/updated config
        current_threshold = self.threshold_slider.get() 

        link_success = link_notes.run_linking_process(current_threshold)

        if not link_success:
            print("Warning: Semantic linking encountered an issue. Please check console output.")

        print("Step 5 Complete: Semantic linking finished.")

        # --- Step 6 & 7: Archiving and Cleanup ---
        print("\n--- Step 6 & 7: Archiving and Cleanup ---")

        print("Archiving original audio file...")
        try:
            shutil.move(audio_filepath, archived_audio_filepath)
            print(f"  Archived audio file to: {archived_audio_filepath}")
        except FileNotFoundError:
             print(f"Warning: Original audio file not found at {audio_filepath} for archiving.")
        except Exception as e:
            print(f"Warning: Failed to archive original audio file from {audio_filepath}: {e}")

        print("Archiving raw transcript text file...")
        if transcription_output_path and os.path.exists(transcription_output_path):
            try:
                shutil.move(transcription_output_path, archived_raw_transcript_filepath)
                print(f"  Archived raw transcript to: {archived_raw_transcript_filepath}")
            except FileNotFoundError:
                 print(f"Warning: Intermediate raw transcript file not found at {transcription_output_path} for archiving.")
            except Exception as e:
                 print(f"Warning: Failed to archive raw transcript from {transcription_output_path}: {e}")
        else:
            print(f"Warning: No intermediate raw transcript file found at {transcription_output_path} to archive.")

        print("Cleaning up intermediate segmented JSON file...")
        try:
            if os.path.exists(segmented_filepath_intermediate):
                 os.remove(segmented_filepath_intermediate)
                 print(f"  Cleaned up {segmented_filepath_intermediate}")
            else:
                print(f"  Intermediate segmented file not found: {segmented_filepath_intermediate} (already gone?)")
        except Exception as e:
            print(f"Warning: Failed to clean up segmented intermediate file {segmented_filepath_intermediate}: {e}")

        print(f"\n--- Zettelpal Pipeline Finished for {os.path.basename(audio_filepath)} ---")
        return True


    def recalculate_links(self):
        # ... (Same as original implementation)
        if self.processing_thread and self.processing_thread.is_alive():
             messagebox.showinfo("Process Running", "A file processing pipeline is currently running. Please wait for it to finish before recalculating links.")
             return

        self.disable_buttons()
        current_threshold = self.threshold_slider.get()
        print(f"\n--- Recalculating all links with threshold: {current_threshold:.2f} ---")
        self.processing_thread = threading.Thread(target=self._run_linking_in_thread, args=(current_threshold,))
        self.processing_thread.start()

    def _run_linking_in_thread(self, threshold):
        # ... (Same as original implementation)
        try:
            link_success = link_notes.run_linking_process(threshold)
            if link_success:
                 print("\n--- Link Recalculation Completed ---")
            else:
                 print("\n--- Link Recalculation Failed ---")
                 self.after(0, lambda: messagebox.showerror("Linking Error", "An error occurred during link recalculation. See console for details."))

        except Exception as e:
            print(f"Error during link recalculation: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Linking Error", f"An error occurred during link recalculation: {e}. See console for details."))
        finally:
            self.after(0, self.enable_buttons)

    def quit_application(self):
        # ... (Same as original implementation)
        if self.processing_thread and self.processing_thread.is_alive():
            if not messagebox.askyesno("Confirm Quit", "A process is running. Quitting now will terminate it. Are you sure?"):
                return
        self.destroy()


if __name__ == "__main__":
    app = ZettelpalGUI()
    app.mainloop()