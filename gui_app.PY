import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import os

# Import the analysis logic and model definitions
from kinetic_analyzer import run_analysis, get_all_models

class KineticAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinetic Model Fitter")
        self.root.geometry("700x650")

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. File Selection Frame ---
        file_frame = ttk.LabelFrame(main_frame, text="Input Data File", padding="10")
        file_frame.pack(fill=tk.X, expand=True)

        self.filepath_var = tk.StringVar()
        self.filepath_entry = ttk.Entry(file_frame, textvariable=self.filepath_var, state='readonly', width=70)
        self.filepath_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        # --- 2. Configuration Frame ---
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, expand=True, pady=10)

        # --- 2a. Analysis Mode ---
        mode_frame = ttk.LabelFrame(config_frame, text="Analysis Mode", padding="10")
        mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.analysis_mode_var = tk.StringVar(value="growth")
        self.analysis_mode_var.trace_add("write", self.update_model_checkboxes)
        
        ttk.Radiobutton(mode_frame, text="Growth (Build-up)", variable=self.analysis_mode_var, value="growth").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Decay (Recovery)", variable=self.analysis_mode_var, value="decay").pack(anchor=tk.W)

        # --- 2b. Model Selection ---
        self.model_frame = ttk.LabelFrame(config_frame, text="Select Kinetic Models", padding="10")
        self.model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.all_models = get_all_models()
        self.model_vars = {}
        self.update_model_checkboxes() # Initial population

        # --- 3. Run Button ---
        self.run_button = ttk.Button(main_frame, text="Run Analysis", command=self.start_analysis)
        self.run_button.pack(fill=tk.X, pady=10)

        # --- 4. Log/Output Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

    def browse_file(self):
        """Opens a file dialog to select an Excel file."""
        filepath = filedialog.askopenfilename(
            title="Select Excel Data File",
            filetypes=(("Excel Files", "*.xlsx"), ("All files", "*.*"))
        )
        if filepath:
            self.filepath_var.set(filepath)
            self.log_message(f"Selected file: {os.path.basename(filepath)}")

    def update_model_checkboxes(self, *args):
        """Clears and repopulates the model checkboxes based on the selected analysis mode."""
        for widget in self.model_frame.winfo_children():
            widget.destroy()
        
        self.model_vars = {}
        mode = self.analysis_mode_var.get()
        models_for_mode = self.all_models.get(mode, {})
        
        for model_name in models_for_mode.keys():
            var = tk.BooleanVar(value=True) # Default to selected
            self.model_vars[model_name] = var
            cb = ttk.Checkbutton(self.model_frame, text=model_name, variable=var)
            cb.pack(anchor=tk.W)

    def log_message(self, message):
        """Appends a message to the log text area in a thread-safe way."""
        def append():
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END) # Auto-scroll
        # Schedule the GUI update to run in the main thread
        self.root.after(0, append)

    def start_analysis(self):
        """Starts the analysis in a new thread to keep the GUI responsive."""
        filepath = self.filepath_var.get()
        if not filepath:
            self.log_message("Error: Please select an input file first.")
            return

        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            self.log_message("Error: Please select at least one model to fit.")
            return
            
        analysis_mode = self.analysis_mode_var.get()

        # Disable button to prevent multiple runs
        self.run_button.config(state="disabled")
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')

        # Run the analysis in a separate thread to avoid freezing the GUI
        analysis_thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(filepath, selected_models, analysis_mode)
        )
        analysis_thread.daemon = True # Allows app to exit even if thread is running
        analysis_thread.start()

    def _run_analysis_thread(self, filepath, selected_models, analysis_mode):
        """Wrapper function that runs the analysis and re-enables the button."""
        try:
            run_analysis(filepath, selected_models, analysis_mode, self.log_message)
        except Exception as e:
            self.log_message(f"An unexpected error occurred: {e}")
        finally:
            # Re-enable the button once analysis is complete
            self.root.after(0, lambda: self.run_button.config(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    app = KineticAnalysisApp(root)
    root.mainloop()
