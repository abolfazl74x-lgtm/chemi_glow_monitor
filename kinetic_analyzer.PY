import os
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# =============================================================================
# 1. Kinetic Models & Core Logic
# =============================================================================
END_OF_REACTION_THRESHOLD = 0.025
MIN_DATA_POINTS_FOR_FIT = 5

def model_zero_order_decay(t, k): return np.maximum(0, 1 - k * t)
def model_first_order_decay(t, k): return np.exp(-k * t)
def model_second_order_decay(t, k): return 1 / (1 + k * t)
def model_avrami_decay(t, k, n): return np.exp(-np.clip((k * t)**n, -700, 700))
def model_biexponential_decay(t, A1, k1, A2, k2, y0): 
    # Exactly matches Origin's ExpDec2 equation (includes y0 offset)
    # Note: Origin reports lifetimes, which are equivalent to: t1 = 1/k1 and t2 = 1/k2
    return y0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)
def model_sigmoidal_decay(t, ymax, x50, slope): return ymax / (1 + np.exp(np.clip(slope * (t - x50), -700, 700)))

def model_zero_order_growth(t, k): return np.minimum(1, k * t)
def model_first_order_growth(t, k): return 1 - np.exp(-k * t)
def model_second_order_growth(t, k): return (k * t) / (1 + k * t)
def model_avrami_growth(t, k, n): return 1 - np.exp(-np.clip((k * t)**n, -700, 700))
def model_biexponential_growth(t, A1, k1, A2, k2, y0): return y0 - (A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t))
def model_sigmoidal_growth(t, L, x50, k): return L / (1 + np.exp(np.clip(-k * (t - x50), -700, 700)))

decay_models = {
    "Zero-Order": {'func': model_zero_order_decay, 'p0': [0.001], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "First-Order": {'func': model_first_order_decay, 'p0': [0.001], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "Second-Order": {'func': model_second_order_decay, 'p0': [0.001], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "Avrami": {'func': model_avrami_decay, 'p0': [0.001, 1.0], 'bounds': ([0, 0], [np.inf, 10]), 'params': ['k', 'n']},
    "Bi-exponential": {
        'func': model_biexponential_decay, 
        # Initial guess suitable for hundreds of seconds scale and allows negative y0
        'p0': [1.5, 0.0005, 0.3, 0.015, -0.8], 
        'bounds': ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]), 
        'params': ['A1', 'k1', 'A2', 'k2', 'y0']
    },
    "Sigmoidal": {'func': model_sigmoidal_decay, 'p0': [1.0, 500.0, 0.005], 'bounds': ([0, -np.inf, 0], [np.inf, np.inf, np.inf]), 'params': ['ymax', 'x50', 'slope']}
}

growth_models = {
    "Zero-Order": {'func': model_zero_order_growth, 'p0': [0.01], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "First-Order": {'func': model_first_order_growth, 'p0': [0.01], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "Second-Order": {'func': model_second_order_growth, 'p0': [0.01], 'bounds': ([0], [np.inf]), 'params': ['k']},
    "Avrami": {'func': model_avrami_growth, 'p0': [0.01, 1.0], 'bounds': ([0, 0], [np.inf, 10]), 'params': ['k', 'n']},
    "Bi-exponential": {'func': model_biexponential_growth, 'p0': [0.5, 0.05, 0.5, 0.005, 1.0], 'bounds': ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]), 'params': ['A1', 'k1', 'A2', 'k2', 'y0']},
    "Sigmoidal": {'func': model_sigmoidal_growth, 'p0': [1.0, 50.0, 0.05], 'bounds': ([0, -np.inf, 0], [np.inf, np.inf, np.inf]), 'params': ['L', 'x50', 'k']}
}

def calculate_aic(n, k, y_true, y_pred):
    if n <= k + 1: return np.inf
    rss = np.sum((y_true - y_pred)**2)
    if rss <= 1e-15: return -np.inf
    return n * np.log(rss / n) + 2 * k

def calculate_t50(model_name, params):
    """ Calculates the half-life (t50) based on the mathematical formula of each model """
    try:
        if "Zero-Order" in model_name:
            return 0.5 / params[0]
        elif "First-Order" in model_name:
            return np.log(2) / params[0]
        elif "Second-Order" in model_name:
            return 1.0 / params[0]
        elif "Avrami" in model_name:
            k, n = params[0], params[1]
            return (np.log(2)**(1/n)) / k
        elif "Sigmoidal" in model_name:
            return params[1]  # For Sigmoidal model, the second parameter is exactly x50
        elif "Bi-exponential" in model_name:
            return np.nan # Bi-exponential model does not have a closed-form algebraic solution for t50
    except:
        return np.nan
    return np.nan

def run_analysis(input_path, selected_models, mode, calc_t50, log_callback):
    try:
        df = pd.read_excel(input_path, header=None)
    except Exception as e:
        log_callback(f"Error loading file: {e}")
        return

    output_dir = f"Analysis_Results_{mode.capitalize()}"
    os.makedirs(output_dir, exist_ok=True)
    all_summary = []
    
    models_dict = growth_models if mode == 'growth' else decay_models
    filtered_models = {k: v for k, v in models_dict.items() if k in selected_models}

    for i in range(1, len(df.columns)):
        test_name = str(df.columns[i]) if isinstance(df.columns[i], str) else f"Test_{i}"
        log_callback(f"Processing {test_name}...")

        x_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y_raw = pd.to_numeric(df.iloc[:, i], errors='coerce').values
        mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        x_raw, y_raw = x_raw[mask], y_raw[mask]

        if len(y_raw) < MIN_DATA_POINTS_FOR_FIT: continue
        peak_idx = np.argmax(y_raw)
        peak_val = y_raw[peak_idx]
        
        summary_row = {'Test': test_name, 'Peak_Value': peak_val}
        test_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)

        if mode == 'growth':
            x_target = x_raw[:peak_idx+1]
            y_target = y_raw[:peak_idx+1]
            if len(x_target) < MIN_DATA_POINTS_FOR_FIT: continue
            min_val = np.min(y_target)
            y_norm = (y_target - min_val) / (peak_val - min_val) if peak_val > min_val else y_target
            x_shift = x_target - x_target[0]
        else:
            x_target = x_raw[peak_idx:]
            y_target = y_raw[peak_idx:]
            if len(x_target) < MIN_DATA_POINTS_FOR_FIT: continue
            end_val = y_target[-1]
            y_norm = (y_target - end_val) / (peak_val - end_val) if peak_val > end_val else y_target
            end_idx = np.where(y_norm < END_OF_REACTION_THRESHOLD)[0]
            fit_limit = end_idx[0] + 1 if len(end_idx) > 0 and end_idx[0] >= MIN_DATA_POINTS_FOR_FIT else len(y_norm)
            x_shift, y_norm = (x_target - x_target[0])[:fit_limit], y_norm[:fit_limit]

        best_model, best_aic = "None", np.inf
        n_points = len(x_shift)

        for model_name, model_info in filtered_models.items():
            model_key = f"{mode.capitalize()}_{model_name}"
            try:
                params, _ = curve_fit(model_info['func'], x_shift, y_norm, p0=model_info['p0'], bounds=model_info['bounds'], maxfev=50000)
                y_pred = model_info['func'](x_shift, *params)
                r2 = r2_score(y_norm, y_pred)
                aic = calculate_aic(n_points, len(params), y_norm, y_pred)

                summary_row[f'{model_key}_R2'] = r2
                summary_row[f'{model_key}_AIC'] = aic
                
                for p_name, p_val in zip(model_info['params'], params):
                    summary_row[f'{model_key}_{p_name}'] = p_val
                
                # Add t50 calculation if checked by the user
                if calc_t50:
                    summary_row[f'{model_key}_t50'] = calculate_t50(model_name, params)
                
                # Calculate Origin's equivalent lifetimes for the Bi-exponential model
                if model_name == "Bi-exponential":
                    summary_row[f'{model_key}_t1_Origin'] = (1.0 / params[1]) if params[1] > 0 else np.nan
                    summary_row[f'{model_key}_t2_Origin'] = (1.0 / params[3]) if params[3] > 0 else np.nan

                # Plot saving
                plt.figure(figsize=(8, 5))
                plt.scatter(x_shift, y_norm, c='black', label='Data', s=20)
                x_plot = np.linspace(0, max(x_shift)*1.05, 300)
                plt.plot(x_plot, model_info['func'](x_plot, *params), 'b-', lw=2, label=f'Fit (R²={r2:.4f})')
                plt.title(f"{test_name} - {model_name} (AIC: {aic:.2f})")
                plt.xlabel("Time (s)"); plt.ylabel("Normalized Concentration")
                plt.legend(); plt.grid(True, ls='--')
                plt.savefig(os.path.join(test_dir, f"{model_name}.png"), dpi=150)
                plt.close()

                if aic < best_aic:
                    best_aic = aic
                    best_model = model_name
            except Exception:
                pass

        summary_row[f'Best_{mode.capitalize()}_Model'] = best_model
        all_summary.append(summary_row)

    if all_summary:
        df_res = pd.DataFrame(all_summary)
        front_cols = ['Test', 'Peak_Value', f'Best_{mode.capitalize()}_Model']
        other_cols = [c for c in df_res.columns if c not in front_cols]
        df_res = df_res[front_cols + sorted(other_cols)]
        out_path = os.path.join(output_dir, f"{mode.capitalize()}_Summary.xlsx")
        df_res.to_excel(out_path, index=False)
        log_callback(f"\nAnalysis completed successfully!\nResults saved to:\n{out_path}")
    else:
        log_callback("No valid data processed.")

# =============================================================================
# 2. Modern Tkinter GUI
# =============================================================================
class KineticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinetic Pro Analyzer v2.0")
        self.root.geometry("650x600")
        self.root.configure(padx=20, pady=20)
        
        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'), foreground="#2c3e50")
        style.configure('TRadiobutton', font=('Segoe UI', 9))
        style.configure('TCheckbutton', font=('Segoe UI', 9))

        # --- 1. File Input Frame ---
        lf_file = ttk.LabelFrame(self.root, text=" 1. Select Excel Data File ", padding=15)
        lf_file.pack(fill=tk.X, pady=(0, 15))
        
        self.var_filepath = tk.StringVar()
        entry_file = ttk.Entry(lf_file, textvariable=self.var_filepath, state='readonly')
        entry_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(lf_file, text="Browse...", command=self.browse).pack(side=tk.LEFT)

        # --- 2. Analysis Mode Frame ---
        lf_mode = ttk.LabelFrame(self.root, text=" 2. Analysis Settings ", padding=15)
        lf_mode.pack(fill=tk.X, pady=(0, 15))
        
        # Radio buttons for Growth/Decay
        self.var_mode = tk.StringVar(value="growth")
        self.var_mode.trace_add("write", self.refresh_models)
        ttk.Radiobutton(lf_mode, text="Growth Phase (Build-up)", variable=self.var_mode, value="growth").pack(side=tk.LEFT, padx=(10, 20))
        ttk.Radiobutton(lf_mode, text="Decay Phase (Recovery)", variable=self.var_mode, value="decay").pack(side=tk.LEFT, padx=(0, 30))

        # Checkbox for t50 calculation
        self.var_t50 = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_mode, text="Calculate Response Time (t50)", variable=self.var_t50).pack(side=tk.LEFT)

        # --- 3. Models Selection Frame ---
        self.lf_models = ttk.LabelFrame(self.root, text=" 3. Select Kinetic Models to Fit ", padding=15)
        self.lf_models.pack(fill=tk.X, pady=(0, 20))
        self.model_vars = {}
        self.refresh_models()

        # --- 4. Action Button ---
        self.btn_run = ttk.Button(self.root, text="▶ START KINETIC ANALYSIS", command=self.start)
        self.btn_run.pack(fill=tk.X, ipady=6, pady=(0, 20))

        # --- 5. Log Area ---
        lf_log = ttk.LabelFrame(self.root, text=" Output Log ", padding=10)
        lf_log.pack(fill=tk.BOTH, expand=True)
        self.txt_log = scrolledtext.ScrolledText(lf_log, height=10, font=('Consolas', 9), bg="#f8f9fa")
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.txt_log.config(state=tk.DISABLED)

    def browse(self):
        f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if f: 
            self.var_filepath.set(f)
            self.log(f"Loaded file: {os.path.basename(f)}")

    def refresh_models(self, *args):
        for w in self.lf_models.winfo_children(): w.destroy()
        self.model_vars.clear()
        models = growth_models if self.var_mode.get() == 'growth' else decay_models
        
        for i, m in enumerate(models.keys()):
            var = tk.BooleanVar(value=True)
            self.model_vars[m] = var
            cb = ttk.Checkbutton(self.lf_models, text=m, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=30, pady=8)

    def log(self, msg):
        def _write():
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, msg + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)
        self.root.after(0, _write)

    def start(self):
        path = self.var_filepath.get()
        if not path:
            messagebox.showerror("Error", "Please select an Excel data file first.")
            return
            
        selected = [m for m, v in self.model_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("Error", "Please select at least one model.")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.delete('1.0', tk.END)
        self.txt_log.config(state=tk.DISABLED)
        
        mode = self.var_mode.get()
        calc_t50 = self.var_t50.get()
        self.log(f"=== Starting {mode.upper()} Analysis ===")
        if calc_t50:
            self.log("Response Time (t50) calculation is ENABLED.")
        
        t = threading.Thread(target=self._run, args=(path, selected, mode, calc_t50))
        t.daemon = True
        t.start()

    def _run(self, path, selected, mode, calc_t50):
        run_analysis(path, selected, mode, calc_t50, self.log)
        self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = KineticApp(root)
    root.mainloop()