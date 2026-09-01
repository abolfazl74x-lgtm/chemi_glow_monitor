import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# =============================================================================
# LOGIC CORE (Computational Logic)
# =============================================================================

def analyze_signal_logic(time, signal, signal_name, params):
    """
    Analyzes the signal based on 7 INDEPENDENT user parameters.
    """

    raw_signal_np = np.array(signal)
    time_np = np.array(time)

    # --- INPUT 2: Smoothing ---
    try:
        window_length = int(params['smooth_window'])
    except:
        window_length = 21

    # Validation: Window length must be odd and less than signal length
    if len(raw_signal_np) < window_length:
        window_length = len(raw_signal_np) if len(raw_signal_np) % 2 != 0 else len(raw_signal_np) - 1
    if window_length < 3:
        window_length = 3
    if window_length % 2 == 0:
        window_length += 1

    try:
        signal_smooth = savgol_filter(raw_signal_np, window_length=window_length, polyorder=3)
    except:
        signal_smooth = raw_signal_np

    # --- Peak Calculation ---
    peak_index = np.argmax(signal_smooth)
    peak_value = signal_smooth[peak_index]
    time_to_peak = time_np[peak_index]

    # =========================================================================
    # INDEPENDENT THRESHOLD CALCULATIONS
    # =========================================================================

    # ---------------------------------------------------------
    # INPUT 3: Total Integral Cutoff % (Main AUC)
    # ---------------------------------------------------------
    thresh_val_integral_total = peak_value * (params['pct_integral_total'] / 100.0)

    cutoff_index_auc = len(signal_smooth) - 1
    if peak_index < len(signal_smooth) - 1:
        signal_after_peak = signal_smooth[peak_index:]
        indices_below = np.where(signal_after_peak <= thresh_val_integral_total)[0]
        if len(indices_below) > 0:
            cutoff_index_auc = peak_index + indices_below[0]

    time_auc_processed = time_np[:cutoff_index_auc + 1]
    signal_auc_processed = signal_smooth[:cutoff_index_auc + 1]
    auc_total = trapezoid(y=signal_auc_processed, x=time_auc_processed)

    # ---------------------------------------------------------
    # INPUT 4: High Integral % (High AUC)
    # ---------------------------------------------------------
    thresh_val_integral_high = peak_value * (params['pct_integral_high'] / 100.0)
    indices_above_high_auc = np.where(signal_smooth >= thresh_val_integral_high)[0]

    auc_high = 0
    if len(indices_above_high_auc) > 1:
        signal_seg_high = signal_smooth[indices_above_high_auc]
        time_seg_high = time_np[indices_above_high_auc]
        auc_high = trapezoid(y=signal_seg_high, x=time_seg_high)

    # ---------------------------------------------------------
    # INPUT 5: Total Emission Time %
    # ---------------------------------------------------------
    thresh_val_time_total = peak_value * (params['pct_time_total'] / 100.0)

    cutoff_index_time = len(signal_smooth) - 1
    if peak_index < len(signal_smooth) - 1:
        signal_after_peak_t = signal_smooth[peak_index:]
        indices_below_t = np.where(signal_after_peak_t <= thresh_val_time_total)[0]
        if len(indices_below_t) > 0:
            cutoff_index_time = peak_index + indices_below_t[0]

    emission_time_total = time_np[cutoff_index_time]

    # ---------------------------------------------------------
    # INPUT 6: High Duration Range %
    # ---------------------------------------------------------
    thresh_val_time_high = peak_value * (params['pct_time_high'] / 100.0)
    indices_above_high_dur = np.where(signal_smooth >= thresh_val_time_high)[0]

    duration_high = 0
    if len(indices_above_high_dur) > 0:
        t_start = time_np[indices_above_high_dur[0]]
        t_end = time_np[indices_above_high_dur[-1]]
        duration_high = t_end - t_start

    # ---------------------------------------------------------
    # INPUT 7: Max Emission in Fixed Window
    # ---------------------------------------------------------
    target_window_dur = float(params['max_emission_window_dur'])
    max_window_auc = 0.0
    best_window_indices = (0, 0) # start_idx, end_idx
    
    # Sliding window logic
    n_points = len(time_np)
    if n_points > 1:
        for start_idx in range(n_points):
            t_start = time_np[start_idx]
            t_end_target = t_start + target_window_dur
            
            # Find end index efficiently
            end_idx = np.searchsorted(time_np, t_end_target, side='right') - 1
            
            if end_idx <= start_idx:
                continue
            
            # Extract slice
            t_slice = time_np[start_idx : end_idx+1]
            s_slice = signal_smooth[start_idx : end_idx+1]
            
            # Calculate Area
            current_auc = trapezoid(y=s_slice, x=t_slice)
            
            if current_auc > max_window_auc:
                max_window_auc = current_auc
                best_window_indices = (start_idx, end_idx)

    # Get coordinates for plotting and Reporting
    w_start_idx, w_end_idx = best_window_indices
    best_window_time = time_np[w_start_idx : w_end_idx+1]
    best_window_signal = signal_smooth[w_start_idx : w_end_idx+1]
    
    # Format the range string for Excel
    if len(best_window_time) > 0:
        range_str = f"{best_window_time[0]:.2f} - {best_window_time[-1]:.2f}"
    else:
        range_str = "N/A"


    # ==================== Other Stats ====================
    total_time_span = time_auc_processed[-1] - time_auc_processed[0]
    mean_value = auc_total / total_time_span if total_time_span > 0 else 0

    if len(time_np) > 1 and (time_np[1] - time_np[0]) > 0:
        initial_slope = (signal_smooth[1] - signal_smooth[0]) / (time_np[1] - time_np[0])
    else:
        initial_slope = 0

    # ==================== Results Dictionary ====================
    results = {
        'Signal Name': signal_name,
        'Peak Value': peak_value,
        'Time to Peak (s)': time_to_peak,

        # Input 7 Result (Updated)
        f'Max Emission AUC (in {target_window_dur}s window)': max_window_auc,
        'Max Emission Range (s)': range_str,  # <--- NEW COLUMN ADDED HERE

        # Input 6 Result
        f'High Duration (s) [>{params["pct_time_high"]}%]': duration_high,

        # Input 5 Result
        f'Total Emission Time (s) [Drop to {params["pct_time_total"]}%]': emission_time_total,

        # Input 4 Result
        f'High Integral (AUC) [>{params["pct_integral_high"]}%]': auc_high,

        # Input 3 Result
        f'Total Integral (AUC) [Cutoff {params["pct_integral_total"]}%]': auc_total,

        'Mean Value': mean_value,
        'Initial Slope': initial_slope
    }

    # ==================== Plotting ====================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Raw & Smooth
    ax.plot(time_np, raw_signal_np, color='lightgrey', label='Raw')
    ax.plot(time_np, signal_smooth, color='blue', label='Smoothed', linewidth=1.5)

    # 2. Visualizing Input 3 (Total Integral Area)
    ax.fill_between(time_auc_processed, signal_auc_processed, color='violet', alpha=0.2, label=f'Total AUC (until {params["pct_integral_total"]}%)')

    # 3. Visualizing Input 5 (Total Time Line)
    ax.axvline(emission_time_total, color='green', linestyle='-', linewidth=2, label=f'Emission End ({params["pct_time_total"]}%)')

    # 4. Visualizing Input 6 (High Duration Range)
    if duration_high > 0:
        ax.axhline(thresh_val_time_high, color='orange', linestyle='--', linewidth=1)
        ax.fill_between(time_np, signal_smooth, thresh_val_time_high,
                        where=(signal_smooth >= thresh_val_time_high),
                        color='orange', alpha=0.3, label=f'High Duration (>{params["pct_time_high"]}%)')

    # 5. Visualizing Input 4 (High AUC - Overlay)
    if auc_high > 0:
         ax.fill_between(time_np, signal_smooth, thresh_val_integral_high,
                        where=(signal_smooth >= thresh_val_integral_high),
                        facecolor='none', hatch='///', edgecolor='red', alpha=0.5, label=f'High AUC Area (>{params["pct_integral_high"]}%)')
    
    # 6. Visualizing Input 7 (Max Emission Window)
    if max_window_auc > 0 and len(best_window_time) > 0:
        # Highlight the area
        ax.fill_between(best_window_time, best_window_signal, color='cyan', alpha=0.4, label=f'Max Emission ({target_window_dur}s)')
        # Add vertical boundaries for clarity
        ax.axvline(best_window_time[0], color='cyan', linestyle=':', linewidth=1)
        ax.axvline(best_window_time[-1], color='cyan', linestyle=':', linewidth=1)

    ax.set_title(f'Signal: {signal_name}', fontsize=10, weight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    return results, fig

# =============================================================================
# GUI CLASS (Graphical Interface)
# =============================================================================

class SixInputAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Signal Analyzer (7 Inputs)")
        self.root.geometry("600x800")

        # --- Variables ---
        self.file_path_var = tk.StringVar()

        # Independent Variables Defaults
        self.smooth_win_var = tk.IntVar(value=21)           
        self.pct_total_int_var = tk.DoubleVar(value=10.0)   
        self.pct_high_int_var = tk.DoubleVar(value=85.0)    
        self.pct_total_time_var = tk.DoubleVar(value=50.0)  
        self.pct_high_dur_var = tk.DoubleVar(value=75.0)    
        self.max_emission_win_var = tk.DoubleVar(value=20.0) 

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Signal Analyzer Pro", font=("Segoe UI", 16, "bold")).pack(pady=(0, 15))

        # --- Section 1: File ---
        file_frame = ttk.LabelFrame(main_frame, text="1. Data Source (Excel)", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        row1 = ttk.Frame(file_frame)
        row1.pack(fill=tk.X)
        ttk.Entry(row1, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(row1, text="Select File", command=self.browse_file).pack(side=tk.LEFT)

        # --- Section 2: Parameters ---
        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters (Inputs 2 - 7)", padding="15")
        param_frame.pack(fill=tk.X, pady=10)

        # Grid Helper
        def add_param_row(idx, label, var, row_idx, note=""):
            # Label
            lbl = ttk.Label(param_frame, text=f"{idx}. {label}", font=("Arial", 9, "bold"))
            lbl.grid(row=row_idx, column=0, sticky="w", pady=8, padx=5)

            # Entry
            ent = ttk.Entry(param_frame, textvariable=var, width=10, justify='center')
            ent.grid(row=row_idx, column=1, sticky="w", pady=8, padx=5)

            # Unit
            unit = "% Peak" 
            if idx == 2: unit = "(Window Size)"
            elif idx == 7: unit = "Seconds" 
            
            ttk.Label(param_frame, text=unit).grid(row=row_idx, column=2, sticky="w")

            # Note
            if note:
                ttk.Label(param_frame, text=note, foreground="gray", font=("Arial", 8)).grid(row=row_idx, column=3, sticky="w", padx=10)

        # 2. Smoothing
        add_param_row(2, "Smoothing Degree", self.smooth_win_var, 0, "Noise filter (Odd number)")

        # 3. Total Integral
        add_param_row(3, "Total Integral Cutoff", self.pct_total_int_var, 1, "Calc AUC until signal drops to X%")

        # 4. High AUC
        add_param_row(4, "High Integral Thresh", self.pct_high_int_var, 2, "Calc AUC for signal > X%")

        # 5. Total Emission Time
        add_param_row(5, "Emission Time Cutoff", self.pct_total_time_var, 3, "Time until signal drops to X%")

        # 6. High Duration
        add_param_row(6, "High Duration Thresh", self.pct_high_dur_var, 4, "Time width while signal > X%")
        
        # 7. Max Emission Window 
        add_param_row(7, "Max Emission Window", self.max_emission_win_var, 5, "Find max AUC in this time span")


        # --- Buttons ---
        btn_frame = ttk.Frame(main_frame, padding="10")
        btn_frame.pack(fill=tk.X, pady=10)

        run_btn = ttk.Button(btn_frame, text="RUN ANALYSIS", command=self.run_analysis)
        run_btn.pack(fill=tk.X, ipady=10)

        # --- Logs ---
        log_frame = ttk.LabelFrame(main_frame, text="Process Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=8, state='disabled', bg="#f0f0f0", font=("Consolas", 9))
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, ">> " + message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if filename:
            self.file_path_var.set(filename)

    def run_analysis(self):
        input_path = self.file_path_var.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid Excel file.")
            return

        # Gather Params
        try:
            params = {
                'smooth_window': self.smooth_win_var.get(),
                'pct_integral_total': self.pct_total_int_var.get(), 
                'pct_integral_high': self.pct_high_int_var.get(),   
                'pct_time_total': self.pct_total_time_var.get(),    
                'pct_time_high': self.pct_high_dur_var.get(),       
                'max_emission_window_dur': self.max_emission_win_var.get() 
            }
        except ValueError:
            messagebox.showerror("Error", "All numeric fields must contain valid numbers.")
            return

        self.log(f"Starting analysis on: {os.path.basename(input_path)}")
        self.log(f"Params: {params}")

        try:
            df = pd.read_excel(input_path)
            time_col = df.columns[0]

            # Output Setup
            folder_name = "Analysis_Results_7Inputs"
            output_dir = os.path.join(os.path.dirname(input_path), folder_name)
            if not os.path.exists(output_dir): os.makedirs(output_dir)

            all_results = []
            count = 0

            # Loop signals
            for sig_name in df.columns[1:]:
                sub_df = df[[time_col, sig_name]].dropna()
                if len(sub_df) < 5: continue

                try:
                    res, fig = analyze_signal_logic(sub_df[time_col], sub_df[sig_name], sig_name, params)
                    all_results.append(res)

                    # Save Plot
                    safe_name = "".join([c if c.isalnum() else "_" for c in str(sig_name)])
                    fig.savefig(os.path.join(output_dir, f"{safe_name}.png"))
                    plt.close(fig)

                    count += 1
                    if count % 5 == 0: self.log(f"Processed {count} signals...")

                except Exception as e:
                    self.log(f"Error processing {sig_name}: {e}")

            if all_results:
                out_file = os.path.join(output_dir, "Final_Data.xlsx")
                pd.DataFrame(all_results).to_excel(out_file, index=False)
                self.log("=" * 30)
                self.log(f"COMPLETED. Processed {count} signals.")
                self.log(f"Results saved in: {output_dir}")
                messagebox.showinfo("Success", f"Analysis Done!\nResults saved in folder: {folder_name}")
            else:
                self.log("No valid signals found.")
                messagebox.showwarning("Warning", "No valid signals found to analyze.")

        except Exception as e:
            messagebox.showerror("Critical Error", str(e))
            self.log(f"Critical Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SixInputAnalyzerApp(root)
    root.mainloop()