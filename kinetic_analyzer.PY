import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import r2_score

# =============================================================================
# Configuration Defaults
# =============================================================================
MIN_DATA_POINTS_FOR_FIT = 5
CONFIDENCE_LEVEL = 0.95

# =============================================================================
# 1. Define Kinetic Model Functions
# =============================================================================

# --- Growth Models (0 to 1) ---
def model_growth_first_order(t, k):
    return 1.0 - np.exp(-k * t)

def model_growth_second_order(t, k):
    return (k * t) / (1.0 + k * t)

def model_growth_avrami(t, k, n):
    return 1.0 - np.exp(-np.clip((k * t)**n, -700, 700))

def model_growth_sigmoidal(t, k, n):
    # k here represents the time to reach half max
    tn = t**n
    kn = k**n
    return tn / (kn + tn)

# --- Decay Models (1 to 0) ---
def model_decay_first_order(t, k):
    return np.exp(-k * t)

def model_decay_second_order(t, k):
    return 1 / (1 + k * t)

def model_decay_avrami(t, k, n):
    return np.exp(-np.clip((k * t)**n, -700, 700))

# =============================================================================
# 2. Define Helper Functions (AIC, Half-Life)
# =============================================================================

def calculate_aic(n, k, y_true, y_pred):
    if n <= k + 1: return np.inf
    rss = np.sum((y_true - y_pred)**2)
    if rss == 0: return -np.inf
    return n * np.log(rss / n) + 2 * k

def get_thalf(model_name, params, mode):
    """Calculates t_1/2 based on model parameters."""
    try:
        if mode == 'growth':
            if "First-Order" in model_name:
                return np.log(2) / params[0]
            elif "Second-Order" in model_name:
                return 1.0 / params[0]
            elif "Avrami" in model_name:
                k, n = params
                return (np.log(2)**(1/n)) / k
            elif "Sigmoidal" in model_name:
                return params[0]
        else: # Decay
            if "First-Order" in model_name:
                return np.log(2) / params[0]
            elif "Second-Order" in model_name:
                return 1.0 / params[0]
            elif "Avrami" in model_name:
                k, n = params
                return (np.log(2)**(1/n)) / k
    except:
        return np.nan
    return np.nan

# =============================================================================
# 3. Setup Models Dictionary
# =============================================================================

def get_all_models():
    """Returns a dictionary containing all growth and decay models."""
    return {
        'growth': {
            "Growth First-Order":  {'func': model_growth_first_order,  'p0': [0.1],      'bounds': ([0], [np.inf]),      'params': ['k']},
            "Growth Second-Order": {'func': model_growth_second_order, 'p0': [0.1],      'bounds': ([0], [np.inf]),      'params': ['k']},
            "Growth Avrami":       {'func': model_growth_avrami,       'p0': [0.1, 1.5], 'bounds': ([0, 0], [np.inf, 10]),'params': ['k', 'n']},
            "Growth Sigmoidal":    {'func': model_growth_sigmoidal,    'p0': [10.0, 2.0],'bounds': ([0, 0], [np.inf, 10]),'params': ['k_time', 'n']}
        },
        'decay': {
            "Decay First-Order":   {'func': model_decay_first_order,   'p0': [0.1],      'bounds': ([0], [np.inf]),      'params': ['k']},
            "Decay Second-Order":  {'func': model_decay_second_order,  'p0': [0.1],      'bounds': ([0], [np.inf]),      'params': ['k']},
            "Decay Avrami":        {'func': model_decay_avrami,        'p0': [0.1, 1.5], 'bounds': ([0, 0], [np.inf, 10]),'params': ['k', 'n']}
        }
    }

# =============================================================================
# 4. Main Analysis Function
# =============================================================================

def run_analysis(input_path, selected_model_names, analysis_mode, log_callback):
    """
    Core analysis function called by the GUI.
    """
    log_callback(f"Starting analysis in mode: {analysis_mode.upper()}")
    
    # Load Data
    try:
        df = pd.read_excel(input_path, header=0)
    except Exception as e:
        log_callback(f"Error reading Excel file: {e}")
        return

    # Setup Output
    output_dir = f"Analysis_Results_{analysis_mode.capitalize()}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_tests_summary = []
    alpha = 1.0 - CONFIDENCE_LEVEL
    
    # Filter models based on selection
    all_models = get_all_models()
    models_to_run = {name: info for name, info in all_models[analysis_mode].items() if name in selected_model_names}

    if not models_to_run:
        log_callback("Error: No models were selected/found for this mode.")
        return

    # Process each column
    for i in range(1, len(df.columns)):
        test_name = df.columns[i]
        log_callback(f"Processing {test_name}...")

        # Data Cleaning
        x_raw = df.iloc[:, 0].values.astype(float)
        y_raw = df.iloc[:, i].values.astype(float)
        finite_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        x_raw, y_raw = x_raw[finite_mask], y_raw[finite_mask]

        if len(y_raw) < MIN_DATA_POINTS_FOR_FIT:
            continue

        # Find Peak
        peak_index = np.argmax(y_raw)
        peak_val = y_raw[peak_index]
        peak_time = x_raw[peak_index]

        # Calculate AUCs
        total_auc = np.trapz(y_raw, x_raw)
        x_to_peak = x_raw[:peak_index+1]
        y_to_peak = y_raw[:peak_index+1]
        auc_to_peak = np.trapz(y_to_peak, x_to_peak) if len(x_to_peak) > 1 else 0

        # Prepare Data for Fitting (Growth vs Decay logic)
        if analysis_mode == 'growth':
            x_target = x_raw[:peak_index+1]
            y_target = y_raw[:peak_index+1]
            
            if len(y_target) < MIN_DATA_POINTS_FOR_FIT:
                log_callback(f"  Skipping {test_name}: Not enough growth points.")
                continue
                
            min_val = np.min(y_target)
            denom = peak_val - min_val
            y_fit_norm = (y_target - min_val) / denom if denom > 1e-9 else y_target
            x_fit_shifted = x_target - x_target[0]
            
        else: # Decay
            x_target = x_raw[peak_index:]
            y_target = y_raw[peak_index:]
            
            if len(y_target) < MIN_DATA_POINTS_FOR_FIT:
                log_callback(f"  Skipping {test_name}: Not enough decay points.")
                continue

            end_val = y_target[-1]
            denom = peak_val - end_val
            y_fit_norm = (y_target - end_val) / denom if denom > 1e-9 else y_target
            x_fit_shifted = x_target - x_target[0]

        # Fitting Loop
        test_plot_dir = os.path.join(output_dir, str(test_name).replace('/', '_'))
        os.makedirs(test_plot_dir, exist_ok=True)
        
        summary = {
            'Test': test_name,
            'Peak_Value': peak_val,
            'Peak_Time': peak_time,
            'Total_AUC': total_auc,
            'AUC_to_Peak': auc_to_peak,
            'Best_Model': "None"
        }

        best_aic = np.inf
        best_model_name = None
        best_params = None
        best_pcov = None

        for model_name, model_info in models_to_run.items():
            try:
                popt, pcov = curve_fit(model_info['func'], x_fit_shifted, y_fit_norm, 
                                     p0=model_info['p0'], bounds=model_info['bounds'], maxfev=5000)
                
                y_pred = model_info['func'](x_fit_shifted, *popt)
                r2 = r2_score(y_fit_norm, y_pred)
                aic = calculate_aic(len(y_fit_norm), len(popt), y_fit_norm, y_pred)

                if aic < best_aic:
                    best_aic = aic
                    best_model_name = model_name
                    best_params = popt
                    best_pcov = pcov
                
                # Plotting
                plt.figure()
                plt.scatter(x_fit_shifted, y_fit_norm, c='r', s=10, label='Data')
                x_smooth = np.linspace(0, max(x_fit_shifted), 100)
                plt.plot(x_smooth, model_info['func'](x_smooth, *popt), 'b-', label='Fit')
                plt.title(f"{model_name} (R2={r2:.3f})")
                plt.legend()
                plt.savefig(os.path.join(test_plot_dir, f"{model_name}.png"))
                plt.close()

            except Exception as e:
                # log_callback(f"  Fit failed for {model_name}: {e}")
                pass

        # Save Best Model Stats
        if best_model_name:
            summary['Best_Model'] = best_model_name
            summary['Best_Model_AIC'] = best_aic
            
            perr = np.sqrt(np.diag(best_pcov)) if best_pcov is not None else [np.nan]*len(best_params)
            dof = max(1, len(y_fit_norm) - len(best_params))
            t_crit = np.abs(t.ppf(alpha / 2, dof))
            
            for idx, p_name in enumerate(models_to_run[best_model_name]['params']):
                val = best_params[idx]
                ci = t_crit * perr[idx]
                summary[f'Param_{p_name}'] = val
                summary[f'Param_{p_name}_CI'] = ci
            
            summary['Response_Time_t50'] = get_thalf(best_model_name, best_params, analysis_mode)

        all_tests_summary.append(summary)

    # Save Final Excel
    if all_tests_summary:
        df_res = pd.DataFrame(all_tests_summary)
        # Order columns nicely
        first_cols = ['Test', 'Peak_Value', 'Peak_Time', 'Total_AUC', 'AUC_to_Peak', 'Best_Model', 'Response_Time_t50']
        cols = first_cols + [c for c in df_res.columns if c not in first_cols]
        # Only reindex if columns exist
        if set(first_cols).issubset(df_res.columns):
            df_res = df_res[cols]
        
        save_path = os.path.join(output_dir, "Summary_Results.xlsx")
        df_res.to_excel(save_path, index=False)
        log_callback(f"Done! Summary saved to: {save_path}")
    else:
        log_callback("Analysis finished, but no valid data was processed.")
