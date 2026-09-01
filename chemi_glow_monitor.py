# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import datetime
import json
import openpyxl

# ==============================================================================
# Helper Functions Section
# ==============================================================================

ROI_SAVE_FILE = "saved_roi_coords.json"
scaling_factor = 1.0  # Global variable for display scaling

def save_roi(roi_rect):
    """Saves the ROI area to a JSON file."""
    try:
        with open(ROI_SAVE_FILE, 'w') as f:
            json.dump({'roi': roi_rect}, f)
        print(f"ROI area saved to {ROI_SAVE_FILE} file.")
        return True
    except Exception as e:
        print(f"Error saving ROI: {e}")
        return False

def load_roi():
    """Loads the ROI area from the JSON file."""
    if not os.path.exists(ROI_SAVE_FILE):
        return None
    try:
        with open(ROI_SAVE_FILE, 'r') as f:
            data = json.load(f)
            print(f"ROI area loaded from {ROI_SAVE_FILE} file.")
            return tuple(data['roi'])
    except Exception as e:
        print(f"Error loading ROI: {e}")
        return None

def resize_frame_for_display(frame, max_width=1280, max_height=720):
    global scaling_factor
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        scaling_factor = 1.0
        return frame
    ratio = min(max_width / w, max_height / h)
    scaling_factor = 1 / ratio
    new_dim = (int(w * ratio), int(h * ratio))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

# ==============================================================================
# New Function: Smart ROI Detection
# ==============================================================================
def find_roi_automatically(frame, seed_roi, threshold_multiplier=1.0):
    """
    Finds the main luminous area using a seed region.
    """
    x1, y1, x2, y2 = seed_roi
    
    # Ensure coordinate validity
    if x1 >= x2 or y1 >= y2:
        return None
        
    # Crop the seed region from the main image
    seed_region = frame[y1:y2, x1:x2]
    if seed_region.size == 0: return None

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_seed = gray_frame[y1:y2, x1:x2]

    # Calculate seed region statistics
    mean_val, std_dev = cv2.meanStdDev(gray_seed)
    mean_val = mean_val[0][0]
    std_dev = std_dev[0][0]

    # If the standard deviation is very low (uniform area), consider a minimum value
    if std_dev < 5: std_dev = 5

    # Calculate threshold: anything with brightness close to the seed's mean
    # Formula: (Mean) minus (StdDev * multiplier)
    # For chemiluminescence where light is in a dark background, the lower limit is important.
    lower_thresh = mean_val - (threshold_multiplier * std_dev)
    
    # Prevent very low thresholds (background noise)
    lower_thresh = max(lower_thresh, 15) 

    print(f"Seed Stats -> Mean: {mean_val:.2f}, Std: {std_dev:.2f}, Calc Thresh: {lower_thresh:.2f}")

    # Apply Thresholding
    _, binary_mask = cv2.threshold(gray_frame, lower_thresh, 255, cv2.THRESH_BINARY)

    # Noise Cleanup (Morphological Operations)
    kernel = np.ones((5, 5), np.uint8)
    # Close holes inside the luminous area
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    # Remove small surrounding noise
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No area found with this threshold.")
        return None
        
    # Find the largest contour (assumes the main reaction is the largest luminous spot)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding
    pad = 10
    h_img, w_img = frame.shape[:2]
    final_x1 = max(0, x - pad)
    final_y1 = max(0, y - pad)
    final_x2 = min(w_img, x + w + pad)
    final_y2 = min(h_img, y + h + pad)

    return (final_x1, final_y1, final_x2, final_y2)

# ==============================================================================
# ROI Selection Class (Updated with 'A' key)
# ==============================================================================

class InteractiveROISelector:
    def __init__(self, cap, initial_frame, video_path):
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self.video_path = video_path

        self.orig_img = initial_frame.copy()
        self.img_to_show = initial_frame.copy()

        ih, iw = self.orig_img.shape[:2]
        saved_roi = load_roi()
        if saved_roi:
            self.ix, self.iy, self.ex, self.ey = saved_roi
        else:
            self.ix, self.iy, self.ex, self.ey = iw // 4, ih // 4, 3 * iw // 4, 3 * ih // 4

        self.drag = False
        self.resizing = False
        self.move_start = (0, 0)
        self.resizing_corner = None
        self.thickness = 2
        self.manual_color = (0, 255, 0)   # Green for manual selection
        self.auto_color = (255, 100, 0)   # Blue for auto-detection
        self.corner_size = 15
        self.win_name = "Select ROI (Press 'A' for Auto-Detect)"
        self.trackbar_name = "Frame"
        self.finished = False
        self.is_seeking = False
        
        # Variable to store machine-detected ROI
        self.detected_roi = None 

    def seek_frame(self, frame_number):
        if self.is_seeking: return
        self.is_seeking = True

        frame_number = int(np.clip(frame_number, 0, self.total_frames - 1))
        if frame_number != self.current_frame_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.orig_img = resize_frame_for_display(frame)
                self.current_frame_index = frame_number
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        cv2.setTrackbarPos(self.trackbar_name, self.win_name, self.current_frame_index)
        self.draw_box()
        self.is_seeking = False

    def on_trackbar_change(self, trackbar_value):
        self.seek_frame(trackbar_value)
    
    def inside_roi(self, x, y): return self.ix < x < self.ex and self.iy < y < self.ey
    
    def corner_hit(self, x, y):
        if abs(x - self.ix) < self.corner_size and abs(y - self.iy) < self.corner_size: return 'tl'
        if abs(x - self.ex) < self.corner_size and abs(y - self.ey) < self.corner_size: return 'br'
        if abs(x - self.ix) < self.corner_size and abs(y - self.ey) < self.corner_size: return 'bl'
        if abs(x - self.ex) < self.corner_size and abs(y - self.iy) < self.corner_size: return 'tr'
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clear previous auto-detected area on click to avoid user confusion
            self.detected_roi = None 
            
            corner = self.corner_hit(x, y)
            if corner:
                self.resizing = True
                self.resizing_corner = corner
            elif self.inside_roi(x, y):
                self.drag = True
                self.move_start = (x, y)
            else:
                self.ix, self.iy = x, y
                self.ex, self.ey = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                dx, dy = x - self.move_start[0], y - self.move_start[1]
                h, w = self.orig_img.shape[:2]
                dw, dh = self.ex - self.ix, self.ey - self.iy
                self.ix = np.clip(self.ix + dx, 0, w - dw)
                self.iy = np.clip(self.iy + dy, 0, h - dh)
                self.ex, self.ey = self.ix + dw, self.iy + dh
                self.move_start = (x, y)
            elif self.resizing:
                h, w = self.orig_img.shape[:2]
                if self.resizing_corner == 'tl': self.ix, self.iy = np.clip(x, 0, self.ex-1), np.clip(y, 0, self.ey-1)
                elif self.resizing_corner == 'br': self.ex, self.ey = np.clip(x, self.ix+1, w-1), np.clip(y, self.iy+1, h-1)
                elif self.resizing_corner == 'bl': self.ix, self.ey = np.clip(x, 0, self.ex-1), np.clip(y, self.iy+1, h-1)
                elif self.resizing_corner == 'tr': self.ex, self.iy = np.clip(x, self.ix+1, w-1), np.clip(y, 0, self.ey-1)
            elif flags & cv2.EVENT_FLAG_LBUTTON:
                self.ex, self.ey = x, y
            self.draw_box()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = False
            self.resizing = False
            self.resizing_corner = None
            if self.ix > self.ex: self.ix, self.ex = self.ex, self.ix
            if self.iy > self.ey: self.iy, self.ey = self.ey, self.iy
            self.draw_box()

    def draw_box(self, message=None, msg_color=(0, 255, 255)):
        img2 = self.orig_img.copy()
        
        # Draw green box (Manual Selection / Seed)
        cv2.rectangle(img2, (int(self.ix), int(self.iy)), (int(self.ex), int(self.ey)), self.manual_color, self.thickness)
        
        # Draw blue box (Smart Detection) if it exists
        if self.detected_roi:
            dx1, dy1, dx2, dy2 = self.detected_roi
            cv2.rectangle(img2, (dx1, dy1), (dx2, dy2), self.auto_color, self.thickness + 1)
            cv2.putText(img2, "Auto-Detected Area", (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.auto_color, 2)

        info_text = f"Frame: {self.current_frame_index + 1}/{self.total_frames}"
        cv2.putText(img2, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        controls = "SPACE: Accept | 'A': Auto-Detect | 'R': Reset | 'S': Save | 'L': Load"
        cv2.putText(img2, controls, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if message:
            cv2.putText(img2, message, (int(self.ix), int(self.iy) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

        self.img_to_show = img2
        cv2.imshow(self.win_name, self.img_to_show)

    def trigger_auto_detect(self):
        """Run smart detection algorithm on the main frame"""
        # 1. Convert display coordinates to original coordinates
        orig_ix = int(self.ix * scaling_factor)
        orig_iy = int(self.iy * scaling_factor)
        orig_ex = int(self.ex * scaling_factor)
        orig_ey = int(self.ey * scaling_factor)
        
        seed_roi = (orig_ix, orig_iy, orig_ex, orig_ey)
        
        # 2. Get main frame (full resolution)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, full_frame = self.cap.read()
        
        if ret:
            # 3. Run smart function
            result = find_roi_automatically(full_frame, seed_roi)
            if result:
                rx1, ry1, rx2, ry2 = result
                # 4. Convert result to display coordinates for user display
                disp_x1 = int(rx1 / scaling_factor)
                disp_y1 = int(ry1 / scaling_factor)
                disp_x2 = int(rx2 / scaling_factor)
                disp_y2 = int(ry2 / scaling_factor)
                
                self.detected_roi = (disp_x1, disp_y1, disp_x2, disp_y2)
                self.draw_box("Auto-Detect Success! Press SPACE to use.", self.auto_color)
            else:
                self.draw_box("Failed to detect area. Try different seed.", (0, 0, 255))
        else:
            self.draw_box("Frame read error.", (0, 0, 255))

    def get_roi(self):
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        if self.total_frames > 1:
            cv2.createTrackbar(self.trackbar_name, self.win_name, 0, self.total_frames - 1, self.on_trackbar_change)
        self.draw_box()

        final_roi_coords = None

        while True:
            cv2.imshow(self.win_name, self.img_to_show)
            k = cv2.waitKey(20) & 0xFF

            if k == 32:  # SPACE
                self.finished = True
                # Priority is given to the detected ROI (blue)
                if self.detected_roi:
                    final_roi_coords = self.detected_roi
                else:
                    final_roi_coords = (int(self.ix), int(self.iy), int(self.ex), int(self.ey))
                break
            
            elif k == ord('a') or k == ord('A'): # Smart detection key
                self.trigger_auto_detect()

            elif k == ord('q'):
                final_roi_coords = None
                break
            elif k == ord('r'):
                h, w = self.orig_img.shape[:2]
                self.ix, self.iy, self.ex, self.ey = w//4, h//4, 3*w//4, 3*h//4
                self.detected_roi = None
                self.draw_box("Reset")
            elif k == ord('s'):
                # Save what is currently selected
                to_save = self.detected_roi if self.detected_roi else (int(self.ix), int(self.iy), int(self.ex), int(self.ey))
                save_roi(to_save)
                self.draw_box("Saved!")
            elif k == ord('l'):
                loaded = load_roi()
                if loaded:
                    # Assume the loaded ROI is the final one, so set it as manual
                    # (since the user might want to change it)
                    self.ix, self.iy, self.ex, self.ey = loaded
                    self.detected_roi = None # Clear automatic mode
                    self.draw_box("Loaded!")

        cv2.destroyWindow(self.win_name)
        return final_roi_coords

# ==============================================================================
# Analysis and Saving (Modified version for brightness intensity)
# ==============================================================================

def analyze_video_colors(video_path, roi):
    """
    Video analysis for measuring brightness intensity (Luminance)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ROI coordinates (original scale)
    x1, y1, x2, y2 = roi
    
    # Ensure correct order of coordinates
    x_start, x_end = min(x1, x2), max(x1, x2)
    y_start, y_end = min(y1, y2), max(y1, y2)

    live_win = "Live Intensity Analysis"
    cv2.namedWindow(live_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(live_win, 800, 600)

    # Create progress window
    progress_root = tk.Toplevel()
    progress_root.title("Processing...")
    progress_root.geometry("400x100")
    tk.Label(progress_root, text="Analyzing Luminance...", font=("Arial", 10)).pack(pady=5)
    progress = ttk.Progressbar(progress_root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=5)
    progress["maximum"] = total_frames

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        roi_frame = frame[y_start:y_end, x_start:x_end]
        if roi_frame.size == 0:
            frame_count += 1
            continue

        # Calculate brightness intensity (Grayscale)
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_roi)
        timestamp = frame_count / fps

        results.append({
            "frame": frame_count,
            "timestamp_sec": timestamp,
            "avg_intensity": avg_intensity
        })

        # Live display (every 5 frames for speed, or every frame)
        if frame_count % 2 == 0:
            disp_frame = resize_frame_for_display(frame.copy())
            
            # Display coordinates
            dx1 = int(x_start / scaling_factor)
            dy1 = int(y_start / scaling_factor)
            dx2 = int(x_end / scaling_factor)
            dy2 = int(y_end / scaling_factor)

            cv2.rectangle(disp_frame, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
            
            # Text and info
            cv2.putText(disp_frame, f"Intensity: {avg_intensity:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(disp_frame, f"Time: {timestamp:.1f}s", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.imshow(live_win, disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 10 == 0:
            progress['value'] = frame_count
            progress_root.update()

    cap.release()
    cv2.destroyWindow(live_win)
    progress_root.destroy()
    return pd.DataFrame(results), fps

def save_to_master_excel(df, video_file, timestamp_str):
    master_file = "analysis_results/master_intensity_log.xlsx"
    sheet_name = "IntensityResults"
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Rename columns for uniqueness
    col_time = f"t_{base_name}"
    col_int = f"int_{base_name}"
    
    df_to_save = df.rename(columns={
        "timestamp_sec": col_time,
        "avg_intensity": col_int
    })[['frame', col_time, col_int]] # Keeping the frame is also useful

    try:
        if os.path.exists(master_file):
            # Use openpyxl engine for better appending
            with pd.ExcelWriter(master_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                # Finding the next empty column is a bit complex, simplest way is read and rewrite
                # But to avoid slowdowns in large files, a separate file or simpler method is suggested for now
                # Implementing the append to new file method here which is safer
                pass 
            
            # Simpler method: read previous file and merge (not suitable for very large files but works)
            existing_df = pd.read_excel(master_file)
            final_df = pd.concat([existing_df, df_to_save], axis=1)
        else:
            final_df = df_to_save

        os.makedirs(os.path.dirname(master_file), exist_ok=True)
        final_df.to_excel(master_file, index=False)
        print(f"Master log updated: {master_file}")

    except Exception as e:
        print(f"Warning: Could not update master excel. Saving local only. Error: {e}")

def main():
    root = tk.Tk()
    root.withdraw()

    while True:
        video_file = filedialog.askopenfilename(
            title="Select Video for Luminance Analysis",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not video_file: break

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            continue

        ret, first_frame = cap.read()
        if not ret: continue
        
        # 1. Select ROI
        disp_frame = resize_frame_for_display(first_frame)
        selector = InteractiveROISelector(cap, disp_frame, video_file)
        roi_disp = selector.get_roi() # Output (x1, y1, x2, y2) in display scale
        cap.release()

        if not roi_disp:
            continue

        # 2. Convert coordinates to original scale
        dx1, dy1, dx2, dy2 = roi_disp
        ox1 = int(dx1 * scaling_factor)
        oy1 = int(dy1 * scaling_factor)
        ox2 = int(dx2 * scaling_factor)
        oy2 = int(dy2 * scaling_factor)
        final_roi = (ox1, oy1, ox2, oy2)

        # 3. Analysis
        df, fps = analyze_video_colors(video_file, final_roi)

        if df is not None and not df.empty:
            # Save results
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            out_dir = "analysis_results"
            os.makedirs(out_dir, exist_ok=True)
            
            # Individual Excel
            excel_name = os.path.join(out_dir, f"intensity_{base_name}_{ts}.xlsx")
            df.to_excel(excel_name, index=False)
            
            # Master Excel
            save_to_master_excel(df, video_file, ts)

            # Draw plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['timestamp_sec'], df['avg_intensity'], label='Intensity', color='black')
            plt.title(f"Chemiluminescence Intensity: {base_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Avg Pixel Intensity (0-255)")
            plt.grid(True)
            plt.legend()
            
            plot_name = os.path.join(out_dir, f"plot_{base_name}_{ts}.png")
            plt.savefig(plot_name)
            plt.close()

            messagebox.showinfo("Done", f"Analysis Complete!\nSaved to: {excel_name}")
        
    root.destroy()

if __name__ == "__main__":
    main()