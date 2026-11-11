import os
import re
import glob
import pandas as pd
import numpy as np

def process_centroid_distances(folder_path=".", output_csv="centroid_total_distances.csv"):
    results = []
    
    # Get all CSV files in target directory
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f) != output_csv]
    
    print(f"Found {len(csv_files)} file(s) to process.\n")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Extract ID between 'aug_' and 'DLC'
        match = re.search(r'aug_(.*?)DLC', filename)
        if not match:
            print(f"[Skipped - Pattern Mismatch]: {filename}")
            continue
            
        file_id = match.group(1)
        
        try:
            df = pd.read_csv(file_path)
            
            # Determine centroid coordinate columns
            if 'centroid_x_final' in df.columns and 'centroid_y_final' in df.columns:
                x_col, y_col = 'centroid_x_final', 'centroid_y_final'
            elif 'center_x_final' in df.columns and 'center_y_final' in df.columns:
                x_col, y_col = 'center_x_final', 'center_y_final'
            else:
                print(f"[Skipped - Missing Centroid Columns]: {filename}")
                continue
            
            # 1. Compute total distance in pixels
            dx = df[x_col].diff()
            dy = df[y_col].diff()
            dist_px = np.sqrt(dx**2 + dy**2).sum()
            
            # 2. Extract conversion factor (cm per pixel)
            cm_per_px = None
            if 'speed_cm_per_frame' in df.columns and 'speed_px_per_frame' in df.columns:
                ratio = (df['speed_cm_per_frame'] / df['speed_px_per_frame']).dropna()
                if not ratio.empty:
                    cm_per_px = ratio.iloc[0]
            elif 'speed_cm_per_s' in df.columns and 'speed_px_per_s' in df.columns:
                ratio = (df['speed_cm_per_s'] / df['speed_px_per_s']).dropna()
                if not ratio.empty:
                    cm_per_px = ratio.iloc[0]
            
            # 3. Compute total distance in cm
            dist_cm = dist_px * cm_per_px if cm_per_px is not None else np.nan
            
            results.append({
                'id': file_id,
                'total_distance_px': dist_px,
                'cm_per_px': cm_per_px,
                'total_distance_cm': dist_cm
            })
            
            cm_str = f"{dist_cm:.2f} cm" if cm_per_px else "N/A"
            print(f"[Processed]: {file_id} -> {dist_px:.2f} px ({cm_str})")
            
        except Exception as e:
            print(f"[Error processing {filename}]: {e}")
            
    # Export results
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSaved results for {len(results)} file(s) to '{output_csv}'.")
    else:
        print("\nNo files processed.")

if __name__ == "__main__":
    process_centroid_distances("all_augmented_csv/results_with_vertices")

