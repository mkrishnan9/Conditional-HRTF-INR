import os
import glob
import pickle as pkl
import SOFAdatasets
from tqdm import tqdm
import pandas as pd
import numpy as np

out_dir = "/export/mkrishn9/hrtf_field/preprocessed_hrirs"
hutubs_folder_path = "/export/mkrishn9/hrtf_field/hutubs"
anthropometry_csv_path = os.path.join(hutubs_folder_path, "AntrhopometricMeasures.csv")


os.makedirs(out_dir, exist_ok=True)

def save_hrir_and_raw_anthro_as_pkl():
    # --- 1. Load Anthropometry Data ---
    print(f"Loading anthropometry from: {anthropometry_csv_path}")
    try:
        # Read CSV, assume first row is header, first column is SubjectID
        anthro_df = pd.read_csv(anthropometry_csv_path, index_col=0) # Use SubjectID as index
        # Ensure index is string type if subject IDs from SOFAdataset are strings
        anthro_df.index = anthro_df.index.map(str)
        # Convert all data columns to numeric, coercing errors to NaN (handle NaNs later if needed)
        # It's generally good practice to ensure numeric types here
        numeric_cols = anthro_df.columns
        anthro_df[numeric_cols] = anthro_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        print(f"Loaded anthropometry for {len(anthro_df)} subjects.")
        print(f"Anthropometry features: {list(anthro_df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: Anthropometry CSV file not found at {anthropometry_csv_path}")
        return
    except Exception as e:
        print(f"ERROR: Could not load or process anthropometry CSV: {e}")
        return


    dataset_dict = {"hutubs": "HUTUBS"} # Extend this if you process more datasets with CSVs
    for name in list(dataset_dict.keys()):
        print(f"Processing dataset: {name}")
        try:
            dataset_obj = getattr(SOFAdatasets, dataset_dict[name])()
        except Exception as e:
            print(f"ERROR: Could not instantiate SOFAdataset for {name}: {e}")
            continue

        print(f"Preprocessing {len(dataset_obj)} ears...")
        for idx in tqdm(range(len(dataset_obj))):
            try:
                location, hrir = dataset_obj[idx]

                subject_idx = idx // 2
                subject_id = str(dataset_obj._get_subject_ID(subject_idx))

                if subject_id in anthro_df.index:
                    full_row = anthro_df.loc[subject_id].astype(np.float32)

                    # Common features (modify this list if the actual feature names differ)
                    common_feats = ["x1","x2","x3","x4","x5","x6","x7","x8","x9",
                                    "x12","x14","x16","x17"]

                    if idx % 2 == 0:  # Even index: left ear
                        side_feats = [f"L_d{i}" for i in range(1,11)] + ["L_theta1", "L_theta2"]
                    else:  # Odd index: right ear
                        side_feats = [f"R_d{i}" for i in range(1,11)] + ["R_theta1", "R_theta2"]

                    selected_feats = common_feats + side_feats

                    # Ensure all selected features are present
                    missing = set(selected_feats) - set(anthro_df.columns)
                    if missing:
                        print(f"Missing columns for subject {subject_id}: {missing}")
                        continue

                    raw_anthro_data = full_row[selected_feats].values

                    # Replace NaNs if any
                    if np.isnan(raw_anthro_data).any():
                        print(f"Skipping subject {subject_id}, ear index {idx} due to NaNs in anthropometry.")
                        continue

                else:
                    print(f"Warning: Subject ID {subject_id} not found. Saving placeholder.")
                    raw_anthro_data = np.zeros(len(common_feats) + 12, dtype=np.float32)

                filename = f"{name}_{idx:03d}.pkl"
                data_to_save = (location, hrir, raw_anthro_data)
                with open(os.path.join(out_dir, filename), 'wb') as handle:
                    pkl.dump(data_to_save, handle, protocol=pkl.HIGHEST_PROTOCOL)

            except Exception as e:
                print(f"ERROR processing index {idx} for dataset {name}: {e}")
                continue

if __name__ == "__main__":
    save_hrir_and_raw_anthro_as_pkl()
    print("Preprocessing finished (saved RAW anthropometry).")
