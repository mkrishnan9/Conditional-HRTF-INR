import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.io
import SOFAdatasets
import h5py

OUT_DIR = "/export/mkrishn9/hrtf_field/preprocessed_hrirs_common"

DATASET_PATHS = {
    "scut": "/export/mkrishn9/hrtf_field/scut",
    "chedar": "/export/mkrishn9/hrtf_field/chedar",
    "cipic": "/export/mkrishn9/hrtf_field/cipic",
    "ari": "/export/mkrishn9/hrtf_field/ari",
    "hutubs": "/export/mkrishn9/hrtf_field/hutubs"
}

os.makedirs(OUT_DIR, exist_ok=True)


def load_hutubs_anthropometry(subject_id, ear_side, anthro_df):
    """
    Loads common anthropometry for a HUTUBS subject. Skips if NaN.
    """
    # Define the common set of features
    #common_feats = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x12", "x14", "x16", "x17"]
    common_feats = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
    if ear_side == 'left':
        side_feats = [f"L_d{i}" for i in range(1, 9)] + ["L_theta1", "L_theta2"]
    else:
        side_feats = [f"R_d{i}" for i in range(1, 9)] + ["R_theta1", "R_theta2"]

    selected_feats = common_feats + side_feats


    if subject_id not in anthro_df.index:
        print(f"Warning: Subject ID {subject_id} not found for HUTUBS. Skipping.")
        return None


    raw_anthro_data = anthro_df.loc[subject_id, selected_feats].values
    if np.isnan(raw_anthro_data).any():
        print(f"Warning: NaNs in anthropometry for HUTUBS subject {subject_id}. Skipping.")
        return None

    return raw_anthro_data.astype(np.float32)

def load_scut_anthropometry(subject_id, ear_side, anthro_df):
    """
    Loads common anthropometry for a Scut subject. Skips if NaN.
    """
    # Define the common set of features
    common_feats = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
    if ear_side == 'left':
        # Use d1-d8 to match CIPIC
        side_feats = [f"d{i}_l" for i in range(1, 9)] + ["theta1_l", "theta2_l"]
    else:
        side_feats = [f"d{i}_r" for i in range(1, 9)] + ["theta1_r", "theta2_r"]

    selected_feats = common_feats + side_feats


    if subject_id not in anthro_df.index:
        print(f"Warning: Subject ID {subject_id} not found for SCUT. Skipping.")
        return None


    raw_anthro_data = anthro_df.loc[subject_id, selected_feats].values
    if np.isnan(raw_anthro_data).any():
        print(f"Warning: NaNs in anthropometry for SCUT subject {subject_id}. Skipping.")
        return None

    raw_anthro_data[:17] /= 10.0

    return raw_anthro_data.astype(np.float32)

def load_chedar_anthropometry(subject_id, ear_side, anthro_df):
    """
    Loads common anthropometry for a CHEDAR subject. Skips if NaN.
    """
    # Define the common set of features
    common_feats = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    side_feats = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "t1", "t2"]


    d7_idx = side_feats.index('d7')
    side_feats_part1 = side_feats[:d7_idx + 1]
    side_feats_part2 = side_feats[d7_idx + 1:]


    selected_feats = common_feats + side_feats

    if subject_id not in anthro_df.index:
        print(f"Warning: Subject ID {subject_id} not found for chedar. Skipping.")
        return None


    common_data = anthro_df.loc[subject_id, common_feats].values
    side_data_1 = anthro_df.loc[subject_id, side_feats_part1].values
    side_data_2 = anthro_df.loc[subject_id, side_feats_part2].values

    # Concatenate the data parts with the new constant values
    final_anthro_data = np.concatenate([
        common_data,  # x1 through x8
        [34.5],       # Constant value after x8
        side_data_1,  # d1 through d7
        [1.02],       # Constant value after d7
        side_data_2   # t1 through t2
    ])



    return final_anthro_data.astype(np.float32)


def load_cipic_anthropometry(subject_id, ear_side, anthro_data):
    """
    Loads and maps common anthropometry for a CIPIC subject. Skips if NaN.
    """
    try:
        # The ID from the SOFA file that we need to find in the .mat 'id' variable
        id_to_find = int(subject_id)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse CIPIC subject ID '{subject_id}'. Skipping.")
        return None

    # Get the list of subject IDs from the .mat file's 'id' variable
    all_mat_ids = anthro_data['id'].flatten()

    # Find the row index where the subject ID is located
    indices = np.where(all_mat_ids == id_to_find)[0]

    # If the ID wasn't found in the list, skip this subject
    if indices.size == 0:
        return None


    subject_idx = indices[0]

    D = anthro_data['D']
    X = anthro_data['X']
    theta = anthro_data['theta']


    if subject_idx >= X.shape[0] or np.isnan(X[subject_idx, :]).all():
        print(subject_idx)
        print(f"Warning: Anthropometry not available for CIPIC subject {subject_id}. Skipping.")
        return None


    #hutubs_x_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 16]
    hutubs_x_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    common_values = X[subject_idx, hutubs_x_indices]


    if ear_side == 'left':
        d_values = D[subject_idx, :8]
        theta_values = theta[subject_idx, :2]
    else:
        d_values = D[subject_idx, 8:16]
        theta_values = theta[subject_idx, 2:4]

    anthropometry_vector = np.concatenate([
        common_values,
        d_values,
        theta_values
    ]).astype(np.float32)


    if np.isnan(anthropometry_vector).any():
        print(f"Warning: Final vector contains NaNs for CIPIC subject {subject_id}. Skipping.")
        return None

    return anthropometry_vector






def load_ari_anthropometry(subject_id, ear_side, anthro_data):
    """
    Loads and maps anthropometry for an ARI subject, handling the unique ID mapping.
    """
    try:
        anthro_id_to_find = 3000 + int(subject_id)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse ARI subject ID '{subject_id}'. Skipping.")
        return None

    all_anthro_ids = anthro_data['id'].flatten()


    indices = np.where(all_anthro_ids == anthro_id_to_find)[0]
    if indices.size == 0:
        print(f"Warning: Anthropometry ID {anthro_id_to_find} not found for ARI subject {subject_id}. Skipping.")
        return None
    subject_idx = indices[0]


    D = anthro_data['D']
    X = anthro_data['X']
    theta = anthro_data['theta']

    if subject_idx >= X.shape[0] or np.isnan(X[subject_idx, :]).all():
        print(f"Warning: Anthropometry not available for ARI subject {subject_id}. Skipping.")
        return None

    #hutubs_x_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 16]
    hutubs_x_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    common_values = X[subject_idx, hutubs_x_indices]

    if ear_side == 'left':
        d_values = D[subject_idx, :8]
        theta_values = theta[subject_idx, :2]
    else:
        d_values = D[subject_idx, 8:16]
        theta_values = theta[subject_idx, 2:4]

    anthropometry_vector = np.concatenate([
        common_values,
        d_values,
        theta_values
    ]).astype(np.float32)

    if np.isnan(anthropometry_vector).any():
        print(f"Warning: Final vector contains NaNs for ARI subject {subject_id}. Skipping.")
        return None

    return anthropometry_vector

def preprocess_datasets():
    """
    Main function to preprocess HRIR and anthropometric data.
    """
    dataset_config = {
        "chedar": {"sofa_name": "CHEDAR", "anthro_loader": load_chedar_anthropometry},
        # "scut": {"sofa_name": "SCUT", "anthro_loader": load_scut_anthropometry},
        # "ari": {"sofa_name": "ARI", "anthro_loader": load_ari_anthropometry},
        # "cipic": {"sofa_name": "CIPIC", "anthro_loader": load_cipic_anthropometry},
        # "hutubs": {"sofa_name": "HUTUBS", "anthro_loader": load_hutubs_anthropometry},
    }


    anthro_databases = {}


    # hutubs_csv_path = os.path.join(DATASET_PATHS["hutubs"], "AntrhopometricMeasures.csv")
    # anthro_df = pd.read_csv(hutubs_csv_path, index_col=0)
    # anthro_df.index = anthro_df.index.map(str)
    # anthro_databases["hutubs"] = anthro_df
    # print("✅ Successfully loaded HUTUBS anthropometry CSV.")

    # scut_csv_path = os.path.join(DATASET_PATHS["scut"], "AnthropometricParameters.csv")
    # scut_df = pd.read_csv(scut_csv_path, index_col=0) # Assuming the first column is the subject ID
    # scut_df.index = scut_df.index.map(str)
    # anthro_databases["scut"] = scut_df
    # print("Successfully loaded SCUT anthropometry CSV.")

    chedar_csv_path = os.path.join(DATASET_PATHS["chedar"], "measurements.csv")
    chedar_df = pd.read_csv(chedar_csv_path)
    chedar_df.index = (chedar_df.index + 1).astype(str)
    chedar_df.index.name = "Subject ID"
    anthro_databases["chedar"] = chedar_df
    print("Successfully loaded chedar anthropometry CSV.")


    # cipic_mat_path = os.path.join(DATASET_PATHS["cipic"], "anthro.mat")
    # anthro_databases["cipic"] = scipy.io.loadmat(cipic_mat_path)
    # print("Successfully loaded CIPIC anthropometry .mat file.")

    # ari_mat_path = os.path.join(DATASET_PATHS["ari"], "anthro.mat")
    # anthro_databases["ari"] = scipy.io.loadmat(ari_mat_path)
    # print("Successfully loaded ARI anthropometry .mat file.")




    for name, config in dataset_config.items():
        print(f"\n--- Processing dataset: {name.upper()} ---")

        if name not in anthro_databases:
            print(f"Skipping {name} due to missing anthropometry data.")
            continue

        dataset_obj = getattr(SOFAdatasets, config["sofa_name"])()


        saved_count = 0
        print(f"Found {len(dataset_obj)} HRIR files to process...")
        for idx in tqdm(range(len(dataset_obj))):
            try:

                location, hrir = dataset_obj[idx]

                subject_idx = idx // 2
                subject_id = str(dataset_obj._get_subject_ID(subject_idx))
                ear_side = 'left' if idx % 2 == 0 else 'right'




                loader_func = config["anthro_loader"]

                if name in ["scut", "chedar"]:
                    subject_id_ant = str(int(subject_id))
                    raw_anthro_data = loader_func(subject_id_ant, ear_side, anthro_databases[name])
                else:
                    raw_anthro_data = loader_func(subject_id, ear_side, anthro_databases[name])


                if raw_anthro_data is None:
                    continue


                filename = f"{name}_{idx:04d}.pkl"
                data_to_save = (location, hrir, raw_anthro_data)
                with open(os.path.join(OUT_DIR, filename), 'wb') as handle:
                    pkl.dump(data_to_save, handle, protocol=pkl.HIGHEST_PROTOCOL)
                saved_count += 1

            except Exception as e:
                print(f"❌ ERROR processing index {idx} for dataset {name}: {e}")
                continue

        print(f"Finished processing for {name.upper()}. Saved {saved_count} files.")


if __name__ == "__main__":
    preprocess_datasets()
    print("\nPreprocessing finished.")
