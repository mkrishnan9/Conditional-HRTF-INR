import argparse
import os
import torch
import numpy as np
import re
import random
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.interpolate import griddata
from tqdm import tqdm

from HRTFdatasets import MergedHRTFDataset

# --- Helper Functions (Adapted from your original code) ---

def calculate_anthro_stats(dataset_subset):
    """
    Calculates mean and std dev for anthropometry across a dataset subset.
    """
    print("Calculating anthropometry statistics on the training set...")
    all_anthro_tensors = []
    # Using tqdm for a progress bar as this can take a moment
    for i in tqdm(range(len(dataset_subset)), desc="Analyzing train set"):
        try:
            # We only need the anthropometry tensor for this calculation
            _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]
            if anthro_tensor is not None:
                all_anthro_tensors.append(anthro_tensor)
        except Exception as e:
            print(f"Warning: Could not retrieve item at index {dataset_subset.indices[i]}: {e}. Skipping.")
            continue

    if not all_anthro_tensors:
        raise ValueError("No valid anthropometric data found in the training set.")

    all_anthro_stacked = torch.stack(all_anthro_tensors, dim=0)
    mean = torch.mean(all_anthro_stacked, dim=0, dtype=torch.float32)
    std = torch.std(all_anthro_stacked, dim=0)
    std = torch.clamp(std, min=1e-8) # Avoid division by zero
    return mean, std

def normalize_anthro(anthro_tensor, mean, std):
    """Applies Z-score normalization."""
    return (anthro_tensor - mean) / std

def calculate_lsd(pred_hrtf, true_hrtf):
    """
    Calculates the Log-Spectral Distance (LSD) in dB.
    Assumes inputs are on a log-magnitude scale.
    """
    # Ensure tensors are float for calculations
    pred_hrtf = pred_hrtf.float()
    true_hrtf = true_hrtf.float()

    # The error is simply the squared difference on the log scale
    error_sq = (pred_hrtf - true_hrtf)**2

    # The mean is taken over the frequency dimension
    lsd_per_point = torch.sqrt(torch.mean(error_sq, dim=-1))

    # Return the average LSD over all spatial points in this sample
    return torch.mean(lsd_per_point)

# --- Core Baseline Logic ---

def evaluate_baseline(train_subset, val_subset, anthro_mean, anthro_std, device):
    """
    Evaluates the baseline nearest-neighbor interpolator.

    For each validation sample:
    1. Finds the closest anthropometry in the entire training set.
    2. Uses that subject's HRTF data.
    3. Interpolates to find the HRTF at the validation sample's locations.
    4. Calculates the error (LSD).
    """
    anthro_mean = anthro_mean.to(device)
    anthro_std = anthro_std.to(device)

    # 1. Cache training data for efficient searching.
    # We store the normalized anthropometry and the full HRTF data for each training subject.
    print("Caching training data for nearest-neighbor search...")
    train_data_cache = []

    # We create a map to handle subjects with left/right ears to avoid redundant processing
    processed_train_subjects = {}
    for i in tqdm(range(len(train_subset)), desc="Caching train data"):
        idx = train_subset.indices[i]
        # Get subject key to avoid adding both left and right ears of the same person
        file_path = train_subset.dataset.file_registry[idx]
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match: continue
        dataset_name, file_num = match.group(1), int(match.group(2))
        subject_key = (dataset_name, file_num // 2)

        if subject_key in processed_train_subjects:
            continue

        locs, hrtfs, anthro, _ = train_subset.dataset[idx]

        if locs is not None and anthro is not None and locs.shape[0] > 0:
            train_data_cache.append({
                'norm_anthro': normalize_anthro(anthro, anthro_mean.cpu(), anthro_std.cpu()).to(device),
                'locations': locs.numpy(), # Scipy works with numpy arrays
                'hrtfs': hrtfs.numpy()
            })
            processed_train_subjects[subject_key] = True

    print(f"Cached {len(train_data_cache)} unique training subjects.")

    # 2. Iterate through validation set and evaluate
    total_lsd = 0.0
    num_val_samples = 0

    processed_val_subjects = {}
    print("Evaluating validation samples...")
    for i in tqdm(range(len(val_subset)), desc="Evaluating validation set"):
        idx = val_subset.indices[i]

        # Get subject key to process each validation subject only once
        file_path = val_subset.dataset.file_registry[idx]
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match: continue
        dataset_name, file_num = match.group(1), int(match.group(2))
        subject_key = (dataset_name, file_num // 2)

        if subject_key in processed_val_subjects:
            continue

        val_locs, val_hrtfs, val_anthro, _ = val_subset.dataset[idx]

        if val_locs is None or val_anthro is None or val_locs.shape[0] == 0:
            continue

        val_anthro_norm = normalize_anthro(val_anthro, anthro_mean.cpu(), anthro_std.cpu()).to(device)

        # 2a. Find the closest neighbor in the training cache
        min_dist = float('inf')
        closest_neighbor_data = None
        for train_subject in train_data_cache:
            dist = torch.linalg.norm(val_anthro_norm - train_subject['norm_anthro'])
            if dist < min_dist:
                min_dist = dist
                closest_neighbor_data = train_subject

        # 2b. Interpolate using the closest neighbor's data
        # `griddata` performs the interpolation.
        predicted_hrtfs_np = griddata(
            points=closest_neighbor_data['locations'], # Known points from train subject
            values=closest_neighbor_data['hrtfs'],     # Known values from train subject
            xi=val_locs.numpy(),                       # Points to predict for val subject
            method='linear',                           # 'linear' is bilinear for 2D
            fill_value=0.0                             # Fill with 0 for points outside convex hull
        )




        predicted_hrtfs = torch.from_numpy(predicted_hrtfs_np)

        # 2c. Calculate and accumulate the error
        lsd = calculate_lsd(predicted_hrtfs, val_hrtfs)
        total_lsd += lsd.item()
        num_val_samples += 1
        processed_val_subjects[subject_key] = True

    # 3. Calculate and return the average LSD
    average_lsd = total_lsd / num_val_samples if num_val_samples > 0 else float('inf')
    return average_lsd


def main(args):
    # --- Configuration ---
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Dataset and Splitting (identical to your original code) ---
    print("Loading dataset...")
    all_available_datasets = ["hutubs", "cipic", "ari", "scut", "chedar"]
    datasets_to_load = all_available_datasets if args.include_datasets.lower() == "all" else [name.strip() for name in args.include_datasets.split(',')]

    merged_dataset = MergedHRTFDataset(
        dataset_names=datasets_to_load,
        preprocessed_dir=args.data_path,
        scale="log" # Using log scale as in your training script
    )

    subjects = {}
    print("Mapping subjects for train/validation split...")
    print("Mapping subjects to registry indices for splitting...")
    for i, file_path in enumerate(merged_dataset.file_registry):
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match: continue
        dataset_name, file_num = match.group(1), int(match.group(2))
        subject_id_in_dataset = file_num // 2
        subject_key = (dataset_name, subject_id_in_dataset)
        if subject_key not in subjects: subjects[subject_key] = []
        subjects[subject_key].append(i)

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    chedar_subjects = []
    other_subjects = []
    for subject_key in subjects.keys():
        if subject_key[0] == 'chedar':
            chedar_subjects.append(subject_key)
        else:
            other_subjects.append(subject_key)

    print(f"Found {len(chedar_subjects)} 'chedar' subjects and {len(other_subjects)} subjects from other datasets.")

    # Perform the validation split ONLY on the non-chedar subjects
    random.shuffle(other_subjects)
    # Note: val_split is now applied to the count of 'other' subjects, not the total.
    num_val_subjects = int(len(other_subjects) * args.val_split)
    val_subject_keys = other_subjects[:num_val_subjects]

    # The training set includes the remaining 'other' subjects AND all 'chedar' subjects
    train_other_subject_keys = other_subjects[num_val_subjects:]
    train_subject_keys = train_other_subject_keys + chedar_subjects


    # The rest of the code remains the same
    train_indices = [idx for key in train_subject_keys for idx in subjects[key]]
    val_indices = [idx for key in val_subject_keys for idx in subjects[key]]

    total_subjects = len(chedar_subjects) + len(other_subjects)

    print(f"Total unique subjects: {total_subjects}, Train: {len(train_subject_keys)}, Val: {len(val_subject_keys)}")

    train_subset = Subset(merged_dataset, train_indices)
    val_subset = Subset(merged_dataset, val_indices)

    # --- Calculate Normalization Stats ---
    try:
        anthro_mean, anthro_std = calculate_anthro_stats(train_subset)
        print("Successfully calculated anthropometry statistics.")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- Run Baseline Evaluation ---
    avg_lsd_error = evaluate_baseline(train_subset, val_subset, anthro_mean, anthro_std, DEVICE)

    print("\n=============================================")
    print("      Baseline Evaluation Finished")
    print("=============================================")
    print(f"Method: Nearest-Neighbor Anthropometry + Bilinear Interpolation")
    print(f"Average Log-Spectral Distance (LSD) on Validation Set: {avg_lsd_error:.4f} dB")
    print("=============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRTF baseline prediction using nearest-neighbor interpolation.")

    parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common", help='Path to preprocessed data')
    parser.add_argument('--gpu', type=int, default=5, help='GPU device ID to use')
    parser.add_argument('--include_datasets', type=str, default="all", help="Comma-separated list of datasets to use. 'all' uses everything.")
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of subjects to use for validation')

    args = parser.parse_args()
    main(args)