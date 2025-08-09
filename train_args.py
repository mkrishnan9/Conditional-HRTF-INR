import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.dataloader import default_collate
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import numpy as np
import pickle as pkl
import random
import re
import os
import collections
from HRTFdatasets import MergedHRTFDataset
from model import (
        HRTFNetwork,
        AnthropometryEncoder,
        CrossAttentionEncoder,
        FCBlock,
        MetaModule,
        BatchLinear,
        Sine,
        MetaSequential,
    )

import logging

import argparse





def custom_collate_fn(batch):
    """
    Collates samples into batches with padding and masks.
    Input: list of tuples (location_tensor, hrtf_tensor, anthro_tensor)
    Output: tuple (padded_locs, padded_hrtfs, masks, collated_anthros)
    """
    # Filter out potential None items from dataset failures
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]


    locations, hrtfs, anthros = zip(*valid_batch)

    B = len(locations)
    anthro_dim = anthros[0].shape[0]
    num_freq_bins = hrtfs[0].shape[1]


    lengths = [loc.shape[0] for loc in locations]
    max_num_loc = max(lengths) if lengths else 0

    if max_num_loc == 0:
        print("Warning: Batch contains samples with 0 locations after filtering.")
        return None


    padded_locs = torch.zeros((B, max_num_loc, 2), dtype=torch.float32)
    padded_hrtfs = torch.zeros((B, max_num_loc, num_freq_bins), dtype=torch.float32)
    masks = torch.zeros((B, max_num_loc, 1), dtype=torch.float32) # Mask shape (B, N_loc, 1) for broadcasting

    collated_anthros = torch.stack(anthros, dim=0) # Shape: (B, anthro_dim)

    for i in range(B):
        n_loc = lengths[i]
        if n_loc > 0:
            padded_locs[i, :n_loc, :] = locations[i]
            padded_hrtfs[i, :n_loc, :] = hrtfs[i]
            masks[i, :n_loc, :] = 1.0 # Set mask to 1 for valid locations

    return padded_locs, padded_hrtfs, masks, collated_anthros

def merged_collate_fn(batch):
    """
    Collates samples from the MergedHRTFDataset into batches with padding and masks.
    """
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]
    if not valid_batch: return None

    locations, hrtfs, anthros, names = zip(*valid_batch)
    B = len(locations)
    anthro_dim = anthros[0].shape[0]
    num_freq_bins = hrtfs[0].shape[1]
    lengths = [loc.shape[0] for loc in locations]
    max_num_loc = max(lengths) if lengths else 0
    if max_num_loc == 0: return None

    padded_locs = torch.zeros((B, max_num_loc, 2), dtype=torch.float32)
    padded_hrtfs = torch.zeros((B, max_num_loc, num_freq_bins), dtype=torch.float32)
    masks = torch.zeros((B, max_num_loc, 1), dtype=torch.float32)
    collated_anthros = torch.stack(anthros, dim=0)

    for i in range(B):
        n_loc = lengths[i]
        if n_loc > 0:
            padded_locs[i, :n_loc, :] = locations[i]
            padded_hrtfs[i, :n_loc, :] = hrtfs[i]
            masks[i, :n_loc, :] = 1.0

    return padded_locs, padded_hrtfs, masks, collated_anthros, list(names)

def metrics(gt, pred, scale="linear"):
    if scale == "linear":
        lsd_elements = torch.square(20 * torch.log10(torch.abs(gt) / torch.abs(pred)))
    elif scale == "log":
        lsd_elements = torch.square(gt - pred)
    else:
        raise ValueError("Either log or linear scale")
    square_sum = (lsd_elements).sum()
    lsd = square_sum
    return torch.sqrt(lsd), square_sum.item()

def calculate_anthro_stats(dataset_subset):
    """
    Calculates mean and std dev for anthropometry across the dataset subset.
    Assumes dataset items are valid (location_tensor, hrtf_tensor, anthro_tensor).
    """
    if not isinstance(dataset_subset, Subset):
        raise TypeError("calculate_anthro_stats expects a PyTorch Subset object.")

    print("Calculating anthropometry statistics on training set...")
    all_anthro_tensors = []

    # Iterate through the indices specified by the Subset
    for i in range(len(dataset_subset)):
        try:
            # Get the item from the original dataset using the subset's index mapping
            # Assumes __getitem__ returns (loc_tensor, hrtf_tensor, anthro_tensor)
            # or raises an error if loading fails.
            _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]


            all_anthro_tensors.append(anthro_tensor)
        except Exception as e:
            # Catch potential errors during dataset access for a specific index
            print(f"Warning: Could not retrieve or unpack item for original dataset index {dataset_subset.indices[i]}: {e}. Skipping this item.")
            continue # Skip this problematic index

    if not all_anthro_tensors:
        raise ValueError("No valid anthropometric data was collected from the training set to calculate stats.")

    # Stack collected tensors
    try:
        all_anthro_stacked = torch.stack(all_anthro_tensors, dim=0)
    except Exception as e:
        # This error is critical if tensor shapes mismatch
        print(f"Error: Failed to stack anthropometry tensors: {e}")
        print("This usually indicates inconsistent anthropometry vector lengths in your dataset.")
        # Print shapes of first few tensors for debugging
        for t_idx, t in enumerate(all_anthro_tensors[:5]):
             print(f"  Tensor {t_idx} shape: {t.shape}")
        raise ValueError("Could not stack anthropometry tensors due to shape mismatch or other error.")

    # Calculate mean and std using torch functions
    mean = torch.mean(all_anthro_stacked, dim=0, dtype=torch.float32)
    std = torch.std(all_anthro_stacked, dim=0)

    # Clamp std dev to prevent division by zero
    std = torch.clamp(std, min=1e-8)

    return mean, std

def normalize_anthro(anthro_tensor, mean, std):
    """Applies Z-score normalization."""
    # Ensure mean and std are broadcastable to anthro_tensor shape if needed
    # Assuming mean/std are 1D tensors of size (anthro_dim,)
    # and anthro_tensor is (batch_size * n_loc, anthro_dim)
    return (anthro_tensor - mean.to(anthro_tensor.device)) / std.to(anthro_tensor.device)

def custom_collate_fn_init(batch):
    """Collates samples with variable number of locations into batch tensors."""
    locations = []
    hrtfs = []
    anthros = []
    # Filter out potential None items from dataset failures before zipping
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]

    if not valid_batch:
        print("Warning: custom_collate_fn received an empty or all-None batch. Skipping.")
        return None

    try:
        locations, hrtfs, anthros = zip(*valid_batch)
    except Exception as e:
        print(f"Error during unzipping batch in collate_fn: {e}. Batch content: {valid_batch}")
        return None # Skip batch

    all_locs_flat = []
    all_hrtfs_flat = []
    all_anthros_repeated = []

    for i in range(len(locations)):
        loc_i, hrtf_i, anthro_i = locations[i], hrtfs[i], anthros[i]

        # Basic validation
        if not isinstance(loc_i, torch.Tensor) or not isinstance(hrtf_i, torch.Tensor) or not isinstance(anthro_i, torch.Tensor):
            print(f"Warning: Invalid data type in batch item {i}. Skipping subject.")
            continue

        n_loc = loc_i.shape[0]
        if n_loc == 0:
            # print(f"Warning: Subject {i} in batch has 0 locations. Skipping.") # Can be verbose
            continue

        # Check if shapes are reasonable (basic check)
        if loc_i.ndim != 2 or hrtf_i.ndim != 2 or anthro_i.ndim != 1:
             print(f"Warning: Unexpected tensor dimensions for subject {i} in batch. Skipping. Shapes: loc {loc_i.shape}, hrtf {hrtf_i.shape}, anthro {anthro_i.shape}")
             continue
        if loc_i.shape[0] != hrtf_i.shape[0]:
             print(f"Warning: Mismatch between locations ({loc_i.shape[0]}) and HRTFs ({hrtf_i.shape[0]}) for subject {i}. Skipping.")
             continue


        # Repeat anthropometry for each location
        try:
            anthro_i_repeated = anthro_i.unsqueeze(0).expand(n_loc, -1) # Repeat anthro
        except Exception as e:
             print(f"Error expanding anthropometry for subject {i}: {e}. Anthro shape: {anthro_i.shape}, n_loc: {n_loc}. Skipping.")
             continue


        all_locs_flat.append(loc_i)
        all_hrtfs_flat.append(hrtf_i)
        all_anthros_repeated.append(anthro_i_repeated)

    if not all_locs_flat:
        # print("Warning: Batch resulted in no valid data after processing subjects. Skipping batch.") # Can be verbose
        return None # Skip if batch becomes empty

    # Concatenate all points in the batch
    try:
        batch_locs = torch.cat(all_locs_flat, dim=0)
        batch_hrtfs = torch.cat(all_hrtfs_flat, dim=0)
        batch_anthros = torch.cat(all_anthros_repeated, dim=0)
    except Exception as e:
        print(f"Error concatenating batch tensors: {e}. Skipping batch.")
        return None


    return batch_locs, batch_hrtfs, batch_anthros

# --- Main Training Function ---

def train(args):
    # --- Configuration ---
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    # Create a unique save directory for this specific run
    run_name = f"lr{args.lr}_bs{args.batch_size}_ld{args.latent_dim}_hd{args.hidden_dim}_nl{args.n_layers}"
    SAVE_DIR = os.path.join(args.save_dir, run_name)
    os.makedirs(SAVE_DIR, exist_ok=True)



    # Setup logging to save to a file in the run's main directory

    print(f"Using device: {DEVICE}")
    print(f"Checkpoints and logs will be saved to: {SAVE_DIR}")




    DATA_PATH = args.data_path
    # <<< UPDATE dimension of your anthropometry data >>>
    ANTHROPOMETRY_DIM = 19 # 23 ### HARDCODED

    # *******************************

    # Model & Training Parameters
    NUM_FREQ_BINS = 92             # Number of frequency bins (1 to 92 inclusive)
    TARGET_FREQ_DIM = NUM_FREQ_BINS # Model output dimension
    DECODER_INIT = 'siren'         # Initialization type for decoder (ensure model.py handles this)
    DECODER_NONLINEARITY = 'sine'  # Activation function for decoder


    SEED = 42


    os.makedirs(SAVE_DIR, exist_ok=True)


    torch.manual_seed(SEED)
    np.random.seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    # --- Dataset and Loaders ---
    print("Loading dataset...")


    all_datasets_to_load = ["hutubs", "cipic", "ari", "scut","chedar"] # Add other dataset names here

    merged_dataset = MergedHRTFDataset(
        dataset_names=all_datasets_to_load,
        preprocessed_dir=DATA_PATH,
        augment=False,
        aug_prob=0.5,
        scale="log" # Example: use log scale for training
    )

    # --- Subject-Aware Splitting Logic ---
    # subjects = {}
    # print("Mapping subjects to registry indices for splitting...")
    # for i, file_path in enumerate(merged_dataset.file_registry):
    #     base_name = os.path.basename(file_path)
    #     match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
    #     if not match: continue
    #     dataset_name, file_num = match.group(1), int(match.group(2))
    #     subject_id_in_dataset = file_num // 2
    #     subject_key = (dataset_name, subject_id_in_dataset)
    #     if subject_key not in subjects: subjects[subject_key] = []
    #     subjects[subject_key].append(i)

    # all_subjects = list(subjects.keys())
    # random.shuffle(all_subjects)
    # num_val_subjects = int(len(all_subjects) * args.val_split)
    # val_subject_keys = all_subjects[:num_val_subjects]
    # train_subject_keys = all_subjects[num_val_subjects:]
    # train_indices = [idx for key in train_subject_keys for idx in subjects[key]]
    # val_indices = [idx for key in val_subject_keys for idx in subjects[key]]
    # print(f"Total unique subjects: {len(all_subjects)}, Train: {len(train_subject_keys)}, Val: {len(val_subject_keys)}")
    # print(f"Total ears: {len(merged_dataset)}, Train: {len(train_indices)}, Val: {len(val_indices)}")

    # --- Subject-Aware Splitting Logic ---
    subjects = {}
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
    print(f"Total ears: {len(merged_dataset)}, Train: {len(train_indices)}, Val: {len(val_indices)}")

    # --- Create Subsets and DataLoaders ---
    train_subset = Subset(merged_dataset, train_indices)
    val_subset = Subset(merged_dataset, val_indices)



    try:
        anthro_mean, anthro_std = calculate_anthro_stats(train_subset)
        anthro_mean = anthro_mean.to(DEVICE)
        anthro_std = anthro_std.to(DEVICE)
        print(f"Anthro Mean (sample): {anthro_mean[:5].cpu().numpy()}")
        print(f"Anthro Std (sample): {anthro_std[:5].cpu().numpy()}")
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error calculating stats: {e}")
        return

    merged_dataset.set_augmentation(False)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=merged_collate_fn,num_workers=8, pin_memory=True)

    merged_dataset.set_augmentation(False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=merged_collate_fn,num_workers=8, pin_memory=True)
    NUM_EPOCHS = args.epochs



    # --- Model, Optimizer, Loss ---
    print("Initializing model...")
    model = HRTFNetwork(
            anthropometry_dim=ANTHROPOMETRY_DIM,
            target_freq_dim=TARGET_FREQ_DIM,
            latent_dim=args.latent_dim,
            decoder_hidden_dim=args.hidden_dim,
            decoder_n_hidden_layers=args.n_layers,
            init_type=DECODER_INIT,
            nonlinearity=DECODER_NONLINEARITY
        ).to(DEVICE)



    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.MSELoss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=args.eta_min)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Starting training...")
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0
        processed_batches = 0
        train_lsd_accum = 0.0
        num_valid_points_epoch = 0


        for i, batch in enumerate(train_loader):


            locations, target_hrtfs, masks, anthropometry, _ = batch

            locations = locations.to(DEVICE, non_blocking=True)
            target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)
            breakpoint()
            masks = masks.to(DEVICE, non_blocking=True)
            anthropometry = anthropometry.to(DEVICE, non_blocking=True)

            # Normalize anthropometry
            try:
                anthropometry_norm = normalize_anthro(anthropometry, anthro_mean, anthro_std)
            except Exception as e:
                 print(f"Error normalizing anthropometry in batch {i+1}: {e}. Shapes: data {anthropometry.shape}, mean {anthro_mean.shape}, std {anthro_std.shape}")
                 continue

            B, N_loc, _ = locations.shape
            anthro_dim = anthropometry_norm.shape[-1]
            num_freq = target_hrtfs.shape[-1]
            anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, anthro_dim)
            locations_flat = locations.reshape(-1, 2)
            anthro_flat = anthro_expanded.reshape(-1, anthro_dim)


            optimizer.zero_grad()

            predicted_hrtfs_flat = model(locations_flat, anthro_flat) # Output: (B*N_loc, num_freq)
            # Reshape output back: (B*N_loc, num_freq) -> (B, N_loc, num_freq)
            predicted_hrtfs = predicted_hrtfs_flat.reshape(B, N_loc, num_freq)


            # Check shapes before loss calculation
            if predicted_hrtfs.shape != target_hrtfs.shape:
                 print(f"Shape mismatch before loss! Predicted: {predicted_hrtfs.shape}, Target: {target_hrtfs.shape}. Skipping batch {i+1}.")
                 continue

            # Calculate loss
            error_squared = torch.square(predicted_hrtfs - target_hrtfs)
            masked_error_squared = error_squared * masks
            # Sum loss over all valid points in the batch
            # Normalize by the number of valid points (sum of mask)
            num_valid_points_batch = torch.sum(masks)

            loss = torch.sum(masked_error_squared) / num_valid_points_batch


            # --- Calculate LSD metric using masks ---
            with torch.no_grad():
                if args.scale == 'log': # LSD for log scale data
                    lsd_elements_sq = torch.square(predicted_hrtfs - target_hrtfs) # error_squared is already this
                else: # LSD for linear scale data (original formula)
                    epsilon = 1e-9
                    pred_abs = torch.abs(predicted_hrtfs) + epsilon
                    gt_abs = torch.abs(target_hrtfs) + epsilon
                    lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                masked_lsd_elements_sq = lsd_elements_sq * masks
                batch_lsd_sum_sq = torch.sum(masked_lsd_elements_sq)

            # Backward pass and optimize (only if loss requires grad)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad() # Still zero grad if batch was skipped

            train_loss_accum += loss.item() * num_valid_points_batch # Accumulate total loss sum
            if num_valid_points_batch > 0:
                 train_lsd_accum += batch_lsd_sum_sq.item() # Accumulate sum of squared LSD
            num_valid_points_epoch += num_valid_points_batch.item()
            processed_batches += 1

        # Calculate average epoch loss and LSD
        if num_valid_points_epoch > 0:
            avg_train_loss = train_loss_accum / num_valid_points_epoch
            # Avg LSD is sqrt( sum(lsd^2) / N )
            avg_train_lsd = np.sqrt(train_lsd_accum / (num_valid_points_epoch*NUM_FREQ_BINS))
        else:
            print(f"Warning: Epoch {epoch+1} completed without any valid training points.")
            avg_train_loss = 0.0
            avg_train_lsd = 0.0




        # --- Validation Loop (needs similar masking logic) ---
        model.eval()
        val_loss_accum = 0.0
        val_lsd_accum = 0.0
        val_valid_points_epoch = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                locations, target_hrtfs, masks, anthropometry, _ = batch
                locations = locations.to(DEVICE, non_blocking=True)
                target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                anthropometry = anthropometry.to(DEVICE, non_blocking=True)




                try:
                    anthropometry_norm = normalize_anthro(anthropometry, anthro_mean.to(DEVICE), anthro_std.to(DEVICE))

                    # Expand and flatten inputs
                    B, N_loc, _ = locations.shape
                    anthro_dim = anthropometry_norm.shape[-1]
                    num_freq = target_hrtfs.shape[-1]
                    anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, anthro_dim)
                    locations_flat = locations.reshape(-1, 2)
                    anthro_flat = anthro_expanded.reshape(-1, anthro_dim)

                    # Predict
                    predicted_hrtfs_flat = model(locations_flat, anthro_flat)
                    predicted_hrtfs = predicted_hrtfs_flat.reshape(B, N_loc, num_freq)

                    if predicted_hrtfs.shape != target_hrtfs.shape:
                         print(f"Shape mismatch during validation! Pred: {predicted_hrtfs.shape}, Target: {target_hrtfs.shape}.")
                         continue

                    # Calculate masked loss
                    error_squared = torch.square(predicted_hrtfs - target_hrtfs)
                    masked_error_squared = error_squared * masks
                    num_valid_points_batch = torch.sum(masks)
                    if num_valid_points_batch > 0:
                         batch_loss_sum = torch.sum(masked_error_squared)
                         val_loss_accum += batch_loss_sum.item()
                         if args.scale == 'log':
                            lsd_elements_sq = torch.square(predicted_hrtfs - target_hrtfs)
                         else:
                            epsilon = 1e-9
                            pred_abs = torch.abs(predicted_hrtfs) + epsilon
                            gt_abs = torch.abs(target_hrtfs) + epsilon
                            lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                         masked_lsd_elements_sq = lsd_elements_sq * masks
                         val_lsd_accum += torch.sum(masked_lsd_elements_sq).item()

                         val_valid_points_epoch += num_valid_points_batch.item()

                except Exception as e:
                     print(f"Error during validation pass: {e}. Skipping batch.")
                     continue

        # Calculate average epoch validation loss and LSD

        avg_val_loss = val_loss_accum / val_valid_points_epoch
        avg_val_lsd = np.sqrt(val_lsd_accum / (val_valid_points_epoch * NUM_FREQ_BINS))



        scheduler.step()


        print(f'Epoch [{epoch+1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train LSD: {avg_train_lsd:.4f}, Val LSD: {avg_val_lsd:.4f}')



        # --- Save Best Model ---
        if val_valid_points_epoch > 0 and avg_val_loss < best_val_loss: # Check validation ran and improved

            best_val_loss = avg_val_loss
            best_val_lsd = avg_val_lsd
            model_save_path = os.path.join(SAVE_DIR, f'hrtf_anthro_model_best_allfreqs.pth')

            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'anthro_mean': anthro_mean.cpu(),
                    'anthro_std': anthro_std.cpu(),
                    }, model_save_path)
            print(f'---> Model improved, saved to {model_save_path}')




    print("\nTraining finished.")

    print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
    print(f"Best Validation LSD achieved: {best_val_lsd:.6f}")
    print(f"Check '{SAVE_DIR}' for the best model checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRTF Anthropometry Model Training")

    # Path Arguments
    parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common_ic", help='Path to preprocessed data')
    parser.add_argument('--save_dir', type=str, default="/export/mkrishn9/hrtf_field/experiments", help='Directory to save experiment checkpoints')
    parser.add_argument('--gpu', type=int, default=7, help='GPU device ID to use')

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Decoder hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--scale', type=str, default="log", help='Scale')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Learning rate annealing')
    parser.add_argument('--val_split', type=float, default=0.15, help='Training testing split')





    # --- 2. Parse Arguments and Run Training ---
    args = parser.parse_args()

    print("=============================================")
    print("       HRTF Anthropometry Model Training     ")
    print("=============================================")
    print(f"Make sure 'hrtf_dataset.py' and 'model.py' are in the Python path.")
    print("---------------------------------------------")

    train(args)

