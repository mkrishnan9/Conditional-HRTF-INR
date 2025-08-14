import argparse
import os
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
from sklearn.model_selection import GroupKFold
import re
import time
torch.set_float32_matmul_precision('high')


import math
from HRTFdatasets import MergedHRTFDataset
from model import (
        CondHRTFNetwork,
        CondHRTFNetwork_FiLM,
        CondHRTFNetwork_Attention,
        CondHRTFNetwork_conv,
        CondHRTFNetwork_with_Dropout,
    )

import logging


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with a linear warmup followed by a cosine decay.
    """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)




# def merged_collate_fn(batch):
#     """
#     Collates samples from the MergedHRTFDataset into batches with padding and masks.
#     """
#     valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]
#     if not valid_batch: return None

#     locations, hrtfs, anthros, names = zip(*valid_batch)
#     B = len(locations)
#     anthro_dim = anthros[0].shape[0]
#     num_freq_bins = hrtfs[0].shape[1]
#     lengths = [loc.shape[0] for loc in locations]
#     max_num_loc = max(lengths) if lengths else 0
#     if max_num_loc == 0: return None

#     padded_locs = torch.zeros((B, max_num_loc, 2), dtype=torch.float32)
#     padded_hrtfs = torch.zeros((B, max_num_loc, num_freq_bins), dtype=torch.float32)
#     masks = torch.zeros((B, max_num_loc, 1), dtype=torch.float32)
#     collated_anthros = torch.stack(anthros, dim=0)

#     for i in range(B):
#         n_loc = lengths[i]
#         if n_loc > 0:
#             padded_locs[i, :n_loc, :] = locations[i]
#             padded_hrtfs[i, :n_loc, :] = hrtfs[i]
#             masks[i, :n_loc, :] = 1.0

#     return padded_locs, padded_hrtfs, masks, collated_anthros, list(names)

def merged_collate_fn(batch):
    """
    collate function for MergedHRTFDataset using pad_sequence.
    Input: list of tuples (location_tensor, hrtf_tensor, anthro_tensor, name)
    Output: tuple (padded_locs, padded_hrtfs, masks, collated_anthros, names)
    """
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]
    if not valid_batch:
        return None


    locations, hrtfs, anthros, names = zip(*valid_batch)

    # Pad sequences
    # pad_sequence expects a list of tensors with shape (L, *)
    # locations: list of (N_loc_i, 2)
    # hrtfs: list of (N_loc_i, num_freq_bins)

    # Need to handle empty locations within subjects - MergedHRTFDataset should ideally not yield these
    # If a subject has 0 locations, it will be filtered out by `valid_batch`.
    # Assuming valid_batch always has items with N_loc > 0 for their tensors.


    # Pad locations and HRTFs
    padded_locs = torch.nn.utils.rnn.pad_sequence(locations, batch_first=True, padding_value=0.0)
    padded_hrtfs = torch.nn.utils.rnn.pad_sequence(hrtfs, batch_first=True, padding_value=0.0)

    # Create masks: 1 where data is present, 0 where padded
    max_num_loc = padded_locs.shape[1]
    masks = torch.zeros((len(locations), max_num_loc, 1), dtype=torch.float32)
    for i, loc_tensor in enumerate(locations):
        masks[i, :loc_tensor.shape[0], :] = 1.0

    # Stack anthropometries - they should all be the same shape (anthro_dim,)
    collated_anthros = torch.stack(anthros, dim=0)

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
    """


    print("Calculating anthropometry statistics on training set...")
    all_anthro_tensors = []

    # Iterate through the indices specified by the Subset
    for i in range(len(dataset_subset)):
        try:
            _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]
            all_anthro_tensors.append(anthro_tensor)
        except Exception as e:
            # Catch potential errors during dataset access for a specific index
            print(f"Warning: Could not retrieve or unpack item for original dataset index {dataset_subset.indices[i]}: {e}. Skipping this item.")
            continue # Skip this problematic index


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
    return (anthro_tensor - mean.to(anthro_tensor.device)) / std.to(anthro_tensor.device)


# --- Main Training Function ---

def train(args):
    # --- Configuration ---
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    SAVE_DIR = os.path.join(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)


    print(f"Using device: {DEVICE}")
    print(f"Checkpoints and logs will be saved to: {SAVE_DIR}")




    DATA_PATH = args.data_path

    ANTHROPOMETRY_DIM = 19 # 23 ### HARDCODED

    # *******************************

    # Model & Training Parameters
    NUM_FREQ_BINS = 92             # Number of frequency bins (1 to 92 inclusive)
    TARGET_FREQ_DIM = NUM_FREQ_BINS # Model output dimension
    DECODER_INIT = 'siren'         # Initialization type for decoder (ensure model.py handles this)
    DECODER_NONLINEARITY = 'sine'  # Activation function for decoder


    SEED = 16



    os.makedirs(SAVE_DIR, exist_ok=True)

    def seed_worker(worker_id):
        """Seeds a DataLoader worker to ensure reproducibility."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(SEED)

    # --- Dataset and Loaders ---
    print("Loading dataset...")


    all_available_datasets = ["hutubs", "cipic", "ari", "scut", "chedar"]
    if args.include_datasets.lower() == "all":
        datasets_to_load = all_available_datasets
    else:
        datasets_to_load = [name.strip() for name in args.include_datasets.split(',')]

    print(f"Including datasets: {datasets_to_load}")

    merged_dataset = MergedHRTFDataset(
        dataset_names=datasets_to_load,
        preprocessed_dir=DATA_PATH,
        augment=False,
        aug_prob=args.aug_prob,
        scale=args.scale # Example: use log scale for training
    )

    # --- Subject-Aware Splitting Logic ---
    subjects = {}
    print("Mapping subjects to registry indices for splitting...")
    registry_to_subject_id = {}
    unique_subject_keys = []
    for i, file_path in enumerate(merged_dataset.file_registry):
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match: continue
        dataset_name, file_num = match.group(1), int(match.group(2))
        subject_id_in_dataset = file_num // 2
        subject_key = (dataset_name, subject_id_in_dataset)

        if subject_key not in unique_subject_keys:
            unique_subject_keys.append(subject_key)

        # Map the dataset index 'i' to the index of its subject_key
        subject_idx = unique_subject_keys.index(subject_key)
        registry_to_subject_id[i] = subject_idx

    indices = np.arange(len(merged_dataset))
    groups = np.array([registry_to_subject_id[i] for i in indices])

    N_SPLITS = 5 # A common choice
    kfold = GroupKFold(n_splits=N_SPLITS)

    fold_results = []
    fold_lsd = []


    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation...")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices, groups=groups)):
        print("-" * 50)
        print(f"Fold {fold+1}/{N_SPLITS}")
        print("-" * 50)

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
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=merged_collate_fn,num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        merged_dataset.set_augmentation(False)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=merged_collate_fn,num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        NUM_EPOCHS = args.epochs

        num_training_steps = NUM_EPOCHS * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)




        # --- Model, Optimizer, Loss ---
        print("Initializing model...")



        model = CondHRTFNetwork(
                d_anthro_in=ANTHROPOMETRY_DIM,
                d_inr_out=TARGET_FREQ_DIM,
                d_latent=args.latent_dim,
                d_inr_hidden=args.hidden_dim,
                n_inr_layers=args.n_layers
            ).to(DEVICE)


        model = torch.compile(model)

        if args.load_checkpoint:
            print(f"Loading checkpoint for fine-tuning: {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights loaded successfully.")

            print(f"Resetting optimizer for fine-tuning with new LR: {args.lr}")
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            # If not loading a checkpoint, define the optimizer normally
            optimizer = optim.Adam(model.parameters(), lr=args.lr)




        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=args.eta_min)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print("Starting training...")
        best_fold_val_loss = float('inf')
        patience = 75
        pat_count = 0

        # --- Training Loop ---
        for epoch in range(args.epochs):
            # epoch_start_time = time.time()
            model.train()
            train_loss_accum = 0.0
            processed_batches = 0
            train_lsd_accum = 0.0
            num_valid_points_epoch = 0


            for i, batch in enumerate(train_loader):


                locations, target_hrtfs, masks, anthropometry, _ = batch

                locations = locations.to(DEVICE, non_blocking=True)
                target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)


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

                # diff = predicted_hrtfs[..., 1:] - predicted_hrtfs[..., :-1]
                # smoothness_loss= torch.mean(torch.pow(diff, 2))
                # lamb_val = 5e-1

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

                loss = torch.sum(masked_error_squared) / num_valid_points_batch # + lamb_val*torch.sum(smoothness_loss)


                # --- Calculate LSD metric using masks ---
                with torch.no_grad():
                    if args.scale == 'log': # LSD for log scale data
                        lsd_elements_sq = torch.square(50*predicted_hrtfs - 50*target_hrtfs) # error_squared is already this
                    else: # LSD for linear scale data (original formula)
                        epsilon = 1e-9
                        pred_abs = torch.abs(predicted_hrtfs) + epsilon
                        gt_abs = torch.abs(target_hrtfs) + epsilon
                        lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                    masked_lsd_elements_sq = lsd_elements_sq * masks
                    batch_lsd_sum_sq = torch.sum(masked_lsd_elements_sq)


                loss.backward()
                optimizer.step()
                scheduler.step()


                train_loss_accum += loss.item() * num_valid_points_batch # Accumulate total loss sum
                if num_valid_points_batch > 0:
                    train_lsd_accum += batch_lsd_sum_sq.item() # Accumulate sum of squared LSD
                num_valid_points_epoch += num_valid_points_batch.item()
                processed_batches += 1

            # Calculate average epoch loss and LSD
            if num_valid_points_epoch > 0:
                avg_train_loss = train_loss_accum / num_valid_points_epoch
                # Avg LSD is sqrt( sum(lsd^2) / NK )
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



                    # Calculate masked loss
                    error_squared = torch.square(predicted_hrtfs - target_hrtfs)
                    masked_error_squared = error_squared * masks
                    num_valid_points_batch = torch.sum(masks)
                    if num_valid_points_batch > 0:
                            batch_loss_sum = torch.sum(masked_error_squared)
                            val_loss_accum += batch_loss_sum.item()
                            if args.scale == 'log':
                                lsd_elements_sq = torch.square(50*(predicted_hrtfs+1) - 50*(target_hrtfs+1))
                            else:
                                epsilon = 1e-9
                                pred_abs = torch.abs(predicted_hrtfs + epsilon +1)
                                gt_abs = torch.abs(target_hrtfs + epsilon + 1)
                                lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                            masked_lsd_elements_sq = lsd_elements_sq * masks
                            val_lsd_accum += torch.sum(masked_lsd_elements_sq).item()

                            val_valid_points_epoch += num_valid_points_batch.item()
            # Calculate average epoch validation loss and LSD

            avg_val_loss = val_loss_accum / val_valid_points_epoch
            avg_val_lsd = np.sqrt(val_lsd_accum / (val_valid_points_epoch * NUM_FREQ_BINS))
            pat_count = pat_count+1



            #scheduler.step()
            # epoch_end_time = time.time()

            # # # Calculate and print the elapsed time
            # elapsed_time = epoch_end_time - epoch_start_time
            # print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds.")



            print(f'Fold {fold+1} | Epoch [{epoch+1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train LSD: {avg_train_lsd:.4f}, Val LSD: {avg_val_lsd:.4f}')



            # --- Save Best Model ---
            if val_valid_points_epoch > 0 and avg_val_loss < best_fold_val_loss: # Check validation ran and improved
                pat_count=0

                best_fold_val_loss = avg_val_loss
                best_fold_val_lsd = avg_val_lsd
                print(f'---> Model improved')
                # model_save_path = os.path.join(SAVE_DIR, f'hrtf_anthro_model_best_allfreqs.pth')

                # torch.save({
                #         'epoch': epoch + 1,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'loss': best_fold_val_loss,
                #         'anthro_mean': anthro_mean.cpu(),
                #         'anthro_std': anthro_std.cpu(),
                #         }, model_save_path)
                # print(f'saved to {model_save_path}')

            if pat_count > patience:
                break

        print("\nTraining finished.")

        print(f"Best Validation Loss achieved: {best_fold_val_loss:.6f}")
        print(f"Best Validation LSD achieved: {best_fold_val_lsd:.6f}")
        #print(f"Check '{SAVE_DIR}' for the best model checkpoint.")
        fold_results.append(best_fold_val_loss)
        fold_lsd.append(best_fold_val_lsd)
        print(f"\n---> Best Validation Loss for Fold {fold+1}: {best_fold_val_loss:.6f}\n")

    # --- Final Results ---
    mean_cv_loss = np.mean(fold_results)
    std_cv_loss = np.std(fold_results)
    mean_cv_lsd = np.mean(fold_lsd)
    std_cv_lsd = np.std(fold_lsd)

    print("=" * 50)
    print("Cross-Validation Finished")
    print(f"Validation Losses per Fold: {[f'{loss:.6f}' for loss in fold_results]}")
    print(f"Average Validation Loss: {mean_cv_loss:.6f}")
    print(f"Standard Deviation of Validation Loss: {std_cv_loss:.6f}")
    print(f"Validation LSD per Fold: {[f'{loss:.6f}' for loss in fold_lsd]}")
    print(f"Average Validation LSD: {mean_cv_lsd:.6f}")
    print(f"Standard Deviation of Validation LSD: {std_cv_lsd:.6f}")
    print("=" * 50)





if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="HRTF Anthropometry Model Training")

    # Path Arguments
    parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common", help='Path to preprocessed data')
    parser.add_argument('--save_dir', type=str, default="/export/mkrishn9/hrtf_field/experiments_test", help='Directory to save experiment checkpoints')
    parser.add_argument('--gpu', type=int, default=4, help='GPU device ID to use')
    parser.add_argument('--include_datasets', type=str, default="all",
                        help="Comma-separated list of datasets to use. 'all' uses everything.")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to a model checkpoint to load for fine-tuning.")


    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Decoder hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--scale', type=str, default="log", help='Scale')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Learning rate annealing')
    parser.add_argument('--val_split', type=float, default=0.15, help='Training testing split')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability of augmentation')


    # --- 2. Parse Arguments and Run Training ---
    args = parser.parse_args()






    print("=============================================")
    print("       HRTF Anthropometry Model Training     ")
    print("=============================================")
    print(f"Make sure 'hrtf_dataset.py' and 'model.py' are in the Python path.")
    print("---------------------------------------------")

    train(args)

