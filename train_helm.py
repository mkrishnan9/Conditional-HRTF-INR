
import argparse
import os
import torch
import torch.nn as nn

#import deepxde as dde
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
        HRTFNetwork3D
    )

import logging



C_SOUND = 343.0  # Speed of sound in m/s

from torch.func import vmap



def spherical_to_cartesian(coords_deg, radius=1.0):
    """
    Converts spherical coordinates (azimuth, elevation) in degrees to 3D Cartesian coordinates.
    Assumes a constant radius, as HRTFs are typically measured on a sphere.
    Input:
        coords_deg: Tensor of shape (N, 2) with columns [azimuth, elevation].
        radius: The radius of the sphere.
    Output:
        Tensor of shape (N, 3) with columns [x, y, z].
    """
    # Convert degrees to radians
    azimuth_rad = torch.deg2rad(coords_deg[:, 0])
    elevation_rad = torch.deg2rad(coords_deg[:, 1])

    # Spherical to Cartesian conversion
    x = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = radius * torch.sin(elevation_rad) # Small correction: radius should not be squared here

    return torch.stack([x, y, z], dim=1)


def compute_laplacian(y, x):
    """
    Computes the Laplacian of a multi-channel output y w.r.t. a 3D input x.
    Args:
        y (torch.Tensor): The model's output tensor of shape (N, F).
                          This is the result of the forward pass.
        x (torch.Tensor): The input tensor of shape (N, 3) for which derivatives
                          are computed. It must have `requires_grad=True`.

    Returns:
        torch.Tensor: Tensor of shape (N, F) representing the Laplacian for each output channel.
    """
    N, F = y.shape
    laplacian = torch.zeros_like(y)

    # Loop over each frequency of the output
    for i in range(F):
        first_grads = torch.autograd.grad(
            y[:, i].sum(),
            x,
            create_graph=True
        )[0]

        # Now, compute the second derivative for this channel
        laplacian_i = 0
        for j in range(x.shape[1]): # Loop over spatial dimensions (0, 1, 2)
            second_grads = torch.autograd.grad(
                first_grads[:, j].sum(),
                x,
                retain_graph=True
            )[0]
            # We only need the diagonal component of the Hessian, d^2y_i / dx_j^2
            laplacian_i += second_grads[:, j]

        laplacian[:, i] = laplacian_i


    return laplacian


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


def merged_collate_fn(batch):
    """
    collate function for MergedHRTFDataset using pad_sequence.
    Input: list of tuples (location_tensor, hrtf_tensor, anthro_tensor, name)
    Output: tuple (padded_locs, padded_hrtfs, masks, collated_anthros, names)
    """
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]
    if not valid_batch:
        return None

    # Unzip the batch items
    locations, hrtfs, anthros, names = zip(*valid_batch)
    if not locations:
        return None

    # Pad sequences
    padded_locs = torch.nn.utils.rnn.pad_sequence(locations, batch_first=True, padding_value=0.0)
    padded_hrtfs = torch.nn.utils.rnn.pad_sequence(hrtfs, batch_first=True, padding_value=0.0)

    # Create masks
    max_num_loc = padded_locs.shape[1]
    masks = torch.zeros((len(locations), max_num_loc, 1), dtype=torch.float32)
    for i, loc_tensor in enumerate(locations):
        masks[i, :loc_tensor.shape[0], :] = 1.0

    collated_anthros = torch.stack(anthros, dim=0)

    return padded_locs, padded_hrtfs, masks, collated_anthros, list(names)

def calculate_anthro_stats(dataset_subset):
    """
    Calculates mean and std dev for anthropometry across the dataset subset.
    """
    if not isinstance(dataset_subset, Subset):
        raise TypeError("calculate_anthro_stats expects a PyTorch Subset object.")

    print("Calculating anthropometry statistics on training set...")
    all_anthro_tensors = []

    for i in range(len(dataset_subset)):
        try:
            _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]
            all_anthro_tensors.append(anthro_tensor)
        except Exception as e:
            print(f"Warning: Could not retrieve item for original dataset index {dataset_subset.indices[i]}: {e}. Skipping.")
            continue

    if not all_anthro_tensors:
        raise ValueError("No valid anthropometric data was collected from the training set.")

    all_anthro_stacked = torch.stack(all_anthro_tensors, dim=0)
    mean = torch.mean(all_anthro_stacked, dim=0, dtype=torch.float32)
    std = torch.std(all_anthro_stacked, dim=0)
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
    ANTHROPOMETRY_DIM = 19
    NUM_FREQ_BINS = 92
    TARGET_FREQ_DIM = NUM_FREQ_BINS
    DECODER_INIT = 'siren'
    DECODER_NONLINEARITY = 'sine'
    SEED = 16

    os.makedirs(SAVE_DIR, exist_ok=True)

    def seed_worker(worker_id):
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

    # ==============================================================================
    # === NEW: SETUP FOR HELMHOLTZ CONSTRAINT ======================================
    # ==============================================================================
    # Create a tensor of frequencies in Hz corresponding to the model's output bins.
    # This is a crucial step. We assume a linear spacing up to ~20kHz.
    # Adjust this if your dataset's frequency mapping is different.
    FFT_SIZE = 256
    SAMPLE_RATE = 44100.0
    # The frequency for bin k is k * SAMPLE_RATE / FFT_SIZE
    freq_indices = torch.arange(1, NUM_FREQ_BINS + 1, device=DEVICE) # Indices from 1 to 92
    freqs_hz = freq_indices * (SAMPLE_RATE / FFT_SIZE)

    # Wavenumber k = 2 * pi * f / c
    wavenumbers_k = 2 * math.pi * freqs_hz / C_SOUND
    # We will need k^2 for the loss term.
    k_squared = wavenumbers_k**2

    # ==============================================================================

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
        augment=True,
        aug_prob=args.aug_prob,
        scale=args.scale
    )

    # --- Subject-Aware Splitting Logic ---
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
        subject_idx = unique_subject_keys.index(subject_key)
        registry_to_subject_id[i] = subject_idx

    indices = np.arange(len(merged_dataset))
    groups = np.array([registry_to_subject_id[i] for i in indices])

    N_SPLITS = 5
    kfold = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    fold_lsd = []

    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation...")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices, groups=groups)):
        print("-" * 50)
        print(f"Fold {fold+1}/{N_SPLITS}")
        print("-" * 50)

        train_subset = Subset(merged_dataset, train_indices)
        val_subset = Subset(merged_dataset, val_indices)

        try:
            anthro_mean, anthro_std = calculate_anthro_stats(train_subset)
            anthro_mean, anthro_std = anthro_mean.to(DEVICE), anthro_std.to(DEVICE)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error calculating stats: {e}")
            return

        merged_dataset.set_augmentation(True)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=merged_collate_fn, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        merged_dataset.set_augmentation(False)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=merged_collate_fn, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

        NUM_EPOCHS = args.epochs
        num_training_steps = NUM_EPOCHS * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)

        model = CondHRTFNetwork(
                d_anthro_in=ANTHROPOMETRY_DIM,
                d_inr_out=TARGET_FREQ_DIM,
                d_latent=args.latent_dim,
                d_inr_hidden=args.hidden_dim,
                n_inr_layers=args.n_layers,
                d_inr_in=3
            ).to(DEVICE)


        #model = torch.compile(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_fold_val_loss = float('inf')
        patience, pat_count = 120, 0

        # --- Training Loop ---
        for epoch in range(args.epochs):
            model.train()
            train_loss_accum, train_phys_loss_accum = 0.0, 0.0
            train_lsd_accum, num_valid_points_epoch = 0.0, 0

            for i, batch in enumerate(train_loader):
                locations, target_hrtfs, masks, anthropometry, _ = batch
                locations = locations.to(DEVICE, non_blocking=True)
                target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                anthropometry = anthropometry.to(DEVICE, non_blocking=True)

                anthropometry_norm = normalize_anthro(anthropometry, anthro_mean, anthro_std)
                B, N_loc, _ = locations.shape

                # Expand anthro data to match the number of locations
                anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, -1)

                # --- Data Loss Calculation ---
                optimizer.zero_grad()

                # Convert 2D spherical locations to 3D Cartesian for the model
                locations_cartesian = spherical_to_cartesian(locations.reshape(-1, 2)).reshape(B, N_loc, 3)

                predicted_hrtfs = model(locations_cartesian.reshape(-1, 3), anthro_expanded.reshape(-1, ANTHROPOMETRY_DIM))
                predicted_hrtfs = predicted_hrtfs.reshape(B, N_loc, -1)

                error_squared = torch.square(predicted_hrtfs - target_hrtfs)
                masked_error_squared = error_squared * masks
                num_valid_points_batch = torch.sum(masks)

                loss_data = torch.sum(masked_error_squared) / num_valid_points_batch
                loss = loss_data

                # --- Helmholtz Physics Loss Calculation ---
                loss_phys = torch.tensor(0.0, device=DEVICE)
                if args.lambda_helmholtz > 0:
                    # 1. Generate random "collocation points" on the sphere to enforce physics
                    num_phys_points = B * N_loc # Reuse batch shape for simplicity
                    rand_az = torch.rand(num_phys_points, 1, device=DEVICE) * 360 - 180
                    rand_el = torch.rand(num_phys_points, 1, device=DEVICE) * 180 - 90
                    phys_locs_spherical = torch.cat([rand_az, rand_el], dim=1)

                    # 2. Convert to Cartesian and enable gradient tracking
                    phys_locs_cartesian = spherical_to_cartesian(phys_locs_spherical)
                    phys_locs_cartesian.requires_grad_(True)

                    # 3. Use corresponding anthro data for these points
                    phys_anthro = anthro_expanded.reshape(-1, ANTHROPOMETRY_DIM)

                    # 4. Forward pass and compute Laplacian
                    h_phys_log = model(phys_locs_cartesian, phys_anthro)

                    h_phys = 10**((50 * (h_phys_log + 1)+85) / 20) ## Inverse of all transformations done during preprocessing
                    laplacian_h = compute_laplacian(h_phys, phys_locs_cartesian)

                    # 5. Calculate Helmholtz residual: laplacian(H) + k^2 * H
                    helmholtz_residual = laplacian_h + k_squared.unsqueeze(0) * h_phys
                    loss_phys = torch.mean(torch.square(helmholtz_residual))

                    # 6. Add weighted physics loss to the total loss
                    loss += args.lambda_helmholtz * loss_phys

                loss.backward()
                optimizer.step()
                scheduler.step()


                # --- Logging ---
                train_loss_accum += loss_data.item() * num_valid_points_batch
                if args.lambda_helmholtz > 0:
                    train_phys_loss_accum += loss_phys.item() * num_phys_points
                num_valid_points_epoch += num_valid_points_batch.item()

                with torch.no_grad():
                    if args.scale == 'log': # LSD for log scale data
                        lsd_elements_sq = torch.square(50*predicted_hrtfs - 50*target_hrtfs)
                    else: # LSD for linear scale data (original formula)
                        epsilon = 1e-9
                        pred_abs = torch.abs(predicted_hrtfs) + epsilon
                        gt_abs = torch.abs(target_hrtfs) + epsilon
                        lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                    masked_lsd_elements_sq = lsd_elements_sq * masks
                    batch_lsd_sum_sq = torch.sum(masked_lsd_elements_sq)
                train_lsd_accum += batch_lsd_sum_sq.item()

            # Calculate average epoch losses
            avg_train_loss = train_loss_accum / num_valid_points_epoch if num_valid_points_epoch > 0 else 0
            avg_train_phys_loss = train_phys_loss_accum / (len(train_loader) * B * N_loc) if args.lambda_helmholtz > 0 else 0
            avg_train_lsd = np.sqrt(train_lsd_accum / (num_valid_points_epoch*NUM_FREQ_BINS))

            # --- Validation Loop ---
            model.eval()
            val_loss_accum, val_lsd_accum, val_valid_points_epoch = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None: continue
                    locations, target_hrtfs, masks, anthropometry, _ = batch
                    locations, target_hrtfs, masks, anthropometry = locations.to(DEVICE), target_hrtfs.to(DEVICE), masks.to(DEVICE), anthropometry.to(DEVICE)

                    anthropometry_norm = normalize_anthro(anthropometry, anthro_mean, anthro_std)
                    B, N_loc, _ = locations.shape
                    anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, -1)

                    # Convert to Cartesian for validation
                    locations_cartesian = spherical_to_cartesian(locations.reshape(-1, 2)).reshape(B, N_loc, 3)
                    predicted_hrtfs = model(locations_cartesian.reshape(-1, 3), anthro_expanded.reshape(-1, ANTHROPOMETRY_DIM))
                    predicted_hrtfs = predicted_hrtfs.reshape(B, N_loc, -1)

                    error_squared = torch.square((predicted_hrtfs) - (target_hrtfs))
                    num_valid_points_batch = torch.sum(masks)
                    if num_valid_points_batch > 0:
                        val_loss_accum += torch.sum(error_squared * masks).item()
                        val_valid_points_epoch += num_valid_points_batch.item()
                        # LSD calculation
                        lsd_elements_sq = torch.square(50*(predicted_hrtfs+1) - 50*(target_hrtfs+1)) if args.scale == 'log' else torch.square(20 * torch.log10((torch.abs(target_hrtfs) + 1e-9) / (torch.abs(predicted_hrtfs) + 1e-9)))
                        val_lsd_accum += torch.sum(lsd_elements_sq * masks).item()

            avg_val_loss = val_loss_accum / val_valid_points_epoch if val_valid_points_epoch > 0 else 0
            avg_val_lsd = np.sqrt(val_lsd_accum / (val_valid_points_epoch * NUM_FREQ_BINS)) if val_valid_points_epoch > 0 else 0

            pat_count += 1
            print(f'Fold {fold+1} | Epoch [{epoch+1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Phys Loss: {avg_train_phys_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val LSD: {avg_val_lsd:.4f}')

            if avg_val_loss < best_fold_val_loss and val_valid_points_epoch > 0:
                pat_count=0
                best_fold_val_loss, best_fold_val_lsd = avg_val_loss, avg_val_lsd
                print(f'---> Model improved, saving checkpoint...')
                # model_save_path = os.path.join(SAVE_DIR, f'model_best_fold_{fold+1}.pth')
                # torch.save({
                #         'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(), 'loss': best_fold_val_loss,
                #         'anthro_mean': anthro_mean.cpu(), 'anthro_std': anthro_std.cpu(),
                #         }, model_save_path)
            if pat_count > patience: break

        print(f"\n---> Best Validation Loss for Fold {fold+1}: {best_fold_val_loss:.6f}\n")
        fold_results.append(best_fold_val_loss)
        fold_lsd.append(best_fold_val_lsd)

    # --- Final Results ---
    mean_cv_loss, std_cv_loss = np.mean(fold_results), np.std(fold_results)
    mean_cv_lsd, std_cv_lsd = np.mean(fold_lsd), np.std(fold_lsd)

    print("=" * 50)
    print("Cross-Validation Finished")
    print(f"Average Validation Loss: {mean_cv_loss:.6f} +/- {std_cv_loss:.6f}")
    print(f"Average Validation LSD: {mean_cv_lsd:.6f} +/- {std_cv_lsd:.6f}")
    print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HRTF Anthropometry Model Training")

    # Path Arguments
    parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common")
    parser.add_argument('--save_dir', type=str, default="/export/mkrishn9/hrtf_field/experiments_test")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--include_datasets', type=str, default="all")
    parser.add_argument('--load_checkpoint', type=str, default=None)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--scale', type=str, default="log")
    parser.add_argument('--eta_min', type=float, default=1e-7)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--aug_prob', type=float, default=0.5)

    # ==============================================================================
    # === NEW ARGUMENT FOR HELMHOLTZ CONSTRAINT ====================================
    # ==============================================================================
    parser.add_argument('--lambda_helmholtz', type=float, default=0.0,
                        help='Weighting factor for the Helmholtz physics loss. Set to 0 to disable.')
    # ==============================================================================


    args = parser.parse_args()

    # --- IMPORTANT NOTE ---
    # The model definition in `model.py` must be updated to accept 3D coordinates.
    # Change the `coord_dim` parameter in its constructor from 2 to 3.
    # Example: In HRTFNetwork.__init__, ensure self.coord_encoder uses `coord_dim=3`.

    print("=============================================")
    print("       HRTF Anthropometry Model Training     ")
    print(" (with Helmholtz Physics-Informed Constraint)")
    print("=============================================")
    if args.lambda_helmholtz > 0:
        print(f"Helmholtz constraint ENABLED with lambda = {args.lambda_helmholtz}")
    else:
        print("Helmholtz constraint DISABLED.")
    print("---------------------------------------------")

    train(args)