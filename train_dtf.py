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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


import math
from HRTFdatasets import MergedHRTFDataset, MergedHRTFDataset1
from model import (
        CondHRTFNetwork,
        CondHRTFNetwork_FiLM,
        CondHRTFNetwork_Attention,
        CondHRTFNetwork_conv,
        CondHRTFNetwork_with_Dropout,
    )

import logging
from collections import defaultdict

def plot_2d_error_map(locations, errors, subject_name, save_path):
    """
    Generates a 2D scatter plot of error magnitude vs. spatial location.

    Args:
        locations (np.ndarray): Array of locations. Shape: (N_loc, 2).
        errors (np.ndarray): Array of errors (e.g., mean LSD) for each location. Shape: (N_loc,).
        subject_name (str): The name of the subject for the title.
        save_path (str): Full path to save the plot image.
    """
    if len(locations) == 0:
        print(f"Skipping error map for {subject_name} as no locations were provided.")
        return

    azimuths = locations[:, 0]
    elevations = locations[:, 1]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the scatter plot where color indicates error
    scatter = ax.scatter(azimuths, elevations, c=errors, cmap='inferno', s=50, alpha=0.8)

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Average LSD (dB)', fontsize=12)

    ax.set_title(f'Spatial Error Map for Subject: {subject_name}', fontsize=16)
    ax.set_xlabel('Azimuth (°)', fontsize=12)
    ax.set_ylabel('Elevation (°)', fontsize=12)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # Make the plot aspect ratio correct

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 2D error map to {save_path}")


def run_full_validation_and_visualize(model, val_loader, anthro_mean, anthro_std, unique_locs_gpu, averages_gpu, device, save_dir, fold):
    """
    Runs a full pass over the validation set to collect errors and generate visualizations.
    """
    model.eval()
    results_by_subject = defaultdict(lambda: {'locs': [], 'errors': []})

    print("\n--- Running Full Validation for Visualization ---")
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            locations, target_hrtfs, masks, anthropometry, names = batch

            # Move to device
            locations, target_hrtfs, masks, anthropometry = locations.to(device), target_hrtfs.to(device), masks.to(device), anthropometry.to(device)
            anthropometry_norm = normalize_anthro(anthropometry, anthro_mean.to(device), anthro_std.to(device))

            # Expand and flatten inputs
            B, N_loc, _ = locations.shape
            anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, -1)
            locations_flat = locations.reshape(-1, 2)
            anthro_flat = anthro_expanded.reshape(-1, anthropometry_norm.shape[-1])

            # Get predictions
            pop_avg_batch = get_population_average_batch_fast(locations, unique_locs_gpu, averages_gpu)
            predicted_residuals = model(locations_flat, anthro_flat).reshape(B, N_loc, -1)
            predicted_hrtfs = pop_avg_batch + predicted_residuals

            # Calculate error (LSD) per location
            error_sq_per_freq = torch.square(predicted_hrtfs - target_hrtfs) * masks
            # Average LSD over the frequency dimension
            mean_error_sq_per_loc = error_sq_per_freq.mean(dim=-1)
            avg_lsd_per_loc = torch.sqrt(mean_error_sq_per_loc)

            # Store results for each subject in the batch
            for i in range(B):
                subject_name = "id"
                valid_indices = torch.where(masks[i, :, 0] == 1)[0]
                if len(valid_indices) > 0:
                    results_by_subject[subject_name]['locs'].append(locations[i, valid_indices, :].cpu().numpy())
                    results_by_subject[subject_name]['errors'].append(avg_lsd_per_loc[i, valid_indices].cpu().numpy())

    # Now, consolidate and plot for a few subjects
    print(f"Collected results for {len(results_by_subject)} subjects.")
    subjects_to_plot = list(results_by_subject.keys())[:1] # Plot for the first 3 subjects

    for subject in subjects_to_plot:
        all_locs = np.concatenate(results_by_subject[subject]['locs'], axis=0)
        all_errors = np.concatenate(results_by_subject[subject]['errors'], axis=0)

        plot_path = os.path.join(save_dir, f"fold_{fold+1}_errormap_{subject}.png")
        plot_2d_error_map(all_locs, all_errors, subject, plot_path)

def get_frequency_axis(num_freq_bins=92, fs=44100, n_fft=256):
    """Generates the frequency axis in Hz for plotting."""
    freq_bins = np.arange(1, num_freq_bins + 1)
    return freq_bins * fs / n_fft


def analyze_latent_space(model, dataloader, anthro_mean, anthro_std, device, save_path):
    """
    Extracts latent vectors 'z' for all subjects in the dataloader,
    and visualizes them using PCA and t-SNE.
    """
    model.eval()
    all_latents = []
    all_datasets = []
    subject_tracker = set()

    print("\n--- Analyzing Latent Space ---")
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            _, _, _, anthropometry, names = batch

            for i in range(anthropometry.shape[0]):
                dataset_name = names[i]

                # Process one subject at a time
                anthro_single = anthropometry[i].unsqueeze(0).to(device)
                anthro_norm = normalize_anthro(anthro_single, anthro_mean.to(device), anthro_std.to(device))

                # Use the new method to get the latent vector
                # Need to unwrap the model if it's compiled
                model_to_run = model._orig_mod if hasattr(model, '_orig_mod') else model
                latent_z = model_to_run.get_latent_vector(anthro_norm)

                all_latents.append(latent_z.cpu().numpy())
                all_datasets.append(dataset_name)

    latent_matrix = np.concatenate(all_latents, axis=0)
    print(f"Collected {latent_matrix.shape[0]} unique subject latents for analysis.")

    # Perform PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latent_matrix)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(latent_matrix)-1), random_state=42)
    latents_tsne = tsne.fit_transform(latent_matrix)

    # Create DataFrame for plotting with seaborn
    df = pd.DataFrame({
        'pca1': latents_pca[:, 0], 'pca2': latents_pca[:, 1],
        'tsne1': latents_tsne[:, 0], 'tsne2': latents_tsne[:, 1],
        'dataset': all_datasets
    })

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='dataset', ax=ax1, s=50, alpha=0.8)
    ax1.set_title('Latent Space Visualization (PCA)', fontsize=16)
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend(title='Dataset')
    ax1.grid(True)

    sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='dataset', ax=ax2, s=50, alpha=0.8)
    ax2.set_title('Latent Space Visualization (t-SNE)', fontsize=16)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend(title='Dataset')
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved latent space analysis plot to {save_path}")

def plot_error_spectrum(pred_hrtf, target_hrtf, pop_avg_hrtf, freqs_hz, location, save_path):
    """
    Plots the model's prediction vs. ground truth and population average for a single location.
    Also plots the error (LSD) on a secondary y-axis.

    Args:
        pred_hrtf (torch.Tensor): The model's predicted HRTF for one location. Shape: (num_freq_bins,).
        target_hrtf (torch.Tensor): The ground truth HRTF. Shape: (num_freq_bins,).
        pop_avg_hrtf (torch.Tensor): The population average HRTF. Shape: (num_freq_bins,).
        freqs_hz (np.ndarray): The frequency values for the x-axis.
        subject_name (str): The name of the subject for the plot title.
        location (tuple): The (azimuth, elevation) for the plot title.
        save_path (str): The full path to save the plot image.
    """
    # Move tensors to CPU and convert to NumPy for plotting
    pred_np = pred_hrtf.detach().cpu().numpy()
    target_np = target_hrtf.detach().cpu().numpy()
    pop_avg_np = pop_avg_hrtf.detach().cpu().numpy()

    # Calculate Log-Spectral Distance (LSD) for log-scaled inputs
    lsd_per_freq = (pred_np - target_np)**2

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot the HRTF magnitudes
    ax1.plot(freqs_hz, target_np, label='Ground Truth', color='black', linewidth=2.5)
    ax1.plot(freqs_hz, pred_np, label='Model Prediction', color='crimson', linestyle='--', linewidth=2)
    ax1.plot(freqs_hz, pop_avg_np, label='Population Average', color='dodgerblue', linestyle=':', linewidth=2)

    ax1.set_title(f'HRTF Spectrum Comparison \nLocation: Az {location[0]}°, El {location[1]}°', fontsize=16)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create a second y-axis for the error plot
    ax2 = ax1.twinx()
    ax2.plot(freqs_hz, lsd_per_freq, label='Squared Error (LSD)', color='darkorange', alpha=0.6)
    ax2.set_ylabel('Squared Error (dB²)', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(bottom=0) # Error can't be negative
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved error spectrum plot to {save_path}")



def prepare_population_averages(train_subset, num_freq_bins=92, device='cpu'):
    """
    Computes population average HRTFs and prepares tensors for a FAST and ROBUST integer-based GPU lookup.
    """

    location_hrtf_map = defaultdict(list)
    print("Computing population averages from training data...")
    for i in range(len(train_subset)):
        try:
            locs, hrtfs, _, _ = train_subset.dataset[train_subset.indices[i]]
            if locs is not None and len(locs) > 0:
                for loc, hrtf in zip(locs, hrtfs):
                    # Use rounded values as keys
                    loc_key = tuple(np.round(loc.numpy(), 2))
                    location_hrtf_map[loc_key].append(hrtf)
        except Exception as e:
            print(f"Warning: Error processing item {i}: {e}")
            continue

    population_avg_dict = {}
    for loc_key, hrtf_list in location_hrtf_map.items():
        population_avg_dict[loc_key] = torch.stack(hrtf_list).mean(dim=0)
    print(f"Computed averages for {len(population_avg_dict)} unique locations")

    if not population_avg_dict:
        return torch.empty((0, 2), device=device, dtype=torch.long), torch.empty((0, num_freq_bins), device=device)

    sorted_locs = sorted(population_avg_dict.keys())

    unique_locs_float = torch.tensor(list(sorted_locs), dtype=torch.float32)
    averages_tensor = torch.stack([population_avg_dict[loc] for loc in sorted_locs])

    # --- NEW: Create an integer version for robust lookup ---
    # Multiply by 100 and convert to a long integer tensor
    unique_locs_int = torch.round(unique_locs_float * 100).long()

    # Move tensors to the target device
    unique_locs_int = unique_locs_int.to(device)
    averages_tensor = averages_tensor.to(device)

    print("Population average tensors (integer-based) prepared for GPU lookup.")
    # Return both the integer locations and the float averages
    return unique_locs_int, averages_tensor


def get_population_average_batch_fast(batch_locations, unique_locs_tensor_int, averages_tensor):
    """
    Gets population averages using a fast, ROBUST, integer-based GPU lookup.
    """
    B, N_loc, _ = batch_locations.shape
    device = unique_locs_tensor_int.device

    # --- UPDATED: Convert incoming batch locations to integers ---
    batch_locs_int = torch.round(batch_locations.to(device) * 100).long()

    # Reshape for broadcasting
    flat_locs_int = batch_locs_int.view(-1, 2)

    # --- UPDATED: Use exact integer comparison ---
    # `.all(-1)` checks if both coordinates (azimuth, elevation) match exactly
    matches = (flat_locs_int.unsqueeze(1) == unique_locs_tensor_int.unsqueeze(0)).all(-1)

    # Find the index of the matching unique location for each item in the batch
    indices = torch.argmax(matches.int(), dim=1)

    # Create a mask to handle locations not found in the averages
    found_mask = matches.any(dim=1)

    # --- For debugging: Check your match rate ---
    # match_rate = 100.0 * found_mask.sum() / found_mask.numel()
    # print(f"Lookup Match Rate: {match_rate:.2f}%")

    # Gather the averages using the found indices
    flat_pop_avg = averages_tensor[indices]

    # Zero out the entries where no match was found
    flat_pop_avg[~found_mask] = 0.0

    # Reshape back to the original batch structure
    pop_avg_batch = flat_pop_avg.view(B, N_loc, -1)

    return pop_avg_batch


def compute_population_averages(train_subset, num_freq_bins=92):
    """Compute population average HRTF for each unique location."""
    from collections import defaultdict
    import numpy as np

    location_hrtf_map = defaultdict(list)

    print("Computing population averages from training data...")
    for i in range(len(train_subset)):
        try:
            if hasattr(train_subset, 'dataset'):
                # Handle Subset
                locs, hrtfs, _, _ = train_subset.dataset[train_subset.indices[i]]
            else:
                locs, hrtfs, _, _ = train_subset[i]

            if locs is not None and len(locs) > 0:
                for loc, hrtf in zip(locs, hrtfs):
                    # Round location for consistent hashing
                    loc_key = tuple(np.round(loc.numpy() if isinstance(loc, torch.Tensor) else loc, 2))
                    location_hrtf_map[loc_key].append(hrtf)
        except Exception as e:
            print(f"Warning: Error processing item {i}: {e}")
            continue

    # Compute averages
    population_avg = {}
    for loc_key, hrtf_list in location_hrtf_map.items():
        population_avg[loc_key] = torch.stack(hrtf_list).mean(dim=0)

    print(f"Computed averages for {len(population_avg)} unique locations")
    return population_avg

def get_population_average_batch(locations, population_avg, num_freq_bins=92):
    """
    Get population averages for a batch of locations.
    Falls back to zero if location not in training set.
    """
    B, N_loc, _ = locations.shape
    pop_avg_batch = torch.zeros(B, N_loc, num_freq_bins, device=locations.device)

    for b in range(B):
        for n in range(N_loc):
            loc = locations[b, n].cpu().numpy()
            loc_key = tuple(np.round(loc, 2))
            if loc_key in population_avg:
                pop_avg_batch[b, n] = population_avg[loc_key].to(locations.device)

    return pop_avg_batch


def plot_lsd_vs_frequency(total_lsd_sq_per_freq, total_valid_points_per_freq, num_freq, save_path=None):

    if isinstance(total_lsd_sq_per_freq, torch.Tensor):
        total_lsd_sq_per_freq = total_lsd_sq_per_freq.cpu().numpy()
    if isinstance(total_valid_points_per_freq, torch.Tensor):
        total_valid_points_per_freq = total_valid_points_per_freq.cpu().numpy()



    mean_lsd_sq_per_freq = total_lsd_sq_per_freq / total_valid_points_per_freq

    avg_lsd_per_freq = np.sqrt(mean_lsd_sq_per_freq)


    fs = 44100
    N_fft = 256   # Number of points in the FFT.


    freq_bins = np.arange(1, num_freq + 1)
    frequencies_hz = freq_bins * fs / N_fft


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(frequencies_hz, avg_lsd_per_freq, label='Average LSD', color='dodgerblue', linewidth=2)



    ax.set_title('Average Log-Spectral Distance (LSD) vs. Frequency', fontsize=16, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Average LSD (dB)', fontsize=12)


    ax.legend()
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.close()



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


    locations, hrtfs, anthros, names, zero_freq_tensor = zip(*valid_batch)

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
    collated_zero_freqs = torch.stack(zero_freq_tensor, dim=0)


    return padded_locs, padded_hrtfs, masks, collated_anthros,collated_zero_freqs, list(names)

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
            _, _, anthro_tensor, _, _ = dataset_subset.dataset[dataset_subset.indices[i]]
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

def weighted_mse_loss(model_output, hrtf_target, zero_freq_bins, k=2, weight_factor=5.0):
    """
    Calculates MSE, applying a higher weight to specified frequency bins and their
    k-nearest neighbors.

    Args:
        model_output (Tensor): The model's prediction. Shape [B, N, D].
        hrtf_target (Tensor): The ground truth. Shape [B, N, D].
        zero_freq_bins (Tensor): Tensor of central bin indices. Shape [B, N, M].
                                 Contains -1 for padded values.
        k (int): The number of neighbors on each side of the central bin to also apply the high weight to.
                 k=0 is the original behavior. k=1 weights the bin itself and its immediate neighbors.
        weight_factor (float): The weight to apply to the special bins.
    """
    # Get the shape and device from the model output
    B, N, D = model_output.shape
    device = model_output.device

    # Start with a base weight of 1.0 for all coordinates
    weights = torch.ones_like(model_output)

    # 1. Create a mask to identify valid (non-padded) central bins.
    # Padded values are -1, so we check for indices >= 0.
    valid_mask = zero_freq_bins >= 0  # Shape: [B, N, M]

    # 2. Generate the offsets for the k-neighborhood.
    # For k=1, this will be tensor([-1, 0, 1]).
    offsets = torch.arange(-k, k + 1, device=device) # Shape: [2*k + 1]

    # 3. Calculate all neighbor indices.
    # We use broadcasting to add the offsets to each central bin.
    # zero_freq_bins [B, N, M] -> unsqueeze -> [B, N, M, 1]
    # offsets [2k+1]
    # Resulting shape of neighbor_indices is [B, N, M, 2*k + 1]
    neighbor_indices = zero_freq_bins.unsqueeze(-1) + offsets

    # 4. Clamp the indices to ensure they are within the valid frequency range [0, D-1].
    # This prevents out-of-bounds errors.
    clamped_indices = torch.clamp(neighbor_indices, 0, D - 1)
    flat_indices = clamped_indices.reshape(B, N, -1)

    expanded_mask = valid_mask.unsqueeze(-1).expand_as(clamped_indices)

    # FIX: Use .reshape() because .expand_as() creates a non-contiguous tensor
    flat_mask = expanded_mask.reshape(B, N, -1)



    # Create the source tensor: it has `weight_factor` where the mask is true, and 1.0 otherwise.
    # This ensures we don't apply high weights for the neighbors of padded (-1) indices.
    src = torch.full(flat_indices.shape, 1.0, device=device, dtype=torch.float)
    src[flat_mask] = weight_factor

    # 6. Apply the weights using scatter_.
    # This operation places the values from `src` into the `weights` tensor at the
    # locations specified by `flat_indices`. If multiple neighbors map to the same
    # index, the weight will just be overwritten, which is fine.
    weights.scatter_(dim=2, index=flat_indices, src=src)

    # 7. Calculate the final weighted squared error.
    loss = weights * (model_output - hrtf_target)**2




    return torch.mean(loss)


# --- Main Training Function ---

def train(args):
    # --- Configuration ---
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    SAVE_DIR = os.path.join(args.save_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)


    print(f"Using device: {DEVICE}")
    print(f"Checkpoints and logs will be saved to: {SAVE_DIR}")




    DATA_PATH = args.data_path

    ANTHROPOMETRY_DIM = 10 #10 # 23 ### HARDCODED

    # *******************************

    # Model & Training Parameters
    NUM_FREQ_BINS = 92             # Number of frequency bins (1 to 92 inclusive)
    TARGET_FREQ_DIM = NUM_FREQ_BINS # Model output dimension
    DECODER_INIT = 'siren'         # Initialization type for decoder (ensure model.py handles this)
    DECODER_NONLINEARITY = 'sine'  # Activation function for decoder


    SEED = 16
    freqs_hz = get_frequency_axis(NUM_FREQ_BINS)



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

    merged_dataset = MergedHRTFDataset1(
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

    N_SPLITS = 5
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
        #population_avg = prepare_population_averages(train_subset, NUM_FREQ_BINS)
        unique_locs_gpu, averages_gpu = prepare_population_averages(train_subset, NUM_FREQ_BINS, device=DEVICE)


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



        model = CondHRTFNetwork_with_Dropout(
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
            epoch_total_lsd_sq_per_freq =0
            epoch_total_val_lsd_sq_per_freq =0
            residual_accum = 0


            for i, batch in enumerate(train_loader):


                locations, target_hrtfs, masks, anthropometry, zero_freq_bins, _ = batch

                locations = locations.to(DEVICE, non_blocking=True)
                target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)

                pop_avg_batch = get_population_average_batch_fast(locations, unique_locs_gpu, averages_gpu)


                masks = masks.to(DEVICE, non_blocking=True)
                anthropometry = anthropometry.to(DEVICE, non_blocking=True)
                zero_freq_bins = zero_freq_bins.to(DEVICE, non_blocking=True)
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

                predicted_residuals_flat = model(locations_flat, anthro_flat) # Output: (B*N_loc, num_freq)
                # Reshape output back: (B*N_loc, num_freq) -> (B, N_loc, num_freq)
                predicted_residuals = predicted_residuals_flat.reshape(B, N_loc, num_freq)
                predicted_hrtfs = predicted_residuals + pop_avg_batch



                residual_reg = 0 #1e-4 * torch.mean(torch.square(predicted_residuals) * masks)
                residual_accum += torch.mean(torch.square(predicted_residuals) * masks)


                diff = predicted_hrtfs[..., 1:] - predicted_hrtfs[..., :-1]
                smoothness_loss= torch.mean(torch.pow(diff, 2))
                lamb_val = 0

                # with torch.no_grad():
                #     residual_magnitude = torch.mean(torch.abs(predicted_residuals*masks))
                #     pop_avg_magnitude = torch.mean(torch.abs(pop_avg_batch*masks))
                #     if (epoch + 1) % 10 == 0:
                #         print(f"Residual/PopAvg magnitude ratio: {(residual_magnitude/pop_avg_magnitude):.3f}")

                # Check shapes before loss calculation
                if predicted_hrtfs.shape != target_hrtfs.shape:
                    print(f"Shape mismatch before loss! Predicted: {predicted_hrtfs.shape}, Target: {target_hrtfs.shape}. Skipping batch {i+1}.")
                    continue

                # Calculate loss
                error_squared = torch.abs(predicted_hrtfs - target_hrtfs) #torch.square(predicted_hrtfs - target_hrtfs) + lamb_val*padded_diff_sq #weighted_mse_loss(predicted_hrtfs, target_hrtfs, zero_freq_bins, weight_factor=5.0) #torch.square(predicted_hrtfs - target_hrtfs)
                masked_error_squared = error_squared * masks
                # Sum loss over all valid points in the batch
                # Normalize by the number of valid points (sum of mask)
                num_valid_points_batch = torch.sum(masks)

                loss = torch.sum(masked_error_squared) / num_valid_points_batch + lamb_val*torch.sum(smoothness_loss) + residual_reg


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
                    epoch_total_lsd_sq_per_freq += torch.sum(masked_lsd_elements_sq, dim=(0, 1))




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

            if (epoch+1)%100 ==0:
                plot_lsd_vs_frequency(
                        epoch_total_lsd_sq_per_freq,
                        num_valid_points_epoch,
                        NUM_FREQ_BINS,
                        save_path='/export/mkrishn9/hrtf_field/experiments_test/img.png' # Example of saving the plot
                )





            # --- Validation Loop (needs similar masking logic) ---
            model.eval()
            val_loss_accum = 0.0
            val_lsd_accum = 0.0
            val_valid_points_epoch = 0
            with torch.no_grad():
                plotted_this_epoch = False
                for batch_idx, batch in enumerate(val_loader): # <--- Use enumerate to get batch_idx
                    if batch is None: continue

                    locations, target_hrtfs, masks, anthropometry, zero_freq_bins, _ = batch
                    locations = locations.to(DEVICE, non_blocking=True)
                    target_hrtfs = target_hrtfs.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)
                    anthropometry = anthropometry.to(DEVICE, non_blocking=True)
                    zero_freq_bins = zero_freq_bins.to(DEVICE, non_blocking=True)


                    anthropometry_norm = normalize_anthro(anthropometry, anthro_mean.to(DEVICE), anthro_std.to(DEVICE))

                    # Expand and flatten inputs
                    B, N_loc, _ = locations.shape
                    anthro_dim = anthropometry_norm.shape[-1]
                    num_freq = target_hrtfs.shape[-1]
                    anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, anthro_dim)
                    locations_flat = locations.reshape(-1, 2)
                    anthro_flat = anthro_expanded.reshape(-1, anthro_dim)

                    # Predict
                    pop_avg_batch = get_population_average_batch_fast(locations, unique_locs_gpu, averages_gpu)
                    predicted_residuals_flat = model(locations_flat, anthro_flat) # Output: (B*N_loc, num_freq)
                    # Reshape output back: (B*N_loc, num_freq) -> (B, N_loc, num_freq)
                    predicted_residuals = predicted_residuals_flat.reshape(B, N_loc, num_freq)
                    predicted_hrtfs =  predicted_residuals  + pop_avg_batch



                    # Calculate masked loss
                    error_squared = torch.square(predicted_hrtfs - target_hrtfs)
                    masked_error_squared = error_squared * masks
                    num_valid_points_batch = torch.sum(masks)
                    if num_valid_points_batch > 0:
                            batch_loss_sum = torch.sum(masked_error_squared)
                            val_loss_accum += batch_loss_sum.item()
                            if args.scale == 'log':
                                lsd_elements_sq = torch.square((predicted_hrtfs) - (target_hrtfs))
                            else:
                                epsilon = 1e-9
                                pred_abs = torch.abs(predicted_hrtfs + epsilon +1)
                                gt_abs = torch.abs(target_hrtfs + epsilon + 1)
                                lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                            masked_lsd_elements_sq = lsd_elements_sq * masks
                            val_lsd_accum += torch.sum(masked_lsd_elements_sq).item()
                            epoch_total_val_lsd_sq_per_freq += torch.sum(masked_lsd_elements_sq, dim=(0, 1))

                            val_valid_points_epoch += num_valid_points_batch.item()

                            if batch_idx == 0 and not plotted_this_epoch and (epoch + 1) % 10 == 0:
                                # Get data for the very first subject in the batch

                                first_valid_loc_idx = torch.where(masks[0, :, 0] == 1)[0][5] # Find first real location

                                pred_to_plot = predicted_hrtfs[0, first_valid_loc_idx, :]
                                target_to_plot = target_hrtfs[0, first_valid_loc_idx, :]
                                pop_avg_to_plot = pop_avg_batch[0, first_valid_loc_idx, :]
                                loc_to_plot = locations[0, first_valid_loc_idx, :].cpu().numpy()

                                plot_save_path = os.path.join(SAVE_DIR, f'fold_{fold+1}_epoch_{epoch+1}_spectrum.png')

                                plot_error_spectrum(
                                    pred_hrtf=pred_to_plot,
                                    target_hrtf=target_to_plot,
                                    pop_avg_hrtf=pop_avg_to_plot,
                                    freqs_hz=freqs_hz,
                                    location=loc_to_plot,
                                    save_path=plot_save_path
                                )
                                plotted_this_epoch = True

                if (epoch + 1) % 10 == 0:
                    print("\n--- Running Final Latent Space Analysis on Last Fold's Model ---")
                    analyze_latent_space(
                        model=model,
                        dataloader=val_loader, # Using the validation loader from the last fold
                        anthro_mean=anthro_mean,
                        anthro_std=anthro_std,
                        device=DEVICE,
                        save_path=os.path.join(SAVE_DIR, 'final_latent_space_analysis.png')
                    )
                    run_full_validation_and_visualize(
                        model=model,
                        val_loader=val_loader,
                        anthro_mean=anthro_mean,
                        anthro_std=anthro_std,
                        unique_locs_gpu=unique_locs_gpu,
                        averages_gpu=averages_gpu,
                        device=DEVICE,
                        save_dir=SAVE_DIR,
                        fold=fold
                    )
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

                plot_lsd_vs_frequency(
                        epoch_total_val_lsd_sq_per_freq,
                        val_valid_points_epoch,
                        NUM_FREQ_BINS,
                        save_path='/export/mkrishn9/hrtf_field/experiments_test/img_val.png' # Example of saving the plot
                )



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
    parser.add_argument('--save_dir', type=str, default="/export/mkrishn9/hrtf_field/experiments_pop_avg", help='Directory to save experiment checkpoints')
    parser.add_argument('--gpu', type=int, default=4, help='GPU device ID to use')
    parser.add_argument('--include_datasets', type=str, default="all",
                        help="Comma-separated list of datasets to use. 'all' uses everything.")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to a model checkpoint to load for fine-tuning.")


    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
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

