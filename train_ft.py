#########Code to pretrain on CHEDAR data and finetune on measured data like HUTUBS and CIPIC.
####### Does not have a cross-validation strategy implemented yet.
#### It also has other issues, like if I am pretraining on CHEDAR then maybe I need to monitor what the validation is on other datasets
### Overall comments: Needs more careful implementation


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
import re
import time
torch.set_float32_matmul_precision('high')


import collections
from HRTFdatasets import MergedHRTFDataset
from model import (
        HRTFNetwork,
        HRTFNetwork_conv
    )

import logging
import matplotlib.pyplot as plt





def plot_midsagittal_plane(locations, hrtfs, title, ax):
    """
    Creates a 2D heatmap of the HRTF on the midsagittal plane,
    using a sophisticated sort to create a continuous front-to-back path.

    Args:
        locations (torch.Tensor): Tensor of shape (num_locations, 2) for [azimuth, elevation].
        hrtfs (torch.Tensor): Tensor of shape (num_locations, num_freq_bins).
        title (str): The title for the plot.
        ax (matplotlib.axes.Axes): The axes object to plot on.
    """
    # Find all points on or near the midsagittal plane (azimuth ≈ 0° or 180°)
    front_indices = torch.where(torch.abs(locations[:, 0]) < 5)[0]
    back_indices = torch.where(torch.abs(locations[:, 0] - 180) < 5)[0]
    midsagittal_indices = torch.cat((front_indices, back_indices))

    if len(midsagittal_indices) == 0:
        ax.set_title(f"{title}\n(No midsagittal plane data)")
        return

    # Get the corresponding coordinates and HRTFs
    midsagittal_coords = locations[midsagittal_indices]
    midsagittal_hrtfs = hrtfs[midsagittal_indices]


    coords_np = midsagittal_coords.cpu().numpy()
    azimuths = coords_np[:, 0]
    elevations = coords_np[:, 1]


    sort_key = elevations - (azimuths / 90 * elevations) + azimuths
    sorted_indices = np.argsort(sort_key)
    sorted_hrtfs = midsagittal_hrtfs[sorted_indices].cpu().numpy()


    # For simplicity, we'll just use a continuous index for the y-axis ticks.
    # A more advanced version could label the elevations along the path.


    img = ax.imshow(sorted_hrtfs.T, aspect='auto', origin='lower',
                    extent=[0, len(sorted_indices), 0, 16000]) # Assuming max freq is 16kHz

    ax.set_title(title)
    ax.set_xlabel("Location Index along Midsagittal Path")
    ax.set_ylabel("Frequency (Hz)")

    return img




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
    DECODER_INIT = 'siren'         # Initialization type for decoder
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
        augment=True,
        aug_prob=args.aug_prob,
        scale=args.scale
    )

    # --- Subject-Aware Splitting Logic so that ears of one subject doesnt get mixed up in training and validation ---
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

    all_subjects = list(subjects.keys())
    random.shuffle(all_subjects)
    num_val_subjects = int(len(all_subjects) * args.val_split)
    val_subject_keys = all_subjects[:num_val_subjects]
    train_subject_keys = all_subjects[num_val_subjects:]
    train_indices = [idx for key in train_subject_keys for idx in subjects[key]]
    val_indices = [idx for key in val_subject_keys for idx in subjects[key]]
    print(f"Total unique subjects: {len(all_subjects)}, Train: {len(train_subject_keys)}, Val: {len(val_subject_keys)}")
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

    merged_dataset.set_augmentation(True)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=merged_collate_fn,num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    merged_dataset.set_augmentation(False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=merged_collate_fn,num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    NUM_EPOCHS = args.epochs



    # --- Model, Optimizer, Loss ---
    print("Initializing model...")


    model = HRTFNetwork_conv(
            anthropometry_dim=ANTHROPOMETRY_DIM,
            target_freq_dim=TARGET_FREQ_DIM,
            latent_dim=args.latent_dim,
            decoder_hidden_dim=args.hidden_dim,
            decoder_n_hidden_layers=args.n_layers,
            init_type=DECODER_INIT,
            nonlinearity=DECODER_NONLINEARITY
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




    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=args.eta_min)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Starting training...")
    best_val_loss = float('inf')

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
                if args.scale == 'log': # LSD for log scale data with the (-1,1) normalization corrected
                    lsd_elements_sq = torch.square(50*(predicted_hrtfs+1) - 50*(target_hrtfs+1)) # error_squared is already this
                else: # LSD for linear scale data (original formula)
                    epsilon = 1e-9
                    pred_abs = torch.abs(predicted_hrtfs) + epsilon
                    gt_abs = torch.abs(target_hrtfs) + epsilon
                    lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                masked_lsd_elements_sq = lsd_elements_sq * masks
                batch_lsd_sum_sq = torch.sum(masked_lsd_elements_sq)


            loss.backward()
            optimizer.step()


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
        avg_val_loss =0
        avg_val_lsd =0
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
                            lsd_elements_sq = torch.square(predicted_hrtfs - target_hrtfs)
                        else:
                            epsilon = 1e-9
                            pred_abs = torch.abs(predicted_hrtfs) + epsilon
                            gt_abs = torch.abs(target_hrtfs) + epsilon
                            lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))
                        masked_lsd_elements_sq = lsd_elements_sq * masks
                        val_lsd_accum += torch.sum(masked_lsd_elements_sq).item()

                        val_valid_points_epoch += num_valid_points_batch.item()
        # Calculate average epoch validation loss and LSD
        if val_valid_points_epoch > 0:
            avg_val_loss = val_loss_accum / val_valid_points_epoch
            avg_val_lsd = np.sqrt(val_lsd_accum / (val_valid_points_epoch * NUM_FREQ_BINS))



        scheduler.step()
        # epoch_end_time = time.time()

        # # # Calculate and print the elapsed time
        # elapsed_time = epoch_end_time - epoch_start_time
        # print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds.")

        # model_save_path = os.path.join(SAVE_DIR, f'hrtf_chedar_pretrained.pth')

        # torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': best_val_loss,
        #         'anthro_mean': anthro_mean.cpu(),
        #         'anthro_std': anthro_std.cpu(),
        #         }, model_save_path)

        print(f'Epoch [{epoch+1:03d}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train LSD: {avg_train_lsd:.4f}, Val LSD: {avg_val_lsd:.4f}')
        # print(f'---> Pretrained model saved to {model_save_path}')


        # --- Save Best Model ---
        if val_valid_points_epoch > 0 and avg_val_loss < best_val_loss: # Check validation ran and improved

            best_val_loss = avg_val_loss
            best_val_lsd = avg_val_lsd
            model_save_path = os.path.join(SAVE_DIR, f'hrtf_anthro_pretrained_model.pth')

            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'anthro_mean': anthro_mean.cpu(),
                    'anthro_std': anthro_std.cpu(),
                    }, model_save_path)
            print(f'---> Model improved, saved to {model_save_path}')

        # if epoch % 50 == 0:
        #     locs_to_plot = locations[0]
        #     target_to_plot = target_hrtfs[0]
        #     pred_to_plot = predicted_hrtfs[0]

        #     # Create a figure with two subplots
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        #     fig.suptitle(f"Epoch {epoch+1} - Validation Sample")

        #     # Plot the Ground Truth
        #     plot_midsagittal_plane(locs_to_plot, target_to_plot, "Ground Truth", ax1)

        #     # Plot the Model's Prediction
        #     img = plot_midsagittal_plane(locs_to_plot, pred_to_plot, "Prediction", ax2)

        #     # Add a colorbar
        #     fig.colorbar(img, ax=ax2)
        #     plt.savefig(os.path.join(SAVE_DIR, f"log_epoch_{epoch+1}_validation_plot.png"))
        #     plt.close(fig)




    print("\nTraining finished.")
    if args.load_checkpoint:
        print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
        print(f"Best Validation LSD achieved: {best_val_lsd:.6f}")
        print(f"Check '{SAVE_DIR}' for the best model checkpoint.")
    else:
        print(f"Training Loss achieved: {avg_train_loss:.6f}")
        print(f" Training LSD achieved: {avg_train_lsd:.6f}")
        print(f"Check '{SAVE_DIR}' for the best model checkpoint.")
        if val_valid_points_epoch > 0:
            print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
            print(f"Best Validation LSD achieved: {best_val_lsd:.6f}")
            print(f"Check '{SAVE_DIR}' for the best model checkpoint.")

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="HRTF Anthropometry Model Training")

    # Path Arguments
    parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common", help='Path to preprocessed data')
    parser.add_argument('--save_dir', type=str, default="/export/mkrishn9/hrtf_field/experiments_test", help='Directory to save experiment checkpoints')
    parser.add_argument('--gpu', type=int, default=1, help='GPU device ID to use')
    parser.add_argument('--include_datasets', type=str, default="all",
                        help="Comma-separated list of datasets to use. 'all' uses everything.")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to a model checkpoint to load for fine-tuning.")


    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

    # Model Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Decoder hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--scale', type=str, default="log", help='Scale')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Learning rate annealing')
    parser.add_argument('--val_split', type=float, default=0.15, help='Training testing split')

    parser.add_argument('--augment', type=bool, default=True, help='Augmentation on/off')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability of augmentation')






    # --- 2. Parse Arguments and Run Training ---
    args = parser.parse_args()






    print("=============================================")
    print("       HRTF Anthropometry Model Training     ")
    print("=============================================")
    print(f"Make sure 'hrtf_dataset.py' and 'model.py' are in the Python path.")
    print("---------------------------------------------")

    train(args)

