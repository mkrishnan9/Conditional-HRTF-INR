import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pickle as pkl
import os
import sys
import random
import math
import re
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, '/export/mkrishn9/hrtf_field')
from HRTFdatasets import MergedHRTFDataset1
from train_cv import merged_collate_fn


class Normalizer:
    """Handles normalization and denormalization of data"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        """Compute mean and std from training data"""
        if isinstance(x, torch.Tensor):
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-8
        else:
            self.mean = x.mean()
            self.std = x.std() + 1e-8
        return self

    def transform(self, x):
        """Apply z-score normalization"""
        if self.mean is None or self.std is None:
            raise ValueError("Must fit normalizer before transform")
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        """Denormalize back to original scale"""
        if self.mean is None or self.std is None:
            raise ValueError("Must fit normalizer before inverse_transform")
        return x * self.std + self.mean


def calculate_lsd(pred_hrtf, true_hrtf, in_db=False):
    """
    Calculate Log-Spectral Distance

    Args:
        pred_hrtf: Predicted HRTF (in linear or dB scale)
        true_hrtf: True HRTF (in linear or dB scale)
        in_db: If True, inputs are already in dB scale

    Returns:
        LSD value in dB
    """
    if in_db:
        # Already in dB scale
        diff = pred_hrtf - true_hrtf
    else:
        # Convert to dB scale (assuming magnitude spectrum)
        # Add small epsilon to avoid log(0)
        pred_db = 20 * torch.log10(torch.abs(pred_hrtf) + 1e-10)
        true_db = 20 * torch.log10(torch.abs(true_hrtf) + 1e-10)
        diff = pred_db - true_db

    # LSD = sqrt(mean(diff^2))
    lsd = torch.sqrt(torch.mean(diff ** 2))
    return lsd


class HRTF_VAE(nn.Module):
    """
    Variational Autoencoder for HRTF feature extraction.
    Based on Yao et al. 2022 architecture.
    """
    def __init__(self, input_dim=92, latent_dim=16):
        super(HRTF_VAE, self).__init__()

        # Encoder layers as per paper: 128-128-64-64-32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # VAE specific: output mean and log variance
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder (symmetric structure)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class InputVectorAE(nn.Module):
    """
    Autoencoder for input vector embedding (anthropometry + direction).
    Based on Yao et al. 2022 architecture.
    """
    def __init__(self, anthro_dim=12, coord_dim=3, latent_dim=16):
        super(InputVectorAE, self).__init__()

        input_dim = anthro_dim + coord_dim  # 12 anthro + 3 coordinates

        # Encoder: 32-64-128-64-32 as per paper
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # Decoder (symmetric)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class FeatureMappingDNN(nn.Module):
    """
    DNN for mapping input features to HRTF features.
    Based on Yao et al. 2022 architecture.
    """
    def __init__(self, input_latent_dim=16, hrtf_latent_dim=16):
        super(FeatureMappingDNN, self).__init__()

        # 5 hidden layers with 128 nodes each as per paper
        self.network = nn.Sequential(
            nn.Linear(input_latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hrtf_latent_dim)
        )

    def forward(self, x):
        return self.network(x)


class DualAutoEncoderHRTF(nn.Module):
    """
    Complete dual-autoencoder HRTF individualization model.
    Combines VAE, AE, and DNN components.
    """
    def __init__(self, anthro_dim=10, num_freq=92, latent_dim=16):
        super(DualAutoEncoderHRTF, self).__init__()

        self.hrtf_vae = HRTF_VAE(input_dim=num_freq, latent_dim=latent_dim)
        self.input_ae = InputVectorAE(anthro_dim=anthro_dim, coord_dim=3, latent_dim=latent_dim)
        self.mapper = FeatureMappingDNN(input_latent_dim=latent_dim, hrtf_latent_dim=latent_dim)

        self.anthro_dim = anthro_dim
        self.num_freq = num_freq

        # Initialize normalizers
        self.hrtf_normalizer = Normalizer()
        self.anthro_normalizer = Normalizer()
        self.coord_normalizer = Normalizer()

    def angles_to_coordinates(self, azimuth, elevation):
        """Convert azimuth/elevation to 3D Cartesian coordinates."""
        # Convert to radians
        azi_rad = torch.deg2rad(azimuth)
        ele_rad = torch.deg2rad(elevation)

        # Spherical to Cartesian conversion
        x = torch.cos(ele_rad) * torch.cos(azi_rad)
        y = torch.cos(ele_rad) * torch.sin(azi_rad)
        z = torch.sin(ele_rad)

        return torch.stack([x, y, z], dim=-1)

    def forward(self, locations, anthropometry, hrtfs=None, mode='train'):
        """
        Forward pass through the dual-autoencoder architecture.

        Args:
            locations: [B*N, 2] azimuth and elevation
            anthropometry: [B*N, anthro_dim] anthropometric features
            hrtfs: [B*N, num_freq] target HRTFs (for training)
            mode: 'train' or 'predict'
        """
        # Convert locations to 3D coordinates
        coords = self.angles_to_coordinates(locations[:, 0], locations[:, 1])

        # Normalize inputs using fitted normalizers
        if self.anthro_normalizer.mean is not None:
            anthropometry_norm = self.anthro_normalizer.transform(anthropometry)
        else:
            anthropometry_norm = anthropometry  # First batch, will fit during training

        if self.coord_normalizer.mean is not None:
            coords_norm = self.coord_normalizer.transform(coords)
        else:
            coords_norm = coords

        # Concatenate normalized anthropometry and coordinates
        input_vector = torch.cat([anthropometry_norm, coords_norm], dim=-1)

        if mode == 'train' and hrtfs is not None:
            # Normalize HRTFs
            if self.hrtf_normalizer.mean is not None:
                hrtfs_norm = self.hrtf_normalizer.transform(hrtfs)
            else:
                hrtfs_norm = hrtfs

            # Training mode: use all components
            # 1. Encode HRTFs through VAE
            hrtf_recon, hrtf_mu, hrtf_logvar = self.hrtf_vae(hrtfs_norm)

            # 2. Encode input vector through AE
            input_recon, input_latent = self.input_ae(input_vector)

            # 3. Map input features to HRTF features
            predicted_hrtf_latent = self.mapper(input_latent)

            # 4. Decode predicted latent to get predicted HRTFs
            predicted_hrtfs = self.hrtf_vae.decode(predicted_hrtf_latent)

            return {
                'hrtf_recon': hrtf_recon,
                'hrtf_mu': hrtf_mu,
                'hrtf_logvar': hrtf_logvar,
                'input_recon': input_recon,
                'predicted_hrtfs': predicted_hrtfs,
                'predicted_latent': predicted_hrtf_latent,
                'target_latent': hrtf_mu,
                'hrtfs_norm': hrtfs_norm
            }
        else:
            # Prediction mode: only use input path
            input_latent = self.input_ae.encode(input_vector)
            predicted_hrtf_latent = self.mapper(input_latent)
            predicted_hrtfs_norm = self.hrtf_vae.decode(predicted_hrtf_latent)

            # Denormalize predictions
            if self.hrtf_normalizer.mean is not None:
                predicted_hrtfs = self.hrtf_normalizer.inverse_transform(predicted_hrtfs_norm)
            else:
                predicted_hrtfs = predicted_hrtfs_norm

            return predicted_hrtfs


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss = Reconstruction loss + KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_dual_autoencoder(model, train_loader, val_loader, device,
                           vae_epochs=100, joint_epochs=200,
                           vae_lr=1e-3, joint_lr=1e-3):
    """
    Training procedure for the dual-autoencoder model following paper specifications.
    """

    # Step 0: Fit normalizers on training data
    print("Step 0: Fitting normalizers on training data...")
    all_hrtfs = []
    all_anthro = []
    all_coords = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 20:  # Sample first 20 batches for fitting
                break
            locations, hrtfs, masks, anthropometry, _, _ = batch
            B, N, _ = locations.shape

            locations = locations.reshape(-1, 2).to(device)
            hrtfs = hrtfs.reshape(-1, model.num_freq).to(device)
            masks = masks.to(device)
            anthropometry = anthropometry.unsqueeze(1).expand(B, N, -1)
            anthropometry = anthropometry.reshape(-1, anthropometry.shape[-1]).to(device)

            valid_idx = masks.reshape(-1, 1).squeeze() > 0
            if valid_idx.sum() == 0:
                continue

            valid_hrtfs = hrtfs[valid_idx]
            valid_anthro = anthropometry[valid_idx]
            valid_locations = locations[valid_idx]

            coords = model.angles_to_coordinates(valid_locations[:, 0], valid_locations[:, 1])

            all_hrtfs.append(valid_hrtfs)
            all_anthro.append(valid_anthro)
            all_coords.append(coords)

    if all_hrtfs:
        all_hrtfs = torch.cat(all_hrtfs, dim=0)
        all_anthro = torch.cat(all_anthro, dim=0)
        all_coords = torch.cat(all_coords, dim=0)

        model.hrtf_normalizer.fit(all_hrtfs)
        model.anthro_normalizer.fit(all_anthro)
        model.coord_normalizer.fit(all_coords)

        print(f"  HRTF mean: {model.hrtf_normalizer.mean.mean().item():.4f}, "
              f"std: {model.hrtf_normalizer.std.mean().item():.4f}")

    # Step 1: Pre-train the VAE
    print("\nStep 1: Pre-training VAE...")
    print(f"  VAE epochs: {vae_epochs}, lr: {vae_lr}")

    vae_optimizer = optim.Adam(model.hrtf_vae.parameters(), lr=vae_lr)
    vae_scheduler = CosineAnnealingLR(vae_optimizer, T_max=vae_epochs, eta_min=1e-6)

    for epoch in range(vae_epochs):
        model.hrtf_vae.train()
        vae_loss_accum = 0.0
        batch_count = 0

        for batch in train_loader:
            locations, hrtfs, masks, anthropometry, _, _ = batch
            locations = locations.reshape(-1, 2).to(device)
            hrtfs = hrtfs.reshape(-1, model.num_freq).to(device)
            masks = masks.to(device)

            valid_idx = masks.reshape(-1, 1).squeeze() > 0
            if valid_idx.sum() == 0:
                continue

            valid_hrtfs = hrtfs[valid_idx]

            # Normalize HRTFs using fitted normalizer
            valid_hrtfs_norm = model.hrtf_normalizer.transform(valid_hrtfs)

            vae_optimizer.zero_grad()
            recon, mu, logvar = model.hrtf_vae(valid_hrtfs_norm)

            loss = vae_loss(recon, valid_hrtfs_norm, mu, logvar, beta=1.0)
            loss.backward()
            vae_optimizer.step()

            vae_loss_accum += loss.item()
            batch_count += 1

        vae_scheduler.step()

        if (epoch + 1) % 25 == 0:
            current_lr = vae_scheduler.get_last_lr()[0]
            print(f"VAE Epoch {epoch+1}/{vae_epochs}, Loss: {vae_loss_accum/max(batch_count, 1):.4f}, LR: {current_lr:.2e}")

    # Step 2: Joint training of AE and DNN
    print("\nStep 2: Joint training of AE and DNN...")
    print(f"  Joint epochs: {joint_epochs}, lr: {joint_lr}")

    # Freeze VAE during joint training
    for param in model.hrtf_vae.parameters():
        param.requires_grad = False

    joint_optimizer = optim.Adam(
        list(model.input_ae.parameters()) + list(model.mapper.parameters()),
        lr=joint_lr
    )

    joint_scheduler = optim.lr_scheduler.StepLR(joint_optimizer, step_size=50, gamma=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 100

    for epoch in range(joint_epochs):
        model.train()
        train_loss_accum = 0.0
        batch_count = 0

        for batch in train_loader:
            locations, hrtfs, masks, anthropometry, _, _ = batch
            B, N, _ = locations.shape

            locations = locations.reshape(-1, 2).to(device)
            hrtfs = hrtfs.reshape(-1, model.num_freq).to(device)
            masks = masks.to(device)
            anthropometry = anthropometry.unsqueeze(1).expand(B, N, -1)
            anthropometry = anthropometry.reshape(-1, anthropometry.shape[-1]).to(device)

            valid_idx = masks.reshape(-1, 1).squeeze() > 0
            if valid_idx.sum() == 0:
                continue

            valid_locations = locations[valid_idx]
            valid_hrtfs = hrtfs[valid_idx]
            valid_anthro = anthropometry[valid_idx]

            joint_optimizer.zero_grad()

            # Forward pass
            outputs = model(valid_locations, valid_anthro, valid_hrtfs, mode='train')

            # Calculate losses
            # DNN loss: MSE between predicted and target latent features
            with torch.no_grad():
                valid_hrtfs_norm = model.hrtf_normalizer.transform(valid_hrtfs)
                target_mu, _ = model.hrtf_vae.encode(valid_hrtfs_norm)

            dnn_loss = F.mse_loss(outputs['predicted_latent'], target_mu)

            # AE reconstruction loss
            coords = model.angles_to_coordinates(valid_locations[:, 0], valid_locations[:, 1])
            anthro_norm = model.anthro_normalizer.transform(valid_anthro)
            coords_norm = model.coord_normalizer.transform(coords)
            input_vector = torch.cat([anthro_norm, coords_norm], dim=-1)

            ae_loss = F.mse_loss(outputs['input_recon'], input_vector)

            # Combined loss with weights from paper (0.7 DNN, 0.3 AE)
            total_loss = 0.7 * dnn_loss + 0.3 * ae_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            joint_optimizer.step()

            train_loss_accum += total_loss.item()
            batch_count += 1

        avg_train_loss = train_loss_accum / max(batch_count, 1)
        joint_scheduler.step()

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss_accum = 0.0
            val_lsd_accum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch in val_loader:
                    locations, hrtfs, masks, anthropometry, _, _ = batch
                    B, N, _ = locations.shape

                    locations = locations.reshape(-1, 2).to(device)
                    hrtfs = hrtfs.reshape(-1, model.num_freq).to(device)
                    masks = masks.to(device)
                    anthropometry = anthropometry.unsqueeze(1).expand(B, N, -1)
                    anthropometry = anthropometry.reshape(-1, anthropometry.shape[-1]).to(device)

                    valid_idx = masks.reshape(-1, 1).squeeze() > 0
                    if valid_idx.sum() == 0:
                        continue

                    valid_locations = locations[valid_idx]
                    valid_hrtfs = hrtfs[valid_idx]
                    valid_anthro = anthropometry[valid_idx]

                    # Get predictions (automatically denormalized in predict mode)
                    predicted_hrtfs = model(valid_locations, valid_anthro, mode='predict')

                    # Calculate loss in normalized space
                    predicted_hrtfs_norm = model.hrtf_normalizer.transform(predicted_hrtfs)
                    valid_hrtfs_norm = model.hrtf_normalizer.transform(valid_hrtfs)
                    val_loss = F.mse_loss(predicted_hrtfs_norm, valid_hrtfs_norm)
                    val_loss_accum += val_loss.item() * valid_idx.sum().item()

                    # Calculate LSD in original scale
                    # Assuming HRTFs are in log scale already
                    for i in range(valid_hrtfs.shape[0]):
                        lsd = calculate_lsd(predicted_hrtfs[i], valid_hrtfs[i], in_db=True)
                        val_lsd_accum += lsd.item()
                        val_count += 1

            avg_val_loss = val_loss_accum / max(val_count, 1)
            avg_val_lsd = val_lsd_accum / max(val_count, 1)

            current_lr = joint_optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{joint_epochs}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val LSD: {avg_val_lsd:.4f} dB, LR: {current_lr:.2e}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"  --> Best validation loss improved to {best_val_loss:.4f}")

                # Save best model
                best_model_path = os.path.join("/export/mkrishn9/hrtf_field/dual_ae_benchmark", "best_model.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'val_lsd': avg_val_lsd,
                    'normalizers': {
                        'hrtf_mean': model.hrtf_normalizer.mean,
                        'hrtf_std': model.hrtf_normalizer.std,
                        'anthro_mean': model.anthro_normalizer.mean,
                        'anthro_std': model.anthro_normalizer.std,
                        'coord_mean': model.coord_normalizer.mean,
                        'coord_std': model.coord_normalizer.std
                    }
                }, best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{joint_epochs}, Train Loss: {avg_train_loss:.4f}")

    # Unfreeze VAE after training
    for param in model.hrtf_vae.parameters():
        param.requires_grad = True

    return model


def run_benchmark(data_path, save_dir, anthro_dim=10, num_freq=92, batch_size=4):
    """
    Run the dual-autoencoder benchmark following paper specifications.
    """
    from torch.utils.data import Subset
    import random
    import re
    from sklearn.model_selection import GroupShuffleSplit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset - HUTUBS only as per paper
    dataset = MergedHRTFDataset1(
        dataset_names=["hutubs"],
        preprocessed_dir=data_path,
        augment=False,
        scale="log"  # Important: log scale for proper LSD calculation
    )

    # Subject-aware splitting
    print("Creating subject-aware train/val split...")

    subject_ids = []
    for i, file_path in enumerate(dataset.file_registry):
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if match:
            dataset_name, file_num = match.group(1), int(match.group(2))
            subject_id = file_num // 2
            subject_ids.append(subject_id)
        else:
            subject_ids.append(i)

    indices = np.arange(len(dataset))
    groups = np.array(subject_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(indices, groups=groups))

    train_indices = indices[train_idx].tolist()
    val_indices = indices[val_idx].tolist()

    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    print(f"Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
    print(f"No subject overlap: {len(train_subjects.intersection(val_subjects)) == 0}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=merged_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=merged_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = DualAutoEncoderHRTF(
        anthro_dim=anthro_dim,
        num_freq=num_freq,
        latent_dim=16
    ).to(device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    model = train_dual_autoencoder(
        model,
        train_loader,
        val_loader,
        device,
        vae_epochs=100,
        joint_epochs=200,
        vae_lr=1e-3,
        joint_lr=1e-3
    )

    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dual_ae_hrtf_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'anthro_dim': anthro_dim,
        'num_freq': num_freq,
        'latent_dim': 16,
        'normalizers': {
            'hrtf_mean': model.hrtf_normalizer.mean,
            'hrtf_std': model.hrtf_normalizer.std,
            'anthro_mean': model.anthro_normalizer.mean,
            'anthro_std': model.anthro_normalizer.std,
            'coord_mean': model.coord_normalizer.mean,
            'coord_std': model.coord_normalizer.std
        }
    }, save_path)
    print(f"Final model saved to {save_path}")

    return model


if __name__ == "__main__":
    DATA_PATH = "/export/mkrishn9/hrtf_field/preprocessed_hrirs_common"
    SAVE_DIR = "/export/mkrishn9/hrtf_field/dual_ae_benchmark"

    model = run_benchmark(
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        anthro_dim=10,
        num_freq=92,
        batch_size=1
    )