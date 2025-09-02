import SOFAdatasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import pickle as pkl
import os
import glob
import pandas as pd
import scipy.signal
import random
import re
from collections import defaultdict
import librosa

### What is marked "old" is from the HRTFField paper. The new versions of HRTFDataset work for a single dataset. Merged works for all.
### HRTFDataset is a bit outdated. Please use Merged.


def get_itd_from_anthro(anthro_vector, source_azimuth, head_width_idx=0, speed_of_sound=343.0):
    """Calculates ITD based on a spherical head model."""
    head_radius = (anthro_vector[head_width_idx] / 100) / 2
    azimuth_rad = np.deg2rad(source_azimuth)
    # I think this is called Woodworth/Schlosberg formula
    itd = (head_radius / speed_of_sound) * (azimuth_rad + np.sin(azimuth_rad))
    return itd

def apply_itd_shift(hrir, shift_in_seconds, sample_rate):
    """Applies a time shift to an HRIR via rolling."""
    shift_in_samples = int(round(shift_in_seconds * sample_rate))
    #### A circular shift. Maybe could use phase shifts in FFT.
    return np.roll(hrir, shift_in_samples, axis=-1)


class HRTFDataset_Norm(Dataset):
    """
    An all-in-one class to handle multiple HRTF datasets.

    - Loads ALL preprocessed files into memory at initialization.
    - Pools anthropometry from ALL subjects to create a unified statistical model.
    - Performs on-the-fly data augmentation using these global statistics.
    - Decomposes to DTF + CTF
    -exploring another regularization technique.
    """
    def __init__(self, dataset_names, preprocessed_dir,
                 augment=False, aug_prob=0.6,
                 freq="all", scale="linear", norm_way=2):

        print("Initializing MergedHRTFDataset...")
        self.dataset_names = dataset_names
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.aug_prob = aug_prob
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.sample_rate = 44100

        # --- 1. Scan for all available preprocessed files ---
        self.file_registry = []
        for name in self.dataset_names:
            found_files = glob.glob(os.path.join(self.preprocessed_dir, f"{name}_*.pkl"))
            self.file_registry.extend(found_files)

        self.file_registry.sort()

        if not self.file_registry:
            raise FileNotFoundError(f"No preprocessed .pkl files found in {preprocessed_dir} for datasets: {dataset_names}")

        print(f"Found {len(self.file_registry)} total preprocessed files.")

        # --- 2. Pre-load ALL data and Pool anthropometry ---
        print("Pre-loading all .pkl files into memory...")
        self.data_cache = [] # MODIFIED: New list to store loaded data
        all_anthros_for_stats = [] # MODIFIED: List for calculating stats
        hrtfs_by_dataset = defaultdict(list)


        for file_path in self.file_registry:
            with open(file_path, 'rb') as handle:
                # Load the data from the pkl file
                location_np, hrir, anthro_np = pkl.load(handle)

                # Derive dataset name from filename
                base_name = os.path.basename(file_path)
                #dataset_name = base_name.split('_')[0]
                parts = base_name.replace('.pkl', '').split('_')
                dataset_name = parts[0]

                tf = np.abs(np.fft.fft(hrir, n=256))
                hrtf_np = tf[:, 1:93]  # Your frequency bins of interest
                hrtfs_by_dataset[dataset_name].append(hrtf_np)



                # Store the loaded data in a dictionary for clarity
                data_item = {
                    "location": location_np,
                    "hrir": hrir,
                    "anthro": anthro_np,
                    "dataset_name": dataset_name
                }
                self.data_cache.append(data_item)


                # Collect anthropometry if augmentation is enabled
                if self.augment:
                    all_anthros_for_stats.append(torch.from_numpy(anthro_np))

        # --- 3. Build a global statistical model from pooled data ---
        if self.augment:
            print("Augmentation enabled. Calculating global statistics from pooled anthropometry...")
            all_anthros_stacked = torch.stack(all_anthros_for_stats, dim=0)

            self.d3_mean = torch.mean(all_anthros_stacked[:, 11]).item() # d3 (width ear scale) at index 11
            self.d3_std = torch.std(all_anthros_stacked[:, 11]).item()
            print(f"Global d3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")
        self.average_hrtfs_by_dataset = {}
        for dataset_name, hrtf_list in hrtfs_by_dataset.items():
            stacked = np.stack(hrtf_list, axis=0)
            self.average_hrtfs_by_dataset[dataset_name] = np.mean(stacked, axis=0)
            print(f"Dataset {dataset_name}: computed averages from {len(hrtf_list)} subjects")

    def __len__(self):
        return len(self.data_cache) # MODIFIED: Length is based on the cache

    def set_augmentation(self, augment: bool):
        """Public method to enable or disable augmentation for this dataset instance."""
        if self.augment != augment:
            self.augment = augment
            status = "ENABLED" if augment else "DISABLED"
            print(f"--> MergedHRTFDataset augmentation is now {status}.")

    def _augment_hrir(self, hrir, anthro,locations):
        """ Applies augmentation using the globally calculated statistics. """
        original_d3_val = anthro[11]

        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)

        scale_factor = target_d3_val / original_d3_val if original_d3_val != 0 else 1.0
        augmented_anthro = anthro.copy()

        distance_indices = list(range(9, 17))
        for i in distance_indices:
            augmented_anthro[i] *= scale_factor

        num_samples_orig = hrir.shape[-1]
        num_samples_new = int(np.round(num_samples_orig * scale_factor))
        hrir = scipy.signal.resample(hrir, num_samples_new, axis=-1)

        original_itds = np.array([get_itd_from_anthro(anthro, loc[0]) for loc in locations])
        target_itds = np.array([get_itd_from_anthro(augmented_anthro, loc[0]) for loc in locations])
        itd_shifts_seconds = target_itds - original_itds

        for i in range(hrir.shape[0]):
            hrir_slice = hrir[i]
            shift_for_this_location = itd_shifts_seconds[i]
            hrir[i] = apply_itd_shift(hrir_slice, shift_for_this_location, self.sample_rate)

        return hrir, augmented_anthro


    def _process_hrtf(self, hrir,dataset_name):
        """ Converts HRIR to HRTF and applies normalization/scaling. """
        tf = np.abs(np.fft.fft(hrir, n=256))
        hrtf_np = tf[:, 1:93]

        if hasattr(self, 'average_hrtfs_by_dataset') and dataset_name in self.average_hrtfs_by_dataset:
            hrtf_np = hrtf_np / self.average_hrtfs_by_dataset[dataset_name]

        if self.scale == "linear":
            hrtf_np = hrtf_np
        if self.scale == "log":
            hrtf_np = 20 * np.log10(hrtf_np + 1e-9)
            # max_val = 15
            # min_val =  -85
            # hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1

        return hrtf_np

    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around for model interpolation. """
        index2 = np.where(locs_np[:, 0] < 30)[0]
        new_locs2 = locs_np[index2].copy()
        new_locs2[:, 0] += 360
        new_hrtfs2 = hrtfs_np[index2].copy()
        index1 = np.where(locs_np[:, 0] > 330)[0]
        new_locs1 = locs_np[index1].copy()
        new_locs1[:, 0] -= 360
        new_hrtfs1 = hrtfs_np[index1].copy()
        extended_locs = np.concatenate((new_locs1, locs_np, new_locs2), axis=0)
        extended_hrtfs = np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)
        return extended_locs, extended_hrtfs

    def __getitem__(self, idx):
        # MODIFIED: Retrieve data from the in-memory cache instead of reading a file
        cached_item = self.data_cache[idx]
        location_np = cached_item["location"]
        hrir = cached_item["hrir"]
        anthro_np = cached_item["anthro"]
        dataset_name = cached_item["dataset_name"]


        # The rest of the processing logic remains the same
        if self.augment and random.random() < self.aug_prob:
            hrir, anthro_np = self._augment_hrir(hrir, anthro_np, location_np)

        hrtf_np = self._process_hrtf(hrir, dataset_name)
        location_extended, hrtf_extended = self.extend_locations(location_np, hrtf_np)


        location_tensor = torch.from_numpy(location_extended.astype(np.float32))
        hrtf_tensor = torch.from_numpy(hrtf_extended.astype(np.float32))
        anthro_tensor = torch.from_numpy(anthro_np.astype(np.float32))
        ear_tensor = anthro_tensor[9:]


        return location_tensor, hrtf_tensor, ear_tensor, dataset_name




class CTF_DTF_Dataset(Dataset):
    """
    An all-in-one class to handle multiple HRTF datasets.

    - Loads ALL preprocessed files into memory at initialization.
    - Pools anthropometry from ALL subjects to create a unified statistical model.
    - Performs on-the-fly data augmentation using these global statistics.
    - Decomposes to DTF + CTF
    -exploring another regularization technique.
    """
    def __init__(self, dataset_names, preprocessed_dir,
                 augment=False, aug_prob=0.6,
                 freq="all", scale="linear", norm_way=3):

        print("Initializing MergedHRTFDataset...")
        self.dataset_names = dataset_names
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.aug_prob = aug_prob
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.sample_rate = 44100

        # --- 1. Scan for all available preprocessed files ---
        self.file_registry = []
        for name in self.dataset_names:
            found_files = glob.glob(os.path.join(self.preprocessed_dir, f"{name}_*.pkl"))
            self.file_registry.extend(found_files)

        self.file_registry.sort()

        if not self.file_registry:
            raise FileNotFoundError(f"No preprocessed .pkl files found in {preprocessed_dir} for datasets: {dataset_names}")

        print(f"Found {len(self.file_registry)} total preprocessed files.")

        # --- 2. Pre-load ALL data and Pool anthropometry ---
        print("Pre-loading all .pkl files into memory...")
        self.data_cache = [] # MODIFIED: New list to store loaded data
        all_anthros_for_stats = [] # MODIFIED: List for calculating stats
        hrtfs_by_dataset = defaultdict(list)


        for file_path in self.file_registry:
            with open(file_path, 'rb') as handle:
                # Load the data from the pkl file
                location_np, hrir, anthro_np = pkl.load(handle)

                # Derive dataset name from filename
                base_name = os.path.basename(file_path)
                #dataset_name = base_name.split('_')[0]
                parts = base_name.replace('.pkl', '').split('_')
                dataset_name = parts[0]
                file_id = int(parts[1]) # The numeric ID (e.g., 3)




                # Store the loaded data in a dictionary for clarity
                data_item = {
                    "location": location_np,
                    "hrir": hrir,
                    "anthro": anthro_np,
                    "dataset_name": dataset_name
                }
                self.data_cache.append(data_item)


                # Collect anthropometry if augmentation is enabled
                if self.augment:
                    all_anthros_for_stats.append(torch.from_numpy(anthro_np))

        # --- 3. Build a global statistical model from pooled data ---
        if self.augment:
            print("Augmentation enabled. Calculating global statistics from pooled anthropometry...")
            all_anthros_stacked = torch.stack(all_anthros_for_stats, dim=0)

            self.d3_mean = torch.mean(all_anthros_stacked[:, 11]).item() # d3 (width ear scale) at index 11
            self.d3_std = torch.std(all_anthros_stacked[:, 11]).item()
            print(f"Global d3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")

    def __len__(self):
        return len(self.data_cache) # MODIFIED: Length is based on the cache

    def set_augmentation(self, augment: bool):
        """Public method to enable or disable augmentation for this dataset instance."""
        if self.augment != augment:
            self.augment = augment
            status = "ENABLED" if augment else "DISABLED"
            print(f"--> MergedHRTFDataset augmentation is now {status}.")

    def _augment_hrir(self, hrir, anthro,locations):
        """ Applies augmentation using the globally calculated statistics. """
        original_d3_val = anthro[11]

        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)

        scale_factor = target_d3_val / original_d3_val if original_d3_val != 0 else 1.0
        augmented_anthro = anthro.copy()

        distance_indices = list(range(9, 17))
        for i in distance_indices:
            augmented_anthro[i] *= scale_factor

        num_samples_orig = hrir.shape[-1]
        num_samples_new = int(np.round(num_samples_orig * scale_factor))
        hrir = scipy.signal.resample(hrir, num_samples_new, axis=-1)

        original_itds = np.array([get_itd_from_anthro(anthro, loc[0]) for loc in locations])
        target_itds = np.array([get_itd_from_anthro(augmented_anthro, loc[0]) for loc in locations])
        itd_shifts_seconds = target_itds - original_itds

        for i in range(hrir.shape[0]):
            hrir_slice = hrir[i]
            shift_for_this_location = itd_shifts_seconds[i]
            hrir[i] = apply_itd_shift(hrir_slice, shift_for_this_location, self.sample_rate)

        return hrir, augmented_anthro

    def _process_hrtf(self, hrir, location):
            """ Converts HRIR to DTF via HRTF and CTF computation. """
            tf = np.abs(np.fft.fft(hrir, n=256))
            hrtf_np = tf[:, 1:93]  # Your frequency bins of interest

            ctf = self._compute_ctf(hrtf_np, location)

            dtf_np = hrtf_np / (ctf + 1e-10)

            dtf_np = self._normalize_dtf(dtf_np, location)

            if self.scale == "log":
                ctf = 20 * np.log10(ctf + 1e-9)


            return dtf_np, ctf

    def _compute_ctf(self, hrtf_np, location):
        """
        Compute CTF using diffuse-field average method.
        CTF represents the direction-independent component.
        """
        # Method 1: Simple RMS average across all directions
        # ctf = np.sqrt(np.mean(hrtf_np**2, axis=0, keepdims=True))

        # Method 2: Weighted average based on solid angle (more accurate)
        # Weight by elevation to account for sphere sampling density
        elevation_weights = np.cos(np.radians(location[:, 1]))
        elevation_weights = elevation_weights / np.sum(elevation_weights)

        # Weighted RMS average
        weighted_hrtf = hrtf_np * elevation_weights[:, np.newaxis]
        ctf = np.sqrt(np.sum(weighted_hrtf**2, axis=0, keepdims=True))

        # Method 3: Geometric mean (alternative)
        # ctf = np.exp(np.mean(np.log(hrtf_np + 1e-10), axis=0, keepdims=True))

        return ctf

    def _normalize_dtf(self, dtf_np, location):
        """
        Normalize DTF - different strategies than HRTF since DTF is already normalized by CTF
        """
        if self.norm_way == 0:
            # Peak normalization
            max_val = np.max(dtf_np)
            if max_val > 1e-9:
                dtf_np = dtf_np / max_val

        elif self.norm_way == 1:
            # Top 5% normalization
            mag_flatten = dtf_np.flatten()
            max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
            if max_mag > 1e-9:
                dtf_np = dtf_np / max_mag

        elif self.norm_way == 2:
            # Energy normalization at equator (still valid for DTF)
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            dtf_equator = dtf_np[equator_index]
            equator_azi = location[equator_index, 0][0]
            new_equator_index = np.argsort(equator_azi)
            new_equator_azi = equator_azi[new_equator_index]
            new_equator_tf = dtf_equator[new_equator_index]

            total_energy = 0
            for x in range(len(new_equator_index)):
                if x == 0:
                    d_azi = 360 - new_equator_azi[-1]
                else:
                    d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                total_energy += np.square(new_equator_tf[x]).mean() * d_azi
            if total_energy > 0:
                dtf_np = dtf_np / np.sqrt(total_energy / 360)
        elif self.norm_way == 3:
            dtf_np = dtf_np

        # Apply scaling (linear or log)
        if self.scale == "log":
            dtf_np = 20 * np.log10(dtf_np + 1e-9)
            # # DTF typically has smaller dynamic range than HRTF
            # max_val = 10   # Adjust based on your data
            # min_val = -60  # Adjust based on your data
            # dtf_np = 2 * (dtf_np - min_val) / (max_val - min_val) - 1

        return dtf_np
    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around for model interpolation. """
        index2 = np.where(locs_np[:, 0] < 30)[0]
        new_locs2 = locs_np[index2].copy()
        new_locs2[:, 0] += 360
        new_hrtfs2 = hrtfs_np[index2].copy()
        index1 = np.where(locs_np[:, 0] > 330)[0]
        new_locs1 = locs_np[index1].copy()
        new_locs1[:, 0] -= 360
        new_hrtfs1 = hrtfs_np[index1].copy()
        extended_locs = np.concatenate((new_locs1, locs_np, new_locs2), axis=0)
        extended_hrtfs = np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)
        return extended_locs, extended_hrtfs

    def __getitem__(self, idx):
        # MODIFIED: Retrieve data from the in-memory cache instead of reading a file
        cached_item = self.data_cache[idx]
        location_np = cached_item["location"]
        hrir = cached_item["hrir"]
        anthro_np = cached_item["anthro"]
        dataset_name = cached_item["dataset_name"]



        # The rest of the processing logic remains the same
        if self.augment and random.random() < self.aug_prob:
            hrir, anthro_np = self._augment_hrir(hrir, anthro_np, location_np)

        hrtf_np, ctf = self._process_hrtf(hrir, location_np)
        location_extended, hrtf_extended = self.extend_locations(location_np, hrtf_np)

        location_tensor = torch.from_numpy(location_extended.astype(np.float32))
        hrtf_tensor = torch.from_numpy(hrtf_extended.astype(np.float32))
        ctf_tensor = torch.from_numpy(ctf.astype(np.float32))
        anthro_tensor = torch.from_numpy(anthro_np.astype(np.float32))
        ear_tensor = anthro_tensor[9:]


        return location_tensor, hrtf_tensor, ear_tensor, dataset_name

class MergedHRTFDataset(Dataset):
    """
    An all-in-one class to handle multiple HRTF datasets.

    - Scans for all preprocessed files from specified datasets.
    - Pools anthropometry from ALL subjects to create a unified statistical model.
    - Performs on-the-fly data augmentation using these global statistics.
    """
    def __init__(self, dataset_names, preprocessed_dir,
                 augment=False, aug_prob=0.6,
                 freq="all", scale="linear", norm_way=2):

        print("Initializing MergedHRTFDataset...")
        self.dataset_names = dataset_names
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.aug_prob = aug_prob
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.sample_rate = 44100

        # --- 1. Scan for all available preprocessed files ---
        self.file_registry = []
        for name in self.dataset_names:
            found_files = glob.glob(os.path.join(self.preprocessed_dir, f"{name}_*.pkl"))
            self.file_registry.extend(found_files)

        self.file_registry.sort()

        if not self.file_registry:
            raise FileNotFoundError(f"No preprocessed .pkl files found in {preprocessed_dir} for datasets: {dataset_names}")

        print(f"Found {len(self.file_registry)} total preprocessed files.")

        # --- 2. Pool ALL anthropometry to build a global statistical model ---
        if self.augment:
            print("Augmentation enabled. Pooling all anthropometry to calculate statistics...")
            all_anthros = []
            for file_path in self.file_registry:
                with open(file_path, 'rb') as handle:
                    _, _, anthro_np = pkl.load(handle)
                    all_anthros.append(torch.from_numpy(anthro_np))

            all_anthros_stacked = torch.stack(all_anthros, dim=0)


            self.d3_mean = torch.mean(all_anthros_stacked[:, 11]).item() # d3 (width ear scale) at index 15 ###HARDCODED
            self.d3_std = torch.std(all_anthros_stacked[:, 11]).item()  ###HARDCODED d3
            print(f"Global d3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")

    def __len__(self):
        return len(self.file_registry)

    def set_augmentation(self, augment: bool):
        """Public method to enable or disable augmentation for this dataset instance."""
        if self.augment != augment:
            self.augment = augment
            status = "ENABLED" if augment else "DISABLED"
            print(f"--> MergedHRTFDataset augmentation is now {status}.")

    def _augment_hrir(self, hrir, anthro,locations):
        """ Applies augmentation using the globally calculated statistics. """
        original_d3_val = anthro[11]  ### HARDCODED

        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)

        scale_factor = target_d3_val / original_d3_val if original_d3_val != 0 else 1.0
        augmented_anthro = anthro.copy()

        # Scale all distance-based ear features
        distance_indices = list(range(9, 17)) #list(range(19)) ### HARDCODED 9-17 is ear features in anthropometry, 1-8 is head/torso measurements.
        for i in distance_indices:
            augmented_anthro[i] *= scale_factor

        num_samples_orig = hrir.shape[-1]
        num_samples_new = int(np.round(num_samples_orig * scale_factor))
        hrir = scipy.signal.resample(hrir, num_samples_new, axis=-1)

        # 3. Calculate the required ITD shift for all location.
        original_itds = np.array([get_itd_from_anthro(anthro, loc[0]) for loc in locations])
        target_itds = np.array([get_itd_from_anthro(augmented_anthro, loc[0]) for loc in locations])
        itd_shifts_seconds = target_itds - original_itds

        # 4. Loop through each location and apply its specific shift.
        # The hrir variable has shape (num_locations, num_samples).
        for i in range(hrir.shape[0]):
            hrir_slice = hrir[i]
            shift_for_this_location = itd_shifts_seconds[i]
            hrir[i] = apply_itd_shift(hrir_slice, shift_for_this_location, self.sample_rate)


        return hrir, augmented_anthro

    def _process_hrtf(self, hrir,location):
        """ Converts HRIR to HRTF and applies normalization/scaling. """
        tf = np.abs(np.fft.fft(hrir, n=256))
        hrtf_np = tf[:, 1:93]
        compute_dtf = True

        if compute_dtf:
            ctf = self._compute_ctf(hrtf_np, location)
            dtf_np = hrtf_np / (ctf + 1e-10)
            if self.scale == "log":
                ctf = 20 * np.log10(ctf + 1e-9)
                dtf_np = 20 * np.log10(dtf_np + 1e-9)
            hrtf_np = dtf_np
        else:
            if self.norm_way == 0:
                max_val = np.max((hrtf_np))
                if max_val > 1e-9: hrtf_np = hrtf_np / max_val
            ## second way is to divide by top 5% top value
            elif self.norm_way == 1:
                mag_flatten = hrtf_np.flatten()
                max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
                hrtf_np = hrtf_np / max_mag
            ## third way is to compute total energy of the equator
            elif self.norm_way == 2:
                equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
                hrtf_equator = hrtf_np[equator_index]
                equator_azi = location[equator_index, 0][0]
                new_equator_index = np.argsort(equator_azi)
                new_equator_azi = equator_azi[new_equator_index]
                new_equator_tf = hrtf_equator[new_equator_index]

                total_energy = 0
                for x in range(len(new_equator_index)):
                    if x == 0:
                        d_azi = 360 - new_equator_azi[-1]
                        # d_azi = new_equator_azi[1] - new_equator_azi[0]
                    else:
                        d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                    total_energy += np.square(new_equator_tf[x]).mean() * d_azi
                hrtf_np = hrtf_np / np.sqrt(total_energy / 360)
                breakpoint()
            elif self.norm_way == 4:
                if self.scale == 'log':
                    max_val = 20 #np.max(hrtf_np)
                    min_val =  -100 #np.min(hrtf_np)
                    hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1

            #     # print(np.sqrt(total_energy / 360))
            # ## fourth way is to normalize on common locations
            # ## [(0.0, 0.0), (180.0, 0.0), (210.0, 0.0), (330.0, 0.0), (30.0, 0.0), (150.0, 0.0)]
            # elif norm_way == 3:
            #     common_index = np.where(np.logical_and(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0),
            #                                            np.array(
            #                                                [round(x) in [0, 180, 210, 330, 30, 150] for x in location[:, 0]])))
            #     tf_common = tf[common_index]
            #     mean_energy = np.sqrt(np.square(tf_common).mean())
            #     # print(mean_energy)
            #     tf = tf / mean_energy

            if self.scale == "linear":
                hrtf_np = hrtf_np
            if self.scale == "log":
                hrtf_np = 20 * np.log10(hrtf_np + 1e-9)
                # max_val = 15 #np.max(hrtf_np)                           # Max HRTF value after normalizing based on equation energy (norm_way=2)
                # min_val =  -85 #np.min(hrtf_np)                         # Min HRTF value after normalizing based on equation energy (norm_way=2)
                # hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1     ## Normalize to (-1,1)



        return hrtf_np

    def _compute_ctf(self, hrtf_np, location):
        """
        Compute CTF using diffuse-field average method.
        CTF represents the direction-independent component.
        """
        # Method 1: Simple RMS average across all directions
        # ctf = np.sqrt(np.mean(hrtf_np**2, axis=0, keepdims=True))

        # Method 2: Weighted average based on solid angle (more accurate)
        # Weight by elevation to account for sphere sampling density
        elevation_weights = np.cos(np.radians(location[:, 1]))
        elevation_weights = elevation_weights / np.sum(elevation_weights)

        # Weighted RMS average
        weighted_hrtf = hrtf_np * elevation_weights[:, np.newaxis]
        ctf = np.sqrt(np.sum(weighted_hrtf**2, axis=0, keepdims=True))

        # Method 3: Geometric mean (alternative)
        # ctf = np.exp(np.mean(np.log(hrtf_np + 1e-10), axis=0, keepdims=True))

        return ctf


    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around for model interpolation. """
        index2 = np.where(locs_np[:, 0] < 30)[0]
        new_locs2 = locs_np[index2].copy()
        new_locs2[:, 0] += 360
        new_hrtfs2 = hrtfs_np[index2].copy()
        index1 = np.where(locs_np[:, 0] > 330)[0]
        new_locs1 = locs_np[index1].copy()
        new_locs1[:, 0] -= 360
        new_hrtfs1 = hrtfs_np[index1].copy()
        extended_locs = np.concatenate((new_locs1, locs_np, new_locs2), axis=0)
        extended_hrtfs = np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)
        return extended_locs, extended_hrtfs

    def __getitem__(self, idx):
        file_path = self.file_registry[idx]
        with open(file_path, 'rb') as handle:
            location_np, hrir, anthro_np = pkl.load(handle)

        if self.augment and random.random() < self.aug_prob:
            hrir, anthro_np = self._augment_hrir(hrir, anthro_np,location_np)

        hrtf_np = self._process_hrtf(hrir,location_np)
        location_extended, hrtf_extended = self.extend_locations(location_np, hrtf_np)


        base_name = os.path.basename(file_path)
        dataset_name = base_name.split('_')[0]

        # print(np.min(hrtf_np))
        # print(np.max(hrtf_np))
        # print(dataset_name)

        location_tensor = torch.from_numpy(location_extended.astype(np.float32))
        hrtf_tensor = torch.from_numpy(hrtf_extended.astype(np.float32))
        anthro_tensor = torch.from_numpy(anthro_np.astype(np.float32))
        ear_tensor = anthro_tensor[9:]

        return location_tensor, hrtf_tensor, ear_tensor, dataset_name


class MergedHRTFDataset1(Dataset):
    """
    An all-in-one class to handle multiple HRTF datasets.

    - Loads ALL preprocessed files into memory at initialization.
    - Pools anthropometry from ALL subjects to create a unified statistical model.
    - Performs on-the-fly data augmentation using these global statistics.
    """
    def __init__(self, dataset_names, preprocessed_dir,
                 augment=False, aug_prob=0.6,
                 freq="all", scale="linear", norm_way=2):

        print("Initializing MergedHRTFDataset...")
        self.dataset_names = dataset_names
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.aug_prob = aug_prob
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.sample_rate = 44100
        self.fft_size = 256
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}





        # --- 1. Scan for all available preprocessed files ---
        self.file_registry = []
        for name in self.dataset_names:
            found_files = glob.glob(os.path.join(self.preprocessed_dir, f"{name}_*.pkl"))
            self.file_registry.extend(found_files)

        self.file_registry.sort()

        if not self.file_registry:
            raise FileNotFoundError(f"No preprocessed .pkl files found in {preprocessed_dir} for datasets: {dataset_names}")

        print(f"Found {len(self.file_registry)} total preprocessed files.")

        # --- 2. Pre-load ALL data and Pool anthropometry ---
        print("Pre-loading all .pkl files into memory...")
        self.data_cache = [] # MODIFIED: New list to store loaded data
        all_anthros_for_stats = [] # MODIFIED: List for calculating stats
        self.zero_freq_lookups = {}

        for file_path in self.file_registry:
            with open(file_path, 'rb') as handle:
                # Load the data from the pkl file
                location_np, hrir, anthro_np = pkl.load(handle)

                # Derive dataset name from filename
                base_name = os.path.basename(file_path)
                #dataset_name = base_name.split('_')[0]
                parts = base_name.replace('.pkl', '').split('_')
                dataset_name = parts[0]
                file_id = int(parts[1]) # The numeric ID (e.g., 3)

                # dataset_obj = getattr(SOFAdatasets, dataset_dict[dataset_name])()


                # if file_id % 2 == 0: # Even ID is a left ear
                #     ear = 'left'
                # else: # Odd ID is a right ear
                #     ear = 'right'

                # subject_idx = (file_id // 2)
                # subject_id = int(dataset_obj._get_subject_ID(subject_idx))


                # subject_key = (dataset_name, subject_id,ear)
                # if subject_key not in self.zero_freq_lookups:
                #     csv_path = os.path.join(self.preprocessed_dir, f"0freq_{dataset_name}/{dataset_name}_{subject_id:03d}_{ear}_0freq.csv")
                #     if os.path.exists(csv_path):
                #         df_freq = pd.read_csv(csv_path)
                #         # Directly extract frequency columns to a NumPy array.
                #         # This assumes the Nth row in the CSV corresponds to the Nth location.
                #         zero_freq_data = df_freq.filter(like='Freq').values.astype(np.float32)
                #         #nan_mask = np.isnan(zero_freq_data)
                #         fft_bins = np.round(zero_freq_data*1000 * self.fft_size / self.sample_rate)
                #         # hrtf_indices = np.clip(fft_bins - 1, 0, 91)
                #         # hrtf_indices[nan_mask] = -1
                #         condition = (fft_bins >= 0) & (fft_bins <= 91)
                #         hrtf_indices = np.where(condition, fft_bins, -1)
                #         self.zero_freq_lookups[subject_key] = hrtf_indices.astype(np.int64)
                #     else:
                #         print(f"Warning: Missing frequency file for {subject_key}: {csv_path}")
                #         # Store None as a placeholder to be handled in __getitem__
                #         self.zero_freq_lookups[subject_key] = None

                # Store the loaded data in a dictionary for clarity
                data_item = {
                    "location": location_np,
                    "hrir": hrir,
                    "anthro": anthro_np,
                    "dataset_name": dataset_name,
                    # "subject_id": subject_id,
                    # "ear": ear
                    }
                self.data_cache.append(data_item) # MODIFIED: Append data to cache

                # Collect anthropometry if augmentation is enabled
                if self.augment:
                    all_anthros_for_stats.append(torch.from_numpy(anthro_np))

        print(f"Successfully loaded {len(self.data_cache)} files into memory.")

        # --- 3. Build a global statistical model from pooled data ---
        if self.augment:
            print("Augmentation enabled. Calculating global statistics from pooled anthropometry...")
            all_anthros_stacked = torch.stack(all_anthros_for_stats, dim=0)

            self.d3_mean = torch.mean(all_anthros_stacked[:, 11]).item() # d3 (width ear scale) at index 11
            self.d3_std = torch.std(all_anthros_stacked[:, 11]).item()
            print(f"Global d3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")

    def __len__(self):
        return len(self.data_cache) # MODIFIED: Length is based on the cache

    def set_augmentation(self, augment: bool):
        """Public method to enable or disable augmentation for this dataset instance."""
        if self.augment != augment:
            self.augment = augment
            status = "ENABLED" if augment else "DISABLED"
            print(f"--> MergedHRTFDataset augmentation is now {status}.")

    def _augment_hrir(self, hrir, anthro,locations):
        """ Applies augmentation using the globally calculated statistics. """
        original_d3_val = anthro[11]

        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)

        scale_factor = target_d3_val / original_d3_val if original_d3_val != 0 else 1.0
        augmented_anthro = anthro.copy()

        distance_indices = list(range(9, 17))
        for i in distance_indices:
            augmented_anthro[i] *= scale_factor

        num_samples_orig = hrir.shape[-1]
        num_samples_new = int(np.round(num_samples_orig * scale_factor))
        hrir = scipy.signal.resample(hrir, num_samples_new, axis=-1)

        original_itds = np.array([get_itd_from_anthro(anthro, loc[0]) for loc in locations])
        target_itds = np.array([get_itd_from_anthro(augmented_anthro, loc[0]) for loc in locations])
        itd_shifts_seconds = target_itds - original_itds

        for i in range(hrir.shape[0]):
            hrir_slice = hrir[i]
            shift_for_this_location = itd_shifts_seconds[i]
            hrir[i] = apply_itd_shift(hrir_slice, shift_for_this_location, self.sample_rate)

        return hrir, augmented_anthro

    # ... The _process_hrtf and extend_locations methods remain unchanged ...
    def _process_hrtf(self, hrir,location):
        """ Converts HRIR to HRTF and applies normalization/scaling. """
        tf = np.abs(np.fft.fft(hrir, n=256))

        hrtf_np = tf[:,1:93]
        compute_dtf = True

        if compute_dtf:
            ctf = self._compute_ctf(hrtf_np, location)
            dtf_np = hrtf_np / (ctf + 1e-10)
            if self.scale == "log":
                ctf = 20 * np.log10(ctf + 1e-9)
                dtf_np = 20 * np.log10(dtf_np + 1e-9)
            hrtf_np = dtf_np
        else:
            if self.norm_way == 0:
                max_val = np.max((hrtf_np))
                if max_val > 1e-9: hrtf_np = hrtf_np / max_val
            ## second way is to divide by top 5% top value
            elif self.norm_way == 1:
                mag_flatten = hrtf_np.flatten()
                max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
                hrtf_np = hrtf_np / max_mag
            ## third way is to compute total energy of the equator
            elif self.norm_way == 2:
                equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
                hrtf_equator = hrtf_np[equator_index]
                equator_azi = location[equator_index, 0][0]
                new_equator_index = np.argsort(equator_azi)
                new_equator_azi = equator_azi[new_equator_index]
                new_equator_tf = hrtf_equator[new_equator_index]

                total_energy = 0
                for x in range(len(new_equator_index)):
                    if x == 0:
                        d_azi = 360 - new_equator_azi[-1]
                        # d_azi = new_equator_azi[1] - new_equator_azi[0]
                    else:
                        d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                    total_energy += np.square(new_equator_tf[x]).mean() * d_azi
                hrtf_np = hrtf_np / np.sqrt(total_energy / 360)
                breakpoint()
            elif self.norm_way == 4:
                if self.scale == 'log':
                    max_val = 20 #np.max(hrtf_np)
                    min_val =  -100 #np.min(hrtf_np)
                    hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1

            #     # print(np.sqrt(total_energy / 360))
            # ## fourth way is to normalize on common locations
            # ## [(0.0, 0.0), (180.0, 0.0), (210.0, 0.0), (330.0, 0.0), (30.0, 0.0), (150.0, 0.0)]
            # elif norm_way == 3:
            #     common_index = np.where(np.logical_and(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0),
            #                                            np.array(
            #                                                [round(x) in [0, 180, 210, 330, 30, 150] for x in location[:, 0]])))
            #     tf_common = tf[common_index]
            #     mean_energy = np.sqrt(np.square(tf_common).mean())
            #     # print(mean_energy)
            #     tf = tf / mean_energy

            if self.scale == "linear":
                hrtf_np = hrtf_np
            if self.scale == "log":
                hrtf_np = 20 * np.log10(hrtf_np + 1e-9)
                # max_val = 15 #np.max(hrtf_np)                           # Max HRTF value after normalizing based on equation energy (norm_way=2)
                # min_val =  -85 #np.min(hrtf_np)                         # Min HRTF value after normalizing based on equation energy (norm_way=2)
                # hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1     ## Normalize to (-1,1)

        return hrtf_np

    def _compute_ctf(self, hrtf_np, location):
        """
        Compute CTF using diffuse-field average method.
        CTF represents the direction-independent component.
        """
        # Method 1: Simple RMS average across all directions
        # ctf = np.sqrt(np.mean(hrtf_np**2, axis=0, keepdims=True))

        # Method 2: Weighted average based on solid angle (more accurate)
        # Weight by elevation to account for sphere sampling density
        elevation_weights = np.cos(np.radians(location[:, 1]))
        elevation_weights = elevation_weights / np.sum(elevation_weights)

        # Weighted RMS average
        weighted_hrtf = hrtf_np * elevation_weights[:, np.newaxis]
        ctf = np.sqrt(np.sum(weighted_hrtf**2, axis=0, keepdims=True))

        # Method 3: Geometric mean (alternative)
        # ctf = np.exp(np.mean(np.log(hrtf_np + 1e-10), axis=0, keepdims=True))

        return ctf

    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around for model interpolation. """
        index2 = np.where(locs_np[:, 0] < 30)[0]
        new_locs2 = locs_np[index2].copy()
        new_locs2[:, 0] += 360
        new_hrtfs2 = hrtfs_np[index2].copy()
        index1 = np.where(locs_np[:, 0] > 330)[0]
        new_locs1 = locs_np[index1].copy()
        new_locs1[:, 0] -= 360
        new_hrtfs1 = hrtfs_np[index1].copy()
        extended_locs = np.concatenate((new_locs1, locs_np, new_locs2), axis=0)
        extended_hrtfs = np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)
        return extended_locs, extended_hrtfs

    def __getitem__(self, idx):
        # MODIFIED: Retrieve data from the in-memory cache instead of reading a file
        cached_item = self.data_cache[idx]
        location_np = cached_item["location"]
        hrir = cached_item["hrir"]
        anthro_np = cached_item["anthro"]
        dataset_name = cached_item["dataset_name"]
        # subject_id = cached_item["subject_id"]
        # ear = cached_item["ear"]

        # subject_key = (dataset_name, subject_id,ear)
        zero_freqs_np = None # self.zero_freq_lookups.get(subject_key)

        if zero_freqs_np is None:
            # Create a default array of zeros if the data wasn't loaded.
            num_locations = location_np.shape[0]
            # Assuming 7 frequency features based on the original code's default. Adjust if needed.
            num_freq_features = 7
            zero_freqs_np = np.zeros((num_locations, num_freq_features), dtype=np.float32)




        # The rest of the processing logic remains the same
        if self.augment and random.random() < self.aug_prob:
            hrir, anthro_np = self._augment_hrir(hrir, anthro_np, location_np)

        hrtf_np = self._process_hrtf(hrir, location_np)
        location_extended, hrtf_extended = self.extend_locations(location_np, hrtf_np)
        _, zero_freqs_extended = self.extend_locations(location_np, zero_freqs_np)

        location_tensor = torch.from_numpy(location_extended.astype(np.float32))
        hrtf_tensor = torch.from_numpy(hrtf_extended.astype(np.float32))
        anthro_tensor = torch.from_numpy(anthro_np.astype(np.float32))
        ear_tensor = anthro_tensor[9:]
        zero_freq_tensor = torch.from_numpy(zero_freqs_extended.astype(np.int64))

        return location_tensor, hrtf_tensor, ear_tensor, dataset_name, zero_freq_tensor



class CoordinateBasedHRTFDataset(Dataset):
    """
    An all-in-one class that loads all HRTF data into memory and serves
    individual coordinate points for efficient batching.

    - Augmentation is performed once per subject during initialization.
    - All subjects are processed and flattened into a single pool of points.
    - __getitem__ becomes a simple, fast lookup.
    """
    def __init__(self, dataset_names, preprocessed_dir,
                 augment=False, aug_prob=0.6,
                 freq="all", scale="linear", norm_way=2):

        print("Initializing CoordinateBasedHRTFDataset...")
        self.dataset_names = dataset_names
        self.preprocessed_dir = preprocessed_dir
        self.augment = augment
        self.aug_prob = aug_prob
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.sample_rate = 44100

        subject_id_counter = 0

        # --- 1. Scan for all available preprocessed files ---
        self.file_registry = []
        for name in self.dataset_names:
            found_files = glob.glob(os.path.join(self.preprocessed_dir, f"{name}_*.pkl"))
            self.file_registry.extend(found_files)
        self.file_registry.sort()

        if not self.file_registry:
            raise FileNotFoundError(f"No .pkl files in {preprocessed_dir} for {dataset_names}")

        print(f"Found {len(self.file_registry)} total subject files.")

        # --- 2. Pool ALL anthropometry to build a global statistical model for augmentation ---
        if self.augment:
            print("Augmentation enabled. Pooling all anthropometry...")
            all_anthros = []
            for file_path in self.file_registry:
                with open(file_path, 'rb') as handle:
                    _, _, anthro_np = pkl.load(handle)
                    all_anthros.append(torch.from_numpy(anthro_np))

            all_anthros_stacked = torch.stack(all_anthros, dim=0)
            self.d3_mean = torch.mean(all_anthros_stacked[:, 11]).item()
            self.d3_std = torch.std(all_anthros_stacked[:, 11]).item()
            print(f"Global d3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")

        # --- 3. Load, process, and flatten ALL data points into a single list ---
        print("Loading and processing all subjects into a coordinate-based pool...")
        self.all_points = []
        subject_key_to_id = {}
        next_subject_id = 0

        for file_path in self.file_registry:
            # Parse the filename to get a unique key for the PERSON
            base_name = os.path.basename(file_path)
            match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
            if not match:
                continue

            dataset_name, file_num = match.group(1), int(match.group(2))
            subject_id_in_dataset = file_num // 2  # Integer division groups pairs (0,1 -> 0; 2,3 -> 1)
            subject_key = (dataset_name, subject_id_in_dataset)

            # Assign a new integer ID if we haven't seen this person before
            if subject_key not in subject_key_to_id:
                subject_key_to_id[subject_key] = next_subject_id
                next_subject_id += 1

            # Get the correct, shared ID for this person's ear
            current_subject_id = subject_key_to_id[subject_key]
            with open(file_path, 'rb') as handle:
                location_np, hrir, anthro_np = pkl.load(handle)

            # Decide whether to augment this specific subject
            if self.augment and random.random() < self.aug_prob:
                hrir, anthro_np = self._augment_hrir(hrir, anthro_np, location_np)

            # Process HRIR -> HRTF and extend locations
            hrtf_np = self._process_hrtf(hrir, location_np)
            location_extended, hrtf_extended = self.extend_locations(location_np, hrtf_np)

            # Convert to tensors
            location_tensor = torch.from_numpy(location_extended.astype(np.float32))
            hrtf_tensor = torch.from_numpy(hrtf_extended.astype(np.float32))
            anthro_tensor = torch.from_numpy(anthro_np.astype(np.float32))
            ear_tensor = anthro_tensor[9:] # Assumes ear features start at index 9

            # Flatten: Add each coordinate point as a separate item
            for i in range(location_tensor.shape[0]):
                self.all_points.append({
                        "location": location_tensor[i],
                        "hrtf": hrtf_tensor[i],
                        "ear_anthro": ear_tensor,
                        "subject_id": current_subject_id # Add the subject ID
                    })


        print(f"--> Initialization complete. Total data points: {len(self.all_points)}")

    def __len__(self):
        """Returns the total number of coordinate points across all subjects."""
        return len(self.all_points)

    def __getitem__(self, idx):
        """Returns a single pre-processed data point."""
        return self.all_points[idx]



    def _augment_hrir(self, hrir, anthro, locations):
        """ Applies augmentation using the globally calculated statistics. """
        original_d3_val = anthro[11]  # HARDCODED

        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)

        scale_factor = target_d3_val / original_d3_val if original_d3_val != 0 else 1.0
        augmented_anthro = anthro.copy()

        distance_indices = list(range(9, 17)) # HARDCODED
        for i in distance_indices:
            augmented_anthro[i] *= scale_factor

        num_samples_orig = hrir.shape[-1]
        num_samples_new = int(np.round(num_samples_orig * scale_factor))
        hrir = scipy.signal.resample(hrir, num_samples_new, axis=-1)

        original_itds = np.array([get_itd_from_anthro(anthro, loc[0]) for loc in locations])
        target_itds = np.array([get_itd_from_anthro(augmented_anthro, loc[0]) for loc in locations])
        itd_shifts_seconds = target_itds - original_itds

        for i in range(hrir.shape[0]):
            hrir_slice = hrir[i]
            shift_for_this_location = itd_shifts_seconds[i]
            hrir[i] = apply_itd_shift(hrir_slice, shift_for_this_location, self.sample_rate)

        return hrir, augmented_anthro

    def _process_hrtf(self, hrir, location):
        """ Converts HRIR to HRTF and applies normalization/scaling. """
        tf = np.abs(np.fft.fft(hrir, n=256))
        hrtf_np = tf[:, 1:93]

        if self.norm_way == 2: # Example for your primary normalization
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            if equator_index[0].size > 0:
                hrtf_equator = hrtf_np[equator_index]
                equator_azi = location[equator_index, 0][0]
                new_equator_index = np.argsort(equator_azi)
                new_equator_azi = equator_azi[new_equator_index]
                new_equator_tf = hrtf_equator[new_equator_index]

                total_energy = 0
                for x in range(len(new_equator_index)):
                    d_azi = (new_equator_azi[x] - new_equator_azi[x-1]) if x > 0 else (360 - new_equator_azi[-1])
                    total_energy += np.square(new_equator_tf[x]).mean() * d_azi

                norm_factor = np.sqrt(total_energy / 360)
                if norm_factor > 1e-9: hrtf_np = hrtf_np / norm_factor
        # ... other norm ways ...

        if self.scale == "log":
            hrtf_np = 20 * np.log10(hrtf_np + 1e-9)
            # max_val, min_val = 15, -85 # HARDCODED from your example
            # hrtf_np = 2 * (hrtf_np - min_val) / (max_val - min_val) - 1

        return hrtf_np

    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around. """
        index2 = np.where(locs_np[:, 0] < 30)[0]
        new_locs2, new_hrtfs2 = locs_np[index2].copy(), hrtfs_np[index2].copy()
        new_locs2[:, 0] += 360
        index1 = np.where(locs_np[:, 0] > 330)[0]
        new_locs1, new_hrtfs1 = locs_np[index1].copy(), hrtfs_np[index1].copy()
        new_locs1[:, 0] -= 360
        return np.concatenate((new_locs1, locs_np, new_locs2), axis=0), np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)






