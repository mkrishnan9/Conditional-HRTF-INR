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




class oldHRTFDataset(Dataset):
    def __init__(self, dataset="hutubs", freq="all", scale="linear", norm_way=0):
        ## assert dataset is one of HRTFDataset
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
        self.name = dataset
        self.dataset_obj = getattr(SOFAdatasets, dataset_dict[self.name])()
        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        # self.max_mag = self._find_global_max_magnitude()

    def __len__(self):
        return self.dataset_obj.__len__()

    def _get_hrtf_anthro(self, idx, freq, scale="linear", norm_way=0):
        # location, hrir = self.dataset_obj[idx]
        with open(os.path.join("/export/mkrishn9/hrtf_field/preprocessed_hrirs", "%s_%04d.pkl" % (self.name, idx)), 'rb') as handle:
            location, hrir, raw_anthro_data = pkl.load(handle)
        tf = np.abs(np.fft.fft(hrir, n=256))
        tf = tf[:, 1:93]  # first 128 freq bins, but up to 16k
        # tf = tf[:, 3:93]   # 500 Hz to 16kHz contribute to localization and are equalized
        ## how to normalize
        ## first way is to devide by max value
        if norm_way == 0:
            tf = tf / np.max(tf)
        ## second way is to devide by top 5% top value
        elif norm_way == 1:
            mag_flatten = tf.flatten()
            max_mag = np.mean(sorted(mag_flatten)[-int(mag_flatten.shape[0] / 20):])
            tf = tf / max_mag
        ## third way is to compute total energy of the equator
        elif norm_way == 2:
            equator_index = np.where(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0))
            tf_equator = tf[equator_index]
            equator_azi = location[equator_index, 0][0]
            new_equator_index = np.argsort(equator_azi)
            new_equator_azi = equator_azi[new_equator_index]
            new_equator_tf = tf_equator[new_equator_index]

            total_energy = 0
            for x in range(len(new_equator_index)):
                if x == 0:
                    d_azi = 360 - new_equator_azi[-1]
                    # d_azi = new_equator_azi[1] - new_equator_azi[0]
                else:
                    d_azi = new_equator_azi[x] - new_equator_azi[x - 1]
                total_energy += np.square(new_equator_tf[x]).mean() * d_azi
            tf = tf / np.sqrt(total_energy / 360)
            # print(np.sqrt(total_energy / 360))
        ## fourth way is to normalize on common locations
        ## [(0.0, 0.0), (180.0, 0.0), (210.0, 0.0), (330.0, 0.0), (30.0, 0.0), (150.0, 0.0)]
        elif norm_way == 3:
            common_index = np.where(np.logical_and(np.logical_and(location[:, 1] > -1, location[:, 1] <= 0),
                                                   np.array(
                                                       [round(x) in [0, 180, 210, 330, 30, 150] for x in location[:, 0]])))
            tf_common = tf[common_index]
            mean_energy = np.sqrt(np.square(tf_common).mean())
            # print(mean_energy)
            tf = tf / mean_energy

        if scale == "linear":
            tf = tf
        elif scale == "log":
            tf = 20 * np.log10(tf)
        if freq == "all":
            return location, tf, raw_anthro_data

        return location, tf[:, freq][:, np.newaxis], raw_anthro_data

    def _find_global_max_magnitude(self):
        max_mag = 0
        for i in range(self.__len__()):
            _, tf_mag, _ = self._get_hrtf_anthro(i, "all", "linear")
            cur_max_mag = np.max(tf_mag)
            if cur_max_mag > max_mag:
                max_mag = cur_max_mag
        return max_mag

    def __getitem__(self, idx):
        location, hrtf, anthro= self._get_hrtf_anthro(idx, self.freq, self.scale, self.norm_way)
        # return location, hrtf / self.max_mag
        return location, hrtf, anthro

    def _plot_frontal_data(self, idx, ax):
        loc_idx = self.dataset_obj._get_frontal_locidx()
        _, hrtf,_ = self._get_hrtf_anthro(idx, "all", "linear")
        # hrtf = hrtf / self.max_mag
        ax.plot(np.log(hrtf[loc_idx]), label=self.dataset_obj.name.upper()+" Subject%s" % self.dataset_obj._get_subject_ID(idx))
        ax.set(xticks=list(np.arange(0, 128 + 16, 16)),
               xticklabels=['{:,.2f}k'.format(x) for x in list(np.arange(0, 128 + 16, 16) / 256 * 44.1)],
               title="Frontal HRTF",
               ylabel='Log Magnitude',
               xlabel='Frequency (Hz)')

class HRTFDataset(Dataset):
    def __init__(self, dataset="hutubs", freq="all", scale="linear", norm_way=0,
                 preprocessed_dir="/export/mkrishn9/hrtf_field/preprocessed_hrirs",
                 hutubs_anthro_path="/export/mkrishn9/hrtf_field/hutubs/AntrhopometricMeasures.csv",
                 augment=False, aug_prob=0.6):
        ## assert dataset is one of HRTFDataset
        dataset_dict = {"ari": "ARI", "hutubs": "HUTUBS", "ita": "ITA", "cipic": "CIPIC",
                        "3d3a": "Prin3D3A", "riec": "RIEC", "bili": "BiLi",
                        "listen": "Listen", "crossmod": "Crossmod", "sadie": "SADIE"}
        self.name = dataset
        try:
            # This object might still be useful for metadata if needed
            self.dataset_obj = getattr(SOFAdatasets, dataset_dict[self.name])()
            potential_len = self.dataset_obj.__len__() # Get the original potential length
        except Exception as e:
             print(f"Warning: Could not instantiate SOFAdatasets for {self.name}: {e}. Trying to proceed without it.")
             # Fallback: Try to determine length by scanning files directly
             potential_len = 1000 # Default large number, will be refined below

        self.freq = freq
        self.scale = scale
        self.norm_way = norm_way
        self.preprocessed_dir = preprocessed_dir

        # --- Scan for available preprocessed files ---
        self.valid_indices = []
        print(f"Scanning {self.preprocessed_dir} for available '{self.name}_*.pkl' files...")
        # Scan directory directly using glob, more robust than sequential scan
        all_files = glob.glob(os.path.join(self.preprocessed_dir, f"{self.name}_*.pkl"))
        found_indices = []
        for f in all_files:
            try:
                # Extract index from filename
                index_str = os.path.basename(f).replace(f"{self.name}_", "").replace(".pkl", "")
                found_indices.append(int(index_str))
            except ValueError:
                print(f"Warning: Could not parse index from filename: {f}")
        self.valid_indices = sorted(found_indices) # Ensure they are sorted

        if not self.valid_indices:
             print(f"ERROR: No valid preprocessed files found matching '{self.name}_*.pkl' in {self.preprocessed_dir}")
             # Optional: raise error or allow empty dataset
             # raise FileNotFoundError(f"No preprocessed data found for {self.name}")

        print(f"Found {len(self.valid_indices)} valid preprocessed files/ears.")
        # --- END Scan ---

        self.augment = augment
        self.aug_prob = aug_prob
        self.sample_rate = 44100


        if self.augment:
            print("Augmentation is ENABLED for this dataset.")
            self.anthro_df = pd.read_csv(hutubs_anthro_path, index_col=0)
            # Convert a representative column of features to a list for selection
            self.anthro_feature_cols = self.anthro_df.columns.tolist()
            all_d3s = pd.concat([self.anthro_df["L_d3"], self.anthro_df["R_d3"]]).dropna()
            self.d3_mean = all_d3s.mean()
            self.d3_std = all_d3s.std()
            print(f"D3 distribution: mean={self.d3_mean:.2f}, std={self.d3_std:.2f}")
        else:
            print("Augmentation is DISABLED for this dataset.")
            self.anthro_df = None

    def set_augment(self, augment):
            self.augment = augment

        # --- NEW METHOD: The core augmentation logic ---
    def _augment_hrir(self, hrir, anthro, locations):
        ### Applies augmentation by sampling a random scale factor from a statistical distribution based on the dataset.


        if self.anthro_df is None:
            return hrir, anthro # Cannot augment without the master table

        original_d3_val = anthro[11] #anthro[15] ###HARDCODED
        min_val = self.d3_mean - 3 * self.d3_std
        max_val = self.d3_mean + 3 * self.d3_std
        target_d3_val = np.clip(np.random.normal(self.d3_mean, self.d3_std), min_val, max_val)
        scale_factor = target_d3_val / original_d3_val
        augmented_anthro = anthro.copy()
        distance_indices = list(range(23))
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

    # --- ADDED METHOD ---
    def extend_locations(self, locs_np, hrtfs_np):
        """ Extends azimuth range to handle wrap-around near 0/360 degrees. """
        # Ensure inputs are numpy arrays
        if isinstance(locs_np, torch.Tensor):
            locs_np = locs_np.numpy()
        if isinstance(hrtfs_np, torch.Tensor):
            hrtfs_np = hrtfs_np.numpy()

        # Find points near 0 degrees (e.g., < 30)
        index2 = np.where(locs_np[:, 0] < 30)[0]
        if len(index2) > 0:
            new_locs2 = locs_np[index2].copy()
            new_locs2[:, 0] += 360
            new_hrtfs2 = hrtfs_np[index2].copy()
        else:
            new_locs2 = np.empty((0, locs_np.shape[1]))
            new_hrtfs2 = np.empty((0, hrtfs_np.shape[1]))

        # Find points near 360 degrees (e.g., > 330)
        index1 = np.where(locs_np[:, 0] > 330)[0]
        if len(index1) > 0:
            new_locs1 = locs_np[index1].copy()
            new_locs1[:, 0] -= 360
            new_hrtfs1 = hrtfs_np[index1].copy()
        else:
            new_locs1 = np.empty((0, locs_np.shape[1]))
            new_hrtfs1 = np.empty((0, hrtfs_np.shape[1]))

        # Concatenate: points < 0 | original points | points > 360
        extended_locs = np.concatenate((new_locs1, locs_np, new_locs2), axis=0)
        extended_hrtfs = np.concatenate((new_hrtfs1, hrtfs_np, new_hrtfs2), axis=0)

        return extended_locs, extended_hrtfs
    # --- END ADDED METHOD ---

    def __len__(self):
        # Return the count of *actually available* data points
        return len(self.valid_indices)

    def _get_hrtf_anthro(self, actual_idx, freq, scale="linear", norm_way=0,override_data=None):
        if override_data:
            hrir, anthro_np, location_np = override_data
        else:
            pkl_path = os.path.join(self.preprocessed_dir, f"{self.name}_{actual_idx:04d}.pkl")
            with open(pkl_path, 'rb') as handle:
                location_np, hrir, anthro_np = pkl.load(handle)

        tf = np.abs(np.fft.fft(hrir, n=256))
        tf = tf[:, 1:93]

        # Apply normalization (Keep your improved logic)
        # ... (norm_way 0, 1, 2, 3 with checks) ...
        if norm_way == 0:
            max_val = np.max(tf)
            tf = tf / max_val if max_val > 1e-9 else tf
        elif norm_way == 1:
             mag_flatten = tf.flatten()
             if len(mag_flatten) > 0:
                  top_5_percent_index = -max(1, int(mag_flatten.shape[0] / 20))
                  max_mag = np.mean(sorted(mag_flatten)[top_5_percent_index:])
                  tf = tf / max_mag if max_mag > 1e-9 else tf
             # else: tf = tf # No change if empty
        # ... (add robust norm_way 2 and 3 logic back here) ...
        elif norm_way == 2: # Simplified example, add your robust version back
             norm_factor = np.sqrt(np.mean(np.square(tf))) # Example: RMS norm
             tf = tf / norm_factor if norm_factor > 1e-9 else tf
        elif norm_way == 3: # Simplified example, add your robust version back
             norm_factor = np.sqrt(np.mean(np.square(tf))) # Example: RMS norm
             tf = tf / norm_factor if norm_factor > 1e-9 else tf


        # Apply scaling
        if scale == "linear":
            pass
        elif scale == "log":
            tf = 20 * np.log10(tf + 1e-9) # Add epsilon for stability

        # Select frequency
        if freq == "all":
            selected_tf = tf
        else:
            try:
                selected_tf = tf[:, freq]
                if selected_tf.ndim == 1:
                     selected_tf = selected_tf[:, np.newaxis]
            except IndexError:
                 print(f"Error: Invalid frequency selection '{freq}' for tf shape {tf.shape}. Using all.")
                 selected_tf = tf # Fallback to all frequencies

        hrtf_np = selected_tf

        # Return numpy arrays BEFORE extension
        return location_np.astype(np.float32), hrtf_np.astype(np.float32), anthro_np.astype(np.float32)

    def __getitem__(self, idx):
        actual_idx = idx #self.valid_indices[idx]

        augmented_data_tuple = None
        if self.augment and random.random() < self.aug_prob:
            # Load original data first to serve as a base for augmentation
            pkl_path = os.path.join(self.preprocessed_dir, f"{self.name}_{actual_idx:04d}.pkl")
            with open(pkl_path, 'rb') as handle:
                location_np, hrir_orig, anthro_orig = pkl.load(handle)

            hrir_aug, anthro_aug = self._augment_hrir(hrir_orig, anthro_orig, location_np)
            augmented_data_tuple = (hrir_aug, anthro_aug, location_np)


        # 1. Get processed data (as numpy arrays)
        location_np, hrtf_np, anthro_np = self._get_hrtf_anthro(actual_idx, self.freq, self.scale, self.norm_way,override_data=augmented_data_tuple)

        # --- APPLY LOCATION EXTENSION ---
        location_extended_np, hrtf_extended_np = self.extend_locations(location_np, hrtf_np)
        # --- END EXTENSION ---

        # 3. Convert final extended/processed data to tensors
        location_tensor = torch.from_numpy(location_extended_np)
        hrtf_tensor = torch.from_numpy(hrtf_extended_np)
        anthro_tensor = torch.from_numpy(anthro_np) # Anthro doesn't change with extension

        return location_tensor, hrtf_tensor, anthro_tensor

    def _plot_frontal_data(self, idx, ax):
         # Get the actual file index corresponding to the dataset index `idx`
         if idx < 0 or idx >= len(self.valid_indices):
             print(f"Plotting Error: Index {idx} is out of valid range {len(self.valid_indices)}")
             return
         actual_idx = self.valid_indices[idx]

         # Need subject ID for label - retrieve from SOFAdatasets object if possible
         subject_id_str = "Unknown"
         if hasattr(self, 'dataset_obj') and self.dataset_obj is not None:
             try:
                 # Assuming _get_subject_ID takes the subject_idx (actual_idx // 2)
                 subject_idx = actual_idx // 2
                 subject_id_str = self.dataset_obj._get_subject_ID(subject_idx)
             except Exception as e:
                 print(f"Warning: Could not get Subject ID for index {actual_idx}: {e}")
                 subject_id_str = f"FileIdx_{actual_idx}"


         # Find the frontal location index within the specific data loaded for actual_idx
         try:
             location, hrtf, _ = self._get_hrtf_anthro(actual_idx, "all", "linear", 0) # Get raw linear magnitude

             if location is None or hrtf is None:
                 print(f"Plotting Error: Failed to load data for index {actual_idx}")
                 return

             # Find index where azimuth and elevation are closest to (0, 0)
             # Handle wrap-around for azimuth=0 vs azimuth=360
             azimuth_diff = np.minimum(np.abs(location[:, 0] - 0), np.abs(location[:, 0] - 360))
             distances = np.sqrt(np.square(azimuth_diff) + np.square(location[:, 1]))

             frontal_loc_idx = np.argmin(distances)
             if np.min(distances) > 1.0: # Check if (0,0) or something close was actually found
                 print(f"Warning: Frontal location (0,0) not found precisely for index {actual_idx}. Using closest: {location[frontal_loc_idx]}")


             # Plot the HRTF magnitude at the frontal location
             if frontal_loc_idx < hrtf.shape[0]:
                # Apply log magnitude for plotting
                log_hrtf_frontal = 20 * np.log10(hrtf[frontal_loc_idx, :] + 1e-9) # Use full frequency range returned

                # Calculate frequencies corresponding to bins 1 to 92 of a 256-point FFT
                num_freq_bins = log_hrtf_frontal.shape[0] # Should be 92
                sampling_rate = 44100 # Assuming this rate
                fft_n = 256 # Based on original code

                freqs_hz = np.fft.fftfreq(fft_n, 1.0/sampling_rate)[1:93]
                freqs_khz = freqs_hz / 1000.0

                ax.plot(freqs_khz, log_hrtf_frontal, label=f"{self.name.upper()} Subject {subject_id_str}")

                # Adjust x-axis ticks and labels based on the actual frequency range (approx 0.17kHz to 16kHz)
                tick_freqs_khz = np.arange(0, 17, 2) # Ticks every 2 kHz up to 16 kHz
                ax.set_xticks(tick_freqs_khz)
                ax.set_xticklabels([f'{k:.1f}k' for k in tick_freqs_khz])

                ax.set(title="Frontal HRTF",
                       ylabel='Log Magnitude (dB)',
                       xlabel='Frequency (kHz)')
                ax.legend() # Add legend to show labels
             else:
                 print(f"Plotting Error: frontal_loc_idx {frontal_loc_idx} out of bounds for HRTF shape {hrtf.shape}")

         except Exception as e:
              print(f"Error during plotting for index {idx} (actual index {actual_idx}): {e}")


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
            max_val = 15 #np.max(hrtf_np)                           #Max HRTF value after normalizing based on equation energy (norm_way=2)
            min_val =  -85 #np.min(hrtf_np)                         #Min HRTF value after normalizing based on equation energy (norm_way=2)
            hrtf_np = 2* (hrtf_np-min_val)/(max_val-min_val) -1     ## Normalize to (-1,1)

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





class oldMergedHRTFDataset(Dataset):
    def __init__(self, all_dataset_names, freq, scale="linear", norm_way=2):
        self.all_dataset_names = all_dataset_names
        # ["ari", "hutubs", "cipic", "3d3a", "riec", "bili", "listen", "crossmod", "sadie", "ita"]
        self.all_datasets = []
        self.length_array = []
        self.all_data = []
        for dataset_name in self.all_dataset_names:
            self.all_datasets.append(HRTFDataset(dataset_name, freq, scale, norm_way))
        for dataset in self.all_datasets:
            for item_idx in range(len(dataset)):
                locs, hrtfs = dataset[item_idx]
                self.all_data.append((locs, hrtfs, dataset.name))
            self.length_array.append(len(dataset))
        # self.length_sum = np.insert(np.cumsum(self.length_array), 0, 0)

    def __len__(self):
        return np.sum(self.length_array)

    def extend_locations(self, locs, hrtfs):
        ## Extend locations from -30 to 0 and from 360 to 390
        index1 = np.where(locs[:, 0] > 330)
        new_locs1 = locs.copy()[index1]
        new_locs1[:, 0] -= 360
        index2 = np.where(locs[:, 0] < 30)
        new_locs2 = locs.copy()[index2]
        new_locs2[:, 0] += 360
        num_loc = new_locs1.shape[0] + locs.shape[0] + new_locs2.shape[0]
        # assign values for locs
        new_locs = torch.zeros(num_loc, 2)
        new_locs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(locs)
        new_locs[:new_locs1.shape[0]] = torch.from_numpy(new_locs1)
        new_locs[-new_locs2.shape[0]:] = torch.from_numpy(new_locs2)
        # assign values for hrtfs
        new_hrtfs = torch.zeros(num_loc, hrtfs.shape[1])
        new_hrtfs[new_locs1.shape[0]:-new_locs2.shape[0]] = torch.from_numpy(hrtfs)
        new_hrtfs[:new_locs1.shape[0]] = torch.from_numpy(hrtfs.copy()[index1])
        new_hrtfs[-new_locs2.shape[0]:] = torch.from_numpy(hrtfs.copy()[index2])
        return new_locs, new_hrtfs

    def __getitem__(self, idx):
        locs, hrtfs, names = self.all_data[idx]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names

    def collate_fn(self, samples):
        B = len(samples)
        len_sorted, _ = torch.sort(torch.Tensor([sample[0].shape[0] for sample in samples]), descending=True)
        max_num_loc = int(len_sorted[0].item())
        n_freq = samples[0][1].shape[1]
        locs, hrtfs, masks, names = [], [], [], []
        locs = -torch.ones((B, max_num_loc, 2))
        hrtfs = -torch.ones((B, max_num_loc, n_freq))
        masks = torch.zeros((B, max_num_loc, n_freq))
        for idx, sample in enumerate(samples):
            num_loc = sample[0].shape[0]
            loc, hrtf, name = sample
            locs[idx, :num_loc, :] = loc
            hrtfs[idx, :num_loc, :] = hrtf
            masks[idx, :num_loc, :] = 1
            names.append(name)
        return locs, hrtfs, masks, default_collate(names)


class PartialHRTFDataset(MergedHRTFDataset):
    def __init__(self, dataset_name="riec", freq=15, scale="linear", norm_way=2):
        super().__init__(dataset_name, freq, scale, norm_way)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        locs, hrtfs, names = self.all_data[idx]
        indices = np.array([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
        26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
        52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  73,  75,  77,
        79,  81,  83,  85,  87,  89,  91,  93,  95,  97,  99, 101, 103,
       105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129,
       131, 133, 135, 137, 139, 141, 143, 144, 146, 148, 150, 152, 154,
       156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180,
       182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206,
       208, 210, 212, 214, 217, 219, 221, 223, 225, 227, 229, 231, 233,
       235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259,
       261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285,
       287, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
       312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336,
       338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 361, 363,
       365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389,
       391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415,
       417, 419, 421, 423, 425, 427, 429, 431, 432, 434, 436, 438, 440,
       442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466,
       468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492,
       494, 496, 498, 500, 502, 505, 507, 509, 511, 513, 515, 517, 519,
       521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545,
       547, 549, 551, 553, 555, 557, 559, 561, 563, 565, 567, 569, 571,
       573, 575, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596,
       598, 600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622,
       624, 626, 628, 630, 632, 634, 636, 638, 640, 642, 644, 646, 649,
       651, 653, 655, 657, 659, 661, 663, 665, 667, 669, 671, 673, 675,
       677, 679, 681, 683, 685, 687, 689, 691, 693, 695, 697, 699, 701,
       703, 705, 707, 709, 711, 713, 715, 717, 719, 720, 722, 724, 726,
       728, 730, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752,
       754, 756, 758, 760, 762, 764, 766, 768, 770, 772, 774, 776, 778,
       780, 782, 784, 786, 788, 790, 793, 795, 797, 799, 801, 803, 805,
       807, 809, 811, 813, 815, 817, 819, 821, 823, 825, 827, 829, 831,
       833, 835, 837, 839, 841, 843, 845, 847, 849, 851, 853, 855, 857,
       859, 861, 863, 864])
        locs = locs[indices]
        hrtfs = hrtfs[indices]
        locs, hrtfs = self.extend_locations(locs, hrtfs)
        return locs, hrtfs, names


class HRTFFitting(Dataset):
    def __init__(self, location, hrtf, part="full"):
        super(HRTFFitting, self).__init__()
        assert part in ["full", "half", "random_half"]
        num_locations = location.shape[0]
        assert hrtf.shape[0] == num_locations

        self.hrtf = hrtf
        self.coords = location

        if part == "full":
            self.indices = np.arange(num_locations)
        elif part == "half":
            self.indices = np.arange(0, num_locations, 2)
        elif part == "random_half":
            self.indices = np.random.choice(num_locations, num_locations // 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords[self.indices], self.hrtf[self.indices]


def fitting_dataset_wrapper(idx, dataset="crossmod", freq=1, part="full"):
    dataset = HRTFDataset(dataset, freq)
    loc, hrtf = dataset[idx]
    return HRTFFitting(loc, hrtf, part)



if __name__ == "__main__":
    res = HRTFDataset()
    loc, hrtf = res[3]
    print(loc.shape)
    print(hrtf.shape)



