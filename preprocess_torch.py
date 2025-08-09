
import os
import glob
import pickle as pkl
import numpy as np
import torch

def extend_locations(locs_np, hrtfs_np):
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

def _process_hrtf(hrir):
    """ Converts HRIR to HRTF and applies normalization/scaling. """
    tf = np.abs(np.fft.fft(hrir, n=256))
    hrtf_np = tf[:, 1:93]


    max_val = np.max(hrtf_np)
    if max_val > 1e-9: hrtf_np = hrtf_np / max_val



    hrtf_np = 20 * np.log10(hrtf_np + 1e-9)
    return hrtf_np


raw_data_dir = "/export/mkrishn9/hrtf_field/preprocessed_hrirs_common" # Your original pkl files
processed_data_dir = "/export/mkrishn9/hrtf_field/preprocessed_hrirs_common_pt" # Tarrget directory
os.makedirs(processed_data_dir, exist_ok=True)

file_paths = glob.glob(os.path.join(raw_data_dir, "*.pkl"))
print(f"Found {len(file_paths)} files to process.")

for i, file_path in enumerate(file_paths):
    with open(file_path, 'rb') as handle:
        location_np, hrir, anthro_np = pkl.load(handle)

    # 1. Perform the expensive processing ONCE
    hrtf_np = _process_hrtf(hrir)
    location_extended, hrtf_extended = extend_locations(location_np, hrtf_np)

    # 2. Prepare data for saving
    data_to_save = {
        'location_original': location_np,
        'location_extended': location_extended,
        'hrir': hrir, # Keep hrir for on-the-fly augmentation
        'hrtf_extended': hrtf_extended, # The pre-processed HRTF
        'anthro': anthro_np
    }

    # 3. Save to a new, efficient file
    base_name = os.path.basename(file_path).replace('.pkl', '.pt')
    new_file_path = os.path.join(processed_data_dir, base_name)
    torch.save(data_to_save, new_file_path)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(file_paths)} files.")

print("✅ Preprocessing complete!")
