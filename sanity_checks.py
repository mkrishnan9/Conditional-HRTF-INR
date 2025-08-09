import h5py
import numpy as np

# Make sure this path is correct
chedar_mat_path = "/export/mkrishn9/hrtf_field/chedar/measurements.mat"

print("--- Exploring the '#refs#' group in the CHEDAR file ---")
try:
    with h5py.File(chedar_mat_path, 'r') as f:
        if '#refs#' in f:
            refs_group = f['#refs#']
            print(f"Found {len(refs_group)} items inside '#refs#'.\n")

            # We will inspect each item in this group
            for name, item in refs_group.items():
                print(f"--- Item Name: '{name}' ---")
                print("Object:", item)

                # Print important attributes if they exist
                if hasattr(item, 'shape'):
                    print("Shape:", item.shape)
                if hasattr(item, 'dtype'):
                    print("Dtype:", item.dtype)
                    # If it's a compound type, this is very important!
                    if item.dtype.names:
                        print("Field Names:", item.dtype.names)
                print("-" * 25) # Separator for clarity
        else:
            print("Error: The '#refs#' group was not found in the file.")

except Exception as e:
    print(f"An error occurred: {e}")
