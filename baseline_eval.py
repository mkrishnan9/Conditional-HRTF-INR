import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import random
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.distance import cdist
import re
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from HRTFdatasets import MergedHRTFDataset
from model import CondHRTFNetwork
import matplotlib.pyplot as plt
from collections import defaultdict
import json


def calculate_anthro_stats(dataset_subset):
    """Calculate mean and std dev for anthropometry across the dataset subset."""
    print("Calculating anthropometry statistics on training set...")
    all_anthro_tensors = []

    for i in range(len(dataset_subset)):
        try:
            # Handle both LocationFilteredDataset and regular Subset
            if hasattr(dataset_subset, 'base_dataset'):
                # LocationFilteredDataset
                _, _, anthro_tensor, _ = dataset_subset[i]
            else:
                # Regular Subset
                _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]
            all_anthro_tensors.append(anthro_tensor)
        except Exception as e:
            print(f"Warning: Could not retrieve item at index {i}: {e}")
            continue

    all_anthro_stacked = torch.stack(all_anthro_tensors, dim=0)
    mean = torch.mean(all_anthro_stacked, dim=0, dtype=torch.float32)
    std = torch.std(all_anthro_stacked, dim=0)
    std = torch.clamp(std, min=1e-8)

    return mean, std


class LocationFilteredDataset(Dataset):
    """
    Wrapper dataset that filters locations based on indices.
    Used to create train/val splits within subjects based on spatial locations.
    """
    def __init__(self, base_dataset, indices, location_filter_dict=None, mode='all'):
        """
        Args:
            base_dataset: The MergedHRTFDataset
            indices: Subject indices to include (from Subset)
            location_filter_dict: Dict mapping dataset_name to location indices to include
            mode: 'all' (use all locations), 'include' (use only specified locations),
                  'exclude' (exclude specified locations)
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.location_filter_dict = location_filter_dict or {}
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index in the base dataset
        actual_idx = self.indices[idx]

        # For Subset compatibility
        if hasattr(self.base_dataset, 'dataset'):
            # This is a Subset
            actual_dataset = self.base_dataset.dataset
            actual_idx = self.base_dataset.indices[idx]
        else:
            actual_dataset = self.base_dataset

        # Get the file path to determine dataset name
        file_path = actual_dataset.file_registry[actual_idx]
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match:
            return actual_dataset[actual_idx]

        dataset_name = match.group(1)

        # Get the data
        locations, hrtfs, anthro, name = actual_dataset[actual_idx]

        # Apply location filtering if specified for this dataset
        if dataset_name in self.location_filter_dict and self.mode != 'all':
            filter_indices = self.location_filter_dict[dataset_name]

            if self.mode == 'include':
                # Only include specified locations
                if len(filter_indices) > 0:
                    locations = locations[filter_indices]
                    hrtfs = hrtfs[filter_indices]
                else:
                    # Return empty tensors if no indices to include
                    locations = torch.zeros(0, 2)
                    hrtfs = torch.zeros(0, hrtfs.shape[-1])
            elif self.mode == 'exclude':
                # Exclude specified locations
                mask = torch.ones(len(locations), dtype=torch.bool)
                if len(filter_indices) > 0:
                    mask[filter_indices] = False
                locations = locations[mask]
                hrtfs = hrtfs[mask]

        return locations, hrtfs, anthro, name


class HRTFBaselineEvaluator:
    """Evaluator for classical HRTF prediction baselines."""

    def __init__(self, train_dataset, val_dataset, device='cuda', scale='log'):
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scale = scale

        # Extract and organize training data
        self.train_data = self._extract_data(train_dataset)
        self.val_data = self._extract_data(val_dataset)

        # Precompute some useful structures
        self._precompute_structures()

    def _extract_data(self, dataset):
        """Extract all data from a dataset (handles LocationFilteredDataset)."""
        data = {
            'locations': [],
            'hrtfs': [],
            'anthropometry': [],
            'names': []
        }

        print(f"Extracting data from dataset with {len(dataset)} items...")
        for i in range(len(dataset)):
            try:
                locs, hrtfs, anthro, name = dataset[i]
                # Skip empty entries
                if len(locs) > 0:
                    data['locations'].append(locs)
                    data['hrtfs'].append(hrtfs)
                    data['anthropometry'].append(anthro)
                    data['names'].append(name)
            except Exception as e:
                print(f"Warning: Could not extract item {i}: {e}")
                continue

        print(f"Successfully extracted {len(data['locations'])} non-empty items")
        return data

    def _precompute_structures(self):
        """Precompute data structures for efficient baseline computation."""
        # Stack all training anthropometry for distance calculations
        if len(self.train_data['anthropometry']) > 0:
            self.train_anthro_matrix = torch.stack(self.train_data['anthropometry'])
        else:
            self.train_anthro_matrix = torch.empty(0, 10)  # Assuming 10 anthro dims

        # Create location -> HRTF mappings for population average
        self.location_hrtf_map = defaultdict(list)
        for locs, hrtfs in zip(self.train_data['locations'], self.train_data['hrtfs']):
            for loc, hrtf in zip(locs, hrtfs):
                # Convert location to tuple for hashing (round to avoid floating point issues)
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                loc_key = tuple(np.round(loc_np, 2))
                self.location_hrtf_map[loc_key].append(hrtf)

        # Compute population average for each location
        self.population_avg = {}
        for loc_key, hrtf_list in self.location_hrtf_map.items():
            self.population_avg[loc_key] = torch.stack(hrtf_list).mean(dim=0)

        # Create KDTree for location-based nearest neighbor search
        all_train_locs = []
        self.loc_to_subject_idx = []  # Maps flattened location index to (subject_idx, loc_idx)

        for subj_idx, locs in enumerate(self.train_data['locations']):
            for loc_idx, loc in enumerate(locs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                all_train_locs.append(loc_np)
                self.loc_to_subject_idx.append((subj_idx, loc_idx))

        if all_train_locs:
            self.location_kdtree = KDTree(np.array(all_train_locs))
            self.all_train_locs_array = np.array(all_train_locs)
        else:
            self.location_kdtree = None
            self.all_train_locs_array = np.array([])

    def evaluate_population_average_exact(self):
        """Baseline 1a: Population average using only exact location matches."""
        total_lsd = 0
        total_points = 0
        skipped_points = 0

        for val_locs, val_hrtfs in zip(self.val_data['locations'], self.val_data['hrtfs']):
            for loc, true_hrtf in zip(val_locs, val_hrtfs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                loc_key = tuple(np.round(loc_np, 2))

                if loc_key in self.population_avg:
                    pred_hrtf = self.population_avg[loc_key]
                    lsd = self._compute_lsd(true_hrtf, pred_hrtf)
                    total_lsd += lsd
                    total_points += 1
                else:
                    skipped_points += 1

        avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
        print(f"Population Average (Exact): LSD = {avg_lsd:.4f}, Skipped {skipped_points} points")
        return avg_lsd

    def evaluate_population_average_interpolated(self, method='linear'):
        """Baseline 1b: Population average with interpolation."""
        if len(self.train_data['locations']) == 0:
            print(f"No training data available for interpolation")
            return float('inf')

        # Build interpolator from all training data
        interpolator = self._build_interpolator(method)
        if interpolator is None:
            return float('inf')

        total_lsd = 0
        total_points = 0

        for val_locs, val_hrtfs in zip(self.val_data['locations'], self.val_data['hrtfs']):
            for loc, true_hrtf in zip(val_locs, val_hrtfs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                pred_hrtf = self._interpolate_hrtf(loc_np, interpolator)
                if pred_hrtf is not None:
                    lsd = self._compute_lsd(true_hrtf, torch.tensor(pred_hrtf))
                    total_lsd += lsd
                    total_points += 1

        avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
        print(f"Population Average (Interpolated-{method}): LSD = {avg_lsd:.4f}")
        return avg_lsd

    def _find_closest_subject_by_overall_lsd(self, val_locs, val_hrtfs):
        """
        Find the training subject with the lowest average LSD across all shared locations.
        This is the "oracle" subject that best matches this validation subject overall.
        """
        best_subject_idx = None
        best_avg_lsd = float('inf')
        best_subject_info = {}

        # Convert validation locations to keys for fast lookup
        val_loc_dict = {}
        for i, loc in enumerate(val_locs):
            loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
            loc_key = tuple(np.round(loc_np, 2))
            val_loc_dict[loc_key] = i

        # Evaluate each training subject
        for train_idx, (train_locs, train_hrtfs) in enumerate(
            zip(self.train_data['locations'], self.train_data['hrtfs'])
        ):
            total_lsd = 0
            num_matches = 0
            matched_locations = []

            # Find all matching locations between this training subject and validation subject
            for train_loc, train_hrtf in zip(train_locs, train_hrtfs):
                train_loc_np = train_loc.numpy() if isinstance(train_loc, torch.Tensor) else train_loc
                train_loc_key = tuple(np.round(train_loc_np, 2))

                if train_loc_key in val_loc_dict:
                    val_idx = val_loc_dict[train_loc_key]
                    val_hrtf = val_hrtfs[val_idx]
                    lsd = self._compute_lsd(val_hrtf, train_hrtf)
                    total_lsd += lsd
                    num_matches += 1
                    matched_locations.append(train_loc_key)

            # Compute average LSD for this subject
            if num_matches > 0:
                avg_lsd = total_lsd / num_matches
                if avg_lsd < best_avg_lsd:
                    best_avg_lsd = avg_lsd
                    best_subject_idx = train_idx
                    best_subject_info = {
                        'subject_idx': train_idx,
                        'avg_lsd': avg_lsd,
                        'num_matches': num_matches,
                        'matched_locations': matched_locations
                    }

        return best_subject_info

    def evaluate_closest_anthro_exact_location(self, anthro_mean=None, anthro_std=None,
                                                save_plots=False, plot_dir='./hrtf_comparison_plots'):
        """
        Baseline 2a: Closest anthropometry with exact location match.
        Now includes the best overall subject match (oracle), detailed comparisons,
        and a new plot showing locations where the oracle is worse than the population average.
        """
        if len(self.train_data['anthropometry']) == 0:
            print("No training data available")
            return float('inf')

        # Calculate normalization stats if not provided
        if anthro_mean is not None and anthro_std is not None:
            print("Using anthropometry normalization for distance calculation")
            train_anthro_normalized = []
            for anthro in self.train_data['anthropometry']:
                anthro_norm = (anthro - anthro_mean) / anthro_std
                train_anthro_normalized.append(anthro_norm)
            train_anthro_matrix_normalized = torch.stack(train_anthro_normalized)
        else:
            print("No normalization - using raw anthropometry values")
            train_anthro_matrix_normalized = self.train_anthro_matrix

        # Create plot directory if saving plots
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)

        total_lsd = 0
        total_points = 0
        skipped_points = 0
        plot_count = 0
        max_plots = 25

        # Track additional metrics
        pop_avg_total_lsd = 0
        oracle_total_lsd = 0

        # Track per-subject performance
        per_subject_results = []

        for val_idx, (val_locs, val_hrtfs, val_anthro) in enumerate(
            zip(self.val_data['locations'], self.val_data['hrtfs'], self.val_data['anthropometry'])
        ):
            # Find the oracle subject (best overall match) for this validation subject
            oracle_info = self._find_closest_subject_by_overall_lsd(val_locs, val_hrtfs)

            # Build oracle subject's location map if found
            oracle_loc_map = {}
            if oracle_info and 'subject_idx' in oracle_info:
                oracle_idx = oracle_info['subject_idx']
                oracle_locs = self.train_data['locations'][oracle_idx]
                oracle_hrtfs = self.train_data['hrtfs'][oracle_idx]
                for loc, hrtf in zip(oracle_locs, oracle_hrtfs):
                    loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                    loc_key = tuple(np.round(loc_np, 2))
                    oracle_loc_map[loc_key] = hrtf

            # Normalize validation anthropometry if using normalization
            if anthro_mean is not None and anthro_std is not None:
                val_anthro_normalized = (val_anthro - anthro_mean) / anthro_std
            else:
                val_anthro_normalized = val_anthro

            # Find closest training subject by anthropometry
            anthro_distances = torch.cdist(
                val_anthro_normalized.unsqueeze(0),
                train_anthro_matrix_normalized
            ).squeeze()
            closest_train_idx = torch.argmin(anthro_distances).item()
            min_distance_normalized = anthro_distances[closest_train_idx].item()

            # Get the closest subject's locations and HRTFs
            closest_train_locs = self.train_data['locations'][closest_train_idx]
            closest_train_hrtfs = self.train_data['hrtfs'][closest_train_idx]

            # Build dictionary for fast lookup
            closest_subject_loc_map = {}
            for train_loc, train_hrtf in zip(closest_train_locs, closest_train_hrtfs):
                loc_np = train_loc.numpy() if isinstance(train_loc, torch.Tensor) else train_loc
                loc_key = tuple(np.round(loc_np, 2))
                closest_subject_loc_map[loc_key] = train_hrtf

            # Track per-location LSDs and locations for this subject
            subject_anthro_lsd, subject_pop_lsd, subject_oracle_lsd = [], [], []
            all_matched_locations = []
            oracle_worse_locations = []
            subject_locations_matched = 0

            # Evaluate each validation location
            for loc_idx, (loc, true_hrtf) in enumerate(zip(val_locs, val_hrtfs)):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                if (loc_np[0]< 0.0) or (loc_np[0]> 360.0):
                    continue
                loc_key = tuple(np.round(loc_np, 2))

                if loc_key in closest_subject_loc_map:
                    all_matched_locations.append(loc_np)

                    # Original prediction (closest anthropometry)
                    pred_hrtf_anthro = closest_subject_loc_map[loc_key]
                    lsd_anthro = self._compute_lsd(true_hrtf, pred_hrtf_anthro)
                    total_lsd += lsd_anthro
                    total_points += 1
                    subject_anthro_lsd.append(lsd_anthro)

                    # Get population average for this location
                    pop_avg_hrtf, lsd_pop = None, None
                    if loc_key in self.population_avg:
                        pop_avg_hrtf = self.population_avg[loc_key]
                        lsd_pop = self._compute_lsd(true_hrtf, pop_avg_hrtf)
                        pop_avg_total_lsd += lsd_pop
                        subject_pop_lsd.append(lsd_pop)

                    # Get oracle subject's HRTF for this location
                    oracle_hrtf, lsd_oracle = None, None
                    if loc_key in oracle_loc_map:
                        oracle_hrtf = oracle_loc_map[loc_key]
                        lsd_oracle = self._compute_lsd(true_hrtf, oracle_hrtf)
                        oracle_total_lsd += lsd_oracle
                        subject_oracle_lsd.append(lsd_oracle)

                    # NEW: Check if oracle is worse than population average
                    if lsd_oracle is not None and lsd_pop is not None and lsd_oracle > lsd_pop:
                        oracle_worse_locations.append(loc_np)

                    subject_locations_matched += 1

                    # Create individual HRTF plot for first few matches
                    if save_plots and plot_count < max_plots:
                        self._plot_individual_hrtf_comparison(
                            true_hrtf, pred_hrtf_anthro, pop_avg_hrtf, oracle_hrtf,
                            lsd_anthro, lsd_pop, lsd_oracle,
                            loc_key, val_idx, loc_idx, min_distance_normalized,
                            closest_train_idx, oracle_idx if oracle_info else None, oracle_info,
                            plot_dir
                        )
                        plot_count += 1
                else:
                    skipped_points += 1

            # NEW: Generate and save the summary plot for oracle performance
            if save_plots and oracle_worse_locations:
                self._plot_oracle_performance_map(
                    all_matched_locations, oracle_worse_locations, val_idx, plot_dir
                )

            # Compute per-subject averages and store results
            if subject_locations_matched > 0:
                avg_anthro = np.mean(subject_anthro_lsd) if subject_anthro_lsd else float('inf')
                avg_pop = np.mean(subject_pop_lsd) if subject_pop_lsd else float('inf')
                avg_oracle = np.mean(subject_oracle_lsd) if subject_oracle_lsd else float('inf')

                per_subject_results.append({
                    'val_idx': val_idx,
                    'locations_matched': subject_locations_matched,
                    'avg_anthro_lsd': avg_anthro,
                    'avg_pop_lsd': avg_pop,
                    'avg_oracle_lsd': avg_oracle,
                    'oracle_subject': oracle_info['subject_idx'] if oracle_info else None,
                    'anthro_subject': closest_train_idx
                })

                # Print detailed per-subject comparison
                print(f"\nVal Subject {val_idx} Summary:")
                print(f"  Locations matched: {subject_locations_matched}")
                print(f"  Closest Anthro (Subj {closest_train_idx}): LSD = {avg_anthro:.4f}")
                print(f"  Population Average: LSD = {avg_pop:.4f}")
                if oracle_info:
                    print(f"  Oracle (Subj {oracle_info['subject_idx']}): LSD = {avg_oracle:.4f}")
                print(f"  Improvement from Pop Avg to Oracle: {avg_pop - avg_oracle:.4f}")
                print(f"  Gap between Anthro and Oracle: {avg_anthro - avg_oracle:.4f}")
                print(f"  Found {len(oracle_worse_locations)} locations where oracle was worse than population average.")


        # Print overall summary
        # ... (rest of the summary printing code remains the same)
        print("\n" + "="*60)
        print("PER-SUBJECT ANALYSIS")
        print("="*60)

        # Compute statistics across subjects
        if per_subject_results:
            improvements = [r['avg_pop_lsd'] - r['avg_oracle_lsd'] for r in per_subject_results if r['avg_pop_lsd'] != float('inf') and r['avg_oracle_lsd'] != float('inf')]
            gaps = [r['avg_anthro_lsd'] - r['avg_oracle_lsd'] for r in per_subject_results if r['avg_anthro_lsd'] != float('inf') and r['avg_oracle_lsd'] != float('inf')]

            if improvements:
                print(f"\nAverage improvement from Pop Avg to Oracle: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
                print(f"Max improvement possible: {np.max(improvements):.4f}")
                print(f"Min improvement possible: {np.min(improvements):.4f}")
            if gaps:
                print(f"Average gap from Anthro to Oracle: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")


        # Compute and print average LSDs
        avg_lsd_anthro = total_lsd / total_points if total_points > 0 else float('inf')
        avg_lsd_pop = pop_avg_total_lsd / total_points if total_points > 0 else float('inf')
        avg_lsd_oracle = oracle_total_lsd / total_points if total_points > 0 else float('inf')

        print(f"\n=== OVERALL METHOD COMPARISON ===")
        print(f"Closest Anthropometry: LSD = {avg_lsd_anthro:.4f}")
        print(f"Population Average: LSD = {avg_lsd_pop:.4f}")
        print(f"Best Overall Match (Oracle): LSD = {avg_lsd_oracle:.4f}")
        print(f"Skipped {skipped_points} points without exact matches")

        return avg_lsd_anthro

    def _plot_oracle_performance_map(self, all_locations, worse_locations, val_idx, plot_dir):
        """
        NEW: Creates a scatter plot showing all measured locations and highlighting
        those where the oracle subject performed worse than the population average.
        """
        all_locs_np = np.array(all_locations)
        worse_locs_np = np.array(worse_locations)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot all matched locations as a background
        ax.scatter(all_locs_np[:, 0], all_locs_np[:, 1], c='lightgray', s=30,
                    alpha=0.6, label='All Matched Locations')

        # Highlight the locations where the oracle was worse
        if worse_locs_np.size > 0:
            ax.scatter(worse_locs_np[:, 0], worse_locs_np[:, 1], c='red', s=40,
                        edgecolor='black', alpha=0.8, label='Oracle Worse than Pop. Avg.')

        ax.set_title(f'Oracle Performance Analysis for Subject {val_idx}\n'
                     f'({len(worse_locations)} of {len(all_locations)} locations shown in red)')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_xlim(-30, 390)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()

        filename = f'oracle_worse_locations_val{val_idx}.png'
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=150)
        print(f"Saved oracle performance map to {filepath}")
        plt.close(fig)

    def _plot_individual_hrtf_comparison(self, true_hrtf, pred_hrtf_anthro, pop_avg_hrtf, oracle_hrtf,
                                        lsd_anthro, lsd_pop, lsd_oracle,
                                        loc_key, val_idx, loc_idx, min_distance_normalized,
                                        closest_train_idx, oracle_idx, oracle_info, plot_dir):
        """
        Refactored plotting logic for individual HRTF comparisons.
        """
        true_hrtf_np = true_hrtf.numpy() if isinstance(true_hrtf, torch.Tensor) else true_hrtf
        pred_hrtf_anthro_np = pred_hrtf_anthro.numpy() if isinstance(pred_hrtf_anthro, torch.Tensor) else pred_hrtf_anthro

        # Create frequency axis
        fs = 44100
        N_fft = 256
        frequencies_hz = np.arange(1, len(true_hrtf_np) + 1) * fs / N_fft

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: All HRTF Comparisons
        ax1 = axes[0]
        ax1.plot(frequencies_hz, true_hrtf_np, 'b-', linewidth=2.5, label='Target HRTF', alpha=0.9)
        ax1.plot(frequencies_hz, pred_hrtf_anthro_np, 'r--', linewidth=2, label=f'Closest Anthro (Subj {closest_train_idx}, LSD: {lsd_anthro:.3f})', alpha=0.8)

        if pop_avg_hrtf is not None:
            pop_avg_np = pop_avg_hrtf.numpy() if isinstance(pop_avg_hrtf, torch.Tensor) else pop_avg_hrtf
            ax1.plot(frequencies_hz, pop_avg_np, 'g-.', linewidth=2, label=f'Population Avg (LSD: {lsd_pop:.3f})', alpha=0.8)

        if oracle_hrtf is not None:
            oracle_hrtf_np = oracle_hrtf.numpy() if isinstance(oracle_hrtf, torch.Tensor) else oracle_hrtf
            ax1.plot(frequencies_hz, oracle_hrtf_np, 'm:', linewidth=2, label=f'Best Match Overall (Subj {oracle_idx}, LSD: {lsd_oracle:.3f})', alpha=0.8)

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)' if self.scale == 'log' else 'Magnitude')
        ax1.set_title(f'HRTF Comparison - Location: {loc_key}\nVal Subject {val_idx} | Anthro Distance: {min_distance_normalized:.3f}')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Errors for all methods
        ax2 = axes[1]
        error_anthro = true_hrtf_np - pred_hrtf_anthro_np
        ax2.plot(frequencies_hz, error_anthro, 'r-', linewidth=1.5, label='Closest Anthro Error', alpha=0.7)

        if pop_avg_hrtf is not None:
            pop_avg_np = pop_avg_hrtf.numpy() if isinstance(pop_avg_hrtf, torch.Tensor) else pop_avg_hrtf
            error_pop = true_hrtf_np - pop_avg_np
            ax2.plot(frequencies_hz, error_pop, 'g-', linewidth=1.5, label='Pop Avg Error', alpha=0.7)

        if oracle_hrtf is not None:
            oracle_hrtf_np = oracle_hrtf.numpy() if isinstance(oracle_hrtf, torch.Tensor) else oracle_hrtf
            error_oracle = true_hrtf_np - oracle_hrtf_np
            ax2.plot(frequencies_hz, error_oracle, 'm-', linewidth=1.5, label='Best Match Error', alpha=0.7)

        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Error (Target - Predicted)')
        ax2.set_title('Prediction Errors')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Summary text
        ax3 = axes[2]
        ax3.axis('off')
        summary_text = "Method Comparison:\n\n"
        summary_text += f"1. Closest Anthropometry (Subject {closest_train_idx}):\n"
        summary_text += f"   - LSD at this location: {lsd_anthro:.3f} dB\n"
        summary_text += f"   - Anthro distance: {min_distance_normalized:.3f}\n\n"

        if lsd_pop is not None:
            summary_text += f"2. Population Average:\n"
            summary_text += f"   - LSD at this location: {lsd_pop:.3f} dB\n"
            # This line might cause an error if loc_key is not in self.location_hrtf_map
            # It should be handled gracefully in a production environment.
            # For now, we assume it exists as in the original code.
            summary_text += f"   - Averaged over {len(self.location_hrtf_map.get(loc_key, []))} subjects\n\n"

        if oracle_info and oracle_idx is not None:
            summary_text += f"3. Best Overall Match (Subject {oracle_idx}):\n"
            summary_text += f"   - LSD at this location: {lsd_oracle:.3f} dB\n"
            summary_text += f"   - Avg LSD across {oracle_info['num_matches']} locations: {oracle_info['avg_lsd']:.3f} dB\n"

        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        filename = f'hrtf_comparison_val{val_idx}_loc{loc_idx}.png'
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()




    def _find_closest_hrtf_for_location(self, val_hrtf, val_loc_key):
        """
        Find the closest HRTF by LSD for a single validation point.
        Only searches at the exact same location.
        """
        min_lsd = float('inf')
        best_hrtf = None
        best_info = {}

        # Search through all training data for this specific location
        for train_idx, (train_locs, train_hrtfs) in enumerate(
            zip(self.train_data['locations'], self.train_data['hrtfs'])
        ):
            for train_loc_idx, (train_loc, train_hrtf) in enumerate(zip(train_locs, train_hrtfs)):
                train_loc_np = train_loc.numpy() if isinstance(train_loc, torch.Tensor) else train_loc
                train_loc_key = tuple(np.round(train_loc_np, 2))

                # Only consider exact location matches
                if train_loc_key == val_loc_key:
                    lsd = self._compute_lsd(val_hrtf, train_hrtf)
                    if lsd < min_lsd:
                        min_lsd = lsd
                        best_hrtf = train_hrtf
                        best_info = {
                            'subject_idx': train_idx,
                            'location_idx': train_loc_idx,
                            'location': train_loc_key,
                            'lsd': lsd,
                            'hrtf': train_hrtf
                        }

        return best_info if best_hrtf is not None else None

    def evaluate_closest_anthro_interpolated(self, method='linear'):
        """Baseline 2b: Closest anthropometry with interpolation."""
        if len(self.train_data['anthropometry']) == 0:
            print("No training data available")
            return float('inf')

        total_lsd = 0
        total_points = 0

        for val_anthro, val_locs, val_hrtfs in zip(
            self.val_data['anthropometry'], self.val_data['locations'], self.val_data['hrtfs']
        ):
            # Find closest training subject
            anthro_distances = torch.cdist(val_anthro.unsqueeze(0), self.train_anthro_matrix).squeeze()
            closest_idx = torch.argmin(anthro_distances).item()

            # Build interpolator for this subject
            subject_interpolator = self._build_subject_interpolator(closest_idx, method)
            if subject_interpolator is None:
                continue

            for loc, true_hrtf in zip(val_locs, val_hrtfs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                pred_hrtf = self._interpolate_hrtf(loc_np, subject_interpolator)
                if pred_hrtf is not None:
                    lsd = self._compute_lsd(true_hrtf, torch.tensor(pred_hrtf))
                    total_lsd += lsd
                    total_points += 1

        avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
        print(f"Closest Anthropometry (Interpolated-{method}): LSD = {avg_lsd:.4f}")
        return avg_lsd

    def evaluate_pca_regression(self, n_components=10, method='linear'):
        """Baseline 3: PCA + Ridge Regression with interpolation."""
        if len(self.train_data['anthropometry']) == 0 or len(self.train_data['hrtfs']) == 0:
            print("Insufficient training data for PCA regression")
            return float('inf')

        # Prepare training data
        X_train = []
        y_train = []

        for anthro, hrtfs in zip(self.train_data['anthropometry'], self.train_data['hrtfs']):
            X_train.append(anthro.numpy())
            # Use mean HRTF across all locations for this subject
            mean_hrtf = hrtfs.mean(dim=0).numpy()
            y_train.append(mean_hrtf)

        if len(X_train) == 0:
            return float('inf')

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Fit PCA on anthropometry
        n_components_actual = min(n_components, X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=n_components_actual)
        X_train_pca = pca.fit_transform(X_train)

        # Fit Ridge regression for each frequency bin
        ridge_models = []
        for freq_idx in range(y_train.shape[1]):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_pca, y_train[:, freq_idx])
            ridge_models.append(ridge)

        # Evaluate on validation set
        total_lsd = 0
        total_points = 0

        for val_anthro, val_locs, val_hrtfs in zip(
            self.val_data['anthropometry'], self.val_data['locations'], self.val_data['hrtfs']
        ):
            # Transform validation anthropometry
            val_anthro_pca = pca.transform(val_anthro.numpy().reshape(1, -1))

            # Predict base HRTF spectrum
            base_hrtf = np.array([model.predict(val_anthro_pca)[0] for model in ridge_models])

            # For location-specific adjustment, find closest training subject
            if len(self.train_anthro_matrix) > 0:
                anthro_distances = torch.cdist(val_anthro.unsqueeze(0), self.train_anthro_matrix).squeeze()
                closest_idx = torch.argmin(anthro_distances).item()

                # Build location adjustment interpolator
                subject_interpolator = self._build_subject_interpolator(closest_idx, method)

                for loc, true_hrtf in zip(val_locs, val_hrtfs):
                    loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                    # Get location-specific adjustment
                    location_hrtf = self._interpolate_hrtf(loc_np, subject_interpolator)

                    if location_hrtf is not None:
                        # Combine base prediction with location adjustment
                        pred_hrtf = 0.7 * base_hrtf + 0.3 * location_hrtf
                    else:
                        pred_hrtf = base_hrtf

                    lsd = self._compute_lsd(true_hrtf, torch.tensor(pred_hrtf))
                    total_lsd += lsd
                    total_points += 1

        avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
        print(f"PCA + Ridge Regression (n_components={n_components}, interp={method}): LSD = {avg_lsd:.4f}")
        return avg_lsd

    def evaluate_vbap_interpolation(self):
        """Evaluate using VBAP (Vector Base Amplitude Panning) interpolation."""
        if self.location_kdtree is None or len(self.all_train_locs_array) == 0:
            print("No training locations available for VBAP")
            return float('inf')

        total_lsd = 0
        total_points = 0

        for val_locs, val_hrtfs in zip(self.val_data['locations'], self.val_data['hrtfs']):
            for loc, true_hrtf in zip(val_locs, val_hrtfs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                pred_hrtf = self._vbap_interpolate(loc_np)
                if pred_hrtf is not None:
                    lsd = self._compute_lsd(true_hrtf, torch.tensor(pred_hrtf))
                    total_lsd += lsd
                    total_points += 1

        avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
        print(f"Population Average (VBAP): LSD = {avg_lsd:.4f}")
        return avg_lsd

    def _build_interpolator(self, method='linear'):
        """Build interpolator from all training data."""
        points = []
        values = []

        for locs, hrtfs in zip(self.train_data['locations'], self.train_data['hrtfs']):
            for loc, hrtf in zip(locs, hrtfs):
                loc_np = loc.numpy() if isinstance(loc, torch.Tensor) else loc
                hrtf_np = hrtf.numpy() if isinstance(hrtf, torch.Tensor) else hrtf
                points.append(loc_np)
                values.append(hrtf_np)

        if len(points) == 0:
            return None

        points = np.array(points)
        values = np.array(values)
        if not self.population_avg:
            return None

        # Points are the locations, values are the AVERAGED HRTFs
        points = np.array(list(self.population_avg.keys()))

        # Ensure values are stacked correctly
        # Note: self.population_avg values are tensors, need to stack them and convert to numpy
        values_list = [v.numpy() if isinstance(v, torch.Tensor) else v for v in self.population_avg.values()]
        values = np.array(values_list)


        if method == 'linear':
            interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
            nearest_interpolator = NearestNDInterpolator(points, values)
            return (interpolator, nearest_interpolator)
        elif method == 'nearest':
            return NearestNDInterpolator(points, values)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _build_subject_interpolator(self, subject_idx, method='linear'):
        """Build interpolator for a specific subject."""
        if subject_idx >= len(self.train_data['locations']):
            return None

        locs = self.train_data['locations'][subject_idx]
        hrtfs = self.train_data['hrtfs'][subject_idx]

        if len(locs) == 0:
            return None

        locs_np = locs.numpy() if isinstance(locs, torch.Tensor) else locs
        hrtfs_np = hrtfs.numpy() if isinstance(hrtfs, torch.Tensor) else hrtfs

        if method == 'linear':
            interpolator = LinearNDInterpolator(locs_np, hrtfs_np, fill_value=np.nan)
            nearest_interpolator = NearestNDInterpolator(locs_np, hrtfs_np)
            return (interpolator, nearest_interpolator)
        elif method == 'nearest':
            return NearestNDInterpolator(locs_np, hrtfs_np)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _interpolate_hrtf(self, location, interpolator):
        """Interpolate HRTF at given location."""
        if interpolator is None:
            return None

        if isinstance(interpolator, tuple):
            # Linear with nearest fallback
            linear_interp, nearest_interp = interpolator
            result = linear_interp(location)
            if np.isnan(result).any():
                result = nearest_interp(location)
        else:
            result = interpolator(location)
        return result

    def _vbap_interpolate(self, location, n_neighbors=3):
        """VBAP interpolation for spherical coordinates."""
        if self.location_kdtree is None or len(self.all_train_locs_array) == 0:
            return None

        # Convert location to spherical if needed
        azimuth, elevation = location

        # Convert to unit vector
        theta = np.radians(azimuth)
        phi = np.radians(90 - elevation)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        target_vec = np.array([x, y, z])

        # Find nearest neighbors
        k = min(n_neighbors, len(self.all_train_locs_array))
        distances, indices = self.location_kdtree.query(location, k=k)

        # Compute VBAP weights
        weights = []
        hrtfs = []

        for dist, idx in zip(distances, indices):
            subj_idx, loc_idx = self.loc_to_subject_idx[idx]

            # Convert training location to unit vector
            train_loc = self.all_train_locs_array[idx]
            az_train, el_train = train_loc
            theta_train = np.radians(az_train)
            phi_train = np.radians(90 - el_train)

            x_train = np.sin(phi_train) * np.cos(theta_train)
            y_train = np.sin(phi_train) * np.sin(theta_train)
            z_train = np.cos(phi_train)
            train_vec = np.array([x_train, y_train, z_train])

            # Compute weight based on angular distance
            cos_angle = np.dot(target_vec, train_vec)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            # VBAP weight
            weight = 1.0 / (angle + 1e-6)
            weights.append(weight)

            hrtf = self.train_data['hrtfs'][subj_idx][loc_idx]
            hrtf_np = hrtf.numpy() if isinstance(hrtf, torch.Tensor) else hrtf
            hrtfs.append(hrtf_np)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average
        pred_hrtf = np.zeros_like(hrtfs[0])
        for weight, hrtf in zip(weights, hrtfs):
            pred_hrtf += weight * hrtf

        return pred_hrtf

    def _compute_lsd(self, true_hrtf, pred_hrtf):
        """Compute Log-Spectral Distance."""
        if self.scale == 'log':
            lsd = torch.sqrt(torch.mean(torch.square(true_hrtf - pred_hrtf)))
        else:
            epsilon = 1e-9
            true_abs = torch.abs(true_hrtf) + epsilon
            pred_abs = torch.abs(pred_hrtf) + epsilon
            lsd = torch.sqrt(torch.mean(torch.square(20 * torch.log10(true_abs / pred_abs))))
        return lsd.item()

    def run_all_baselines(self, anthro_mean=None, anthro_std=None):
        """Run all baseline evaluations."""
        results = {}

        print("\n" + "="*60)
        print("Running Classical Baseline Evaluations")
        print("="*60 + "\n")

        # 1. Population Average
        print("1. Population Average Baselines:")
        print("-" * 40)
        results['pop_avg_exact'] = self.evaluate_population_average_exact()
        results['pop_avg_linear'] = self.evaluate_population_average_interpolated('linear')
        results['pop_avg_nearest'] = self.evaluate_population_average_interpolated('nearest')
        results['pop_avg_vbap'] = self.evaluate_vbap_interpolation()

        # 2. Closest Anthropometry
        print("\n2. Closest Anthropometry Baselines:")
        print("-" * 40)
        results['closest_anthro_exact'] = self.evaluate_closest_anthro_exact_location(save_plots=True,anthro_mean=anthro_mean, anthro_std=anthro_std, plot_dir="/export/mkrishn9/hrtf_field/experiments_test/baseline_results/hrtf_plots")
        results['closest_anthro_linear'] = self.evaluate_closest_anthro_interpolated('linear')
        results['closest_anthro_nearest'] = self.evaluate_closest_anthro_interpolated('nearest')

        # 3. PCA + Regression
        print("\n3. PCA + Ridge Regression Baselines:")
        print("-" * 40)
        for n_comp in [5, 10, 15]:
            results[f'pca_{n_comp}_linear'] = self.evaluate_pca_regression(n_comp, 'linear')
            results[f'pca_{n_comp}_nearest'] = self.evaluate_pca_regression(n_comp, 'nearest')

        return results


def get_dataset_locations(merged_dataset, dataset_names):
    """Extract unique locations for each dataset."""
    dataset_locations = defaultdict(list)

    for i, file_path in enumerate(merged_dataset.file_registry):
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match:
            continue

        dataset_name = match.group(1)
        if dataset_name not in dataset_names:
            continue

        locations, _, _, _ = merged_dataset[i]
        if isinstance(locations, torch.Tensor):
            locations = locations.numpy()
        dataset_locations[dataset_name].append(locations)

    unique_locations = {}
    for dataset_name, loc_list in dataset_locations.items():
        if not loc_list:
            continue
        all_locs = np.vstack(loc_list)
        unique_locs = np.unique(np.round(all_locs, decimals=2), axis=0)
        unique_locations[dataset_name] = unique_locs
        print(f"Dataset {dataset_name}: {len(unique_locs)} unique locations")

    return unique_locations


def select_holdout_locations(unique_locations, holdout_fraction=0.1, seed=42):
    """Select locations to hold out for each dataset."""
    np.random.seed(seed)
    holdout_indices = {}

    for dataset_name, locations in unique_locations.items():
        n_locations = len(locations)
        n_holdout = max(1, int(n_locations * holdout_fraction))

        all_indices = np.arange(n_locations)
        np.random.shuffle(all_indices)
        holdout_idx = all_indices[:n_holdout]

        holdout_indices[dataset_name] = holdout_idx
        print(f"Dataset {dataset_name}: Holding out {n_holdout}/{n_locations} locations")

        held_out_locs = locations[holdout_idx][:5]
        print(f"  Sample held-out locations: {held_out_locs.tolist()}")

    return holdout_indices


def get_location_indices_for_file(locations, target_locations, tolerance=0.01):
    """Find indices of target_locations within the given locations tensor."""
    indices = []
    locations_np = locations.numpy() if isinstance(locations, torch.Tensor) else locations

    for target_loc in target_locations:
        distances = np.sqrt(np.sum((locations_np - target_loc)**2, axis=1))
        matching_idx = np.where(distances < tolerance)[0]
        indices.extend(matching_idx.tolist())

    return sorted(list(set(indices)))


def merged_collate_fn(batch):
    """Collate function for DataLoader."""
    valid_batch = [item for item in batch if item is not None and all(i is not None for i in item)]
    if not valid_batch:
        return None

    locations, hrtfs, anthros, names = zip(*valid_batch)

    padded_locs = torch.nn.utils.rnn.pad_sequence(locations, batch_first=True, padding_value=0.0)
    padded_hrtfs = torch.nn.utils.rnn.pad_sequence(hrtfs, batch_first=True, padding_value=0.0)

    max_num_loc = padded_locs.shape[1]
    masks = torch.zeros((len(locations), max_num_loc, 1), dtype=torch.float32)
    for i, loc_tensor in enumerate(locations):
        masks[i, :loc_tensor.shape[0], :] = 1.0

    collated_anthros = torch.stack(anthros, dim=0)

    return padded_locs, padded_hrtfs, masks, collated_anthros, list(names)


def evaluate_neural_network(model_path, val_loader, device, anthro_mean, anthro_std, scale='log'):
    """Evaluate the neural network model."""
    print("\nEvaluating Neural Network Model:")
    print("-" * 40)

    checkpoint = torch.load(model_path, map_location=device)
    model = CondHRTFNetwork(
        d_anthro_in=10,
        d_inr_out=92,
        d_latent=256,
        d_inr_hidden=256,
        n_inr_layers=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_lsd = 0
    total_points = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            locations, target_hrtfs, masks, anthropometry, _ = batch
            locations = locations.to(device)
            target_hrtfs = target_hrtfs.to(device)
            masks = masks.to(device)
            anthropometry = anthropometry.to(device)

            anthropometry_norm = (anthropometry - anthro_mean) / anthro_std

            B, N_loc, _ = locations.shape
            anthro_dim = anthropometry_norm.shape[-1]
            num_freq = target_hrtfs.shape[-1]

            anthro_expanded = anthropometry_norm.unsqueeze(1).expand(B, N_loc, anthro_dim)
            locations_flat = locations.reshape(-1, 2)
            anthro_flat = anthro_expanded.reshape(-1, anthro_dim)

            predicted_hrtfs_flat = model(locations_flat, anthro_flat)
            predicted_hrtfs = predicted_hrtfs_flat.reshape(B, N_loc, num_freq)

            if scale == 'log':
                lsd_elements_sq = torch.square(predicted_hrtfs - target_hrtfs)
            else:
                epsilon = 1e-9
                pred_abs = torch.abs(predicted_hrtfs) + epsilon
                gt_abs = torch.abs(target_hrtfs) + epsilon
                lsd_elements_sq = torch.square(20 * torch.log10(gt_abs / pred_abs))

            masked_lsd_sq = lsd_elements_sq * masks

            batch_lsd = torch.sqrt(torch.sum(masked_lsd_sq) / (torch.sum(masks) * num_freq))
            total_lsd += batch_lsd.item() * torch.sum(masks).item()
            total_points += torch.sum(masks).item()

    avg_lsd = total_lsd / total_points if total_points > 0 else float('inf')
    print(f"Neural Network: LSD = {avg_lsd:.4f}")
    return avg_lsd


def main(args):
    """Main evaluation function with location holdout."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset...")
    all_available_datasets = ["hutubs", "cipic", "ari", "scut", "chedar"]
    if args.include_datasets.lower() == "all":
        datasets_to_load = all_available_datasets
    else:
        datasets_to_load = [name.strip() for name in args.include_datasets.split(',')]

    print(f"Including datasets: {datasets_to_load}")

    merged_dataset = MergedHRTFDataset(
        dataset_names=datasets_to_load,
        preprocessed_dir=args.data_path,
        augment=False,
        scale=args.scale
    )

    # Parse which datasets should have location holdout
    if args.location_holdout_datasets.lower() == "all":
        holdout_datasets = datasets_to_load
    elif args.location_holdout_datasets.lower() == "none":
        holdout_datasets = []
    else:
        holdout_datasets = [name.strip() for name in args.location_holdout_datasets.split(',')]

    print(f"Applying location holdout to datasets: {holdout_datasets}")

    # Get unique locations and select holdout
    unique_locations = get_dataset_locations(merged_dataset, holdout_datasets)
    holdout_location_indices = select_holdout_locations(
        unique_locations,
        holdout_fraction=args.location_holdout_fraction,
        seed=args.seed
    )

    # Create location filters
    holdout_location_filters = {}
    for dataset_name, holdout_idx in holdout_location_indices.items():
        holdout_locs = unique_locations[dataset_name][holdout_idx]
        holdout_location_filters[dataset_name] = holdout_locs

    # Subject-aware splitting
    registry_to_subject_id = {}
    unique_subject_keys = []

    for i, file_path in enumerate(merged_dataset.file_registry):
        base_name = os.path.basename(file_path)
        match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
        if not match:
            continue
        dataset_name, file_num = match.group(1), int(match.group(2))
        subject_id_in_dataset = file_num // 2
        subject_key = (dataset_name, subject_id_in_dataset)

        if subject_key not in unique_subject_keys:
            unique_subject_keys.append(subject_key)

        subject_idx = unique_subject_keys.index(subject_key)
        registry_to_subject_id[i] = subject_idx

    indices = np.arange(len(merged_dataset))
    groups = np.array([registry_to_subject_id[i] for i in indices])

    # Use first fold for evaluation
    kfold = GroupKFold(n_splits=5)
    splits_iterator = iter(kfold.split(indices, groups=groups))
    train_indices, val_indices = next(splits_iterator)
    train_indices, val_indices = next(splits_iterator)
    train_indices, val_indices = next(splits_iterator)
    train_indices, val_indices = next(splits_iterator)

    # Convert holdout locations to indices for each file
    location_index_filters = {}
    for dataset_name, holdout_locs in holdout_location_filters.items():
        # Find a sample file from this dataset
        sample_idx = None
        for idx in train_indices:
            file_path = merged_dataset.file_registry[idx]
            if dataset_name in os.path.basename(file_path):
                sample_idx = idx
                break

        if sample_idx is not None:
            locations, _, _, _ = merged_dataset[sample_idx]
            location_indices = get_location_indices_for_file(locations, holdout_locs)
            location_index_filters[dataset_name] = location_indices

    # Create filtered datasets
    train_dataset = LocationFilteredDataset(
        merged_dataset,
        train_indices,
        location_index_filters,
        mode='exclude'  # Exclude holdout locations from training
    )

    # For validation: combine held-out locations from training subjects
    # and all locations from validation subjects
    train_val_dataset = LocationFilteredDataset(
        merged_dataset,
        train_indices,
        location_index_filters,
        mode='include'  # Only holdout locations from training subjects
    )

    val_dataset = LocationFilteredDataset(
        merged_dataset,
        val_indices,
        {},  # No filtering for validation subjects
        mode='all'
    )

    # Combine validation datasets
    if holdout_datasets:
        print("Combining validation subjects and held-out locations from training subjects.")
        combined_val_dataset = ConcatDataset([train_val_dataset, val_dataset])
    else:
        print("Using standard validation split (no location holdout).")
        # When no location holdout, the validation set is just the validation subjects.
        combined_val_dataset = val_dataset

    combined_val_dataset = val_dataset

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(combined_val_dataset)}")

    anthro_mean, anthro_std = calculate_anthro_stats(train_dataset)
    print(f"Anthro Mean (sample): {anthro_mean[:5].numpy()}")
    print(f"Anthro Std (sample): {anthro_std[:5].numpy()}")

    # Initialize evaluator with filtered datasets
    evaluator = HRTFBaselineEvaluator(train_dataset, combined_val_dataset, device, scale=args.scale)

    # Run all baselines
    baseline_results = evaluator.run_all_baselines(anthro_mean=anthro_mean,anthro_std=anthro_std)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (with Location Holdout)")
    print("="*60)

    sorted_results = sorted(baseline_results.items(), key=lambda x: x[1])

    print("\nMethods ranked by LSD (lower is better):")
    print("-" * 40)
    for i, (method, lsd) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {method:30s}: {lsd:.4f}")

    # Save results
    if args.save_results:
        results_file = os.path.join(args.save_dir, 'baseline_evaluation_results_location_holdout.json')
        os.makedirs(args.save_dir, exist_ok=True)

        results_with_metadata = {
            'results': baseline_results,
            'config': {
                'location_holdout_fraction': args.location_holdout_fraction,
                'location_holdout_datasets': holdout_datasets,
                'datasets': datasets_to_load,
                'scale': args.scale,
                'seed': args.seed
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        print(f"\nResults saved to {results_file}")

    return baseline_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRTF Classical Baselines Evaluation with Location Holdout")

    # Path arguments
    parser.add_argument('--data_path', type=str,
                        default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common",
                        help='Path to preprocessed data')
    parser.add_argument('--save_dir', type=str,
                        default='/export/mkrishn9/hrtf_field/experiments_test/baseline_results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    # Dataset arguments
    parser.add_argument('--include_datasets', type=str, default='hutubs,cipic,axd',
                        help="Comma-separated list of datasets to use. 'all' uses everything.")

    # Location holdout arguments
    parser.add_argument('--location_holdout_datasets', type=str, default='hutubs,axd',
                        help="Comma-separated list of datasets to apply location holdout. "
                             "'all' applies to all datasets, 'none' disables location holdout.")
    parser.add_argument('--location_holdout_fraction', type=float, default=0.3,
                        help="Fraction of locations to hold out for validation (default: 0.1)")


    # Other arguments
    parser.add_argument('--scale', type=str, default='log', help='Scale (log or linear)')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--nn_checkpoint', type=str, default=None,
                        help='Path to neural network checkpoint for comparison')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to file')

    args = parser.parse_args()

    print("="*60)
    print("HRTF BASELINES EVALUATION WITH LOCATION HOLDOUT")
    print("="*60)
    print(f"Location holdout fraction: {args.location_holdout_fraction}")
    print(f"Datasets with location holdout: {args.location_holdout_datasets}")
    print("-"*60)

    results = main(args)

# import argparse
# import os
# import torch
# import numpy as np
# import re
# import random
# from torch.utils.data import Dataset, DataLoader, Subset
# from scipy.interpolate import griddata
# from tqdm import tqdm

# from HRTFdatasets import MergedHRTFDataset


# def calculate_anthro_stats(dataset_subset):
#     """
#     Calculates mean and std dev for anthropometry across a dataset subset.
#     """
#     print("Calculating anthropometry statistics on the training set...")
#     all_anthro_tensors = []
#     # Using tqdm for a progress bar as this can take a moment
#     for i in tqdm(range(len(dataset_subset)), desc="Analyzing train set"):
#         try:
#             # We only need the anthropometry tensor for this calculation
#             _, _, anthro_tensor, _ = dataset_subset.dataset[dataset_subset.indices[i]]
#             if anthro_tensor is not None:
#                 all_anthro_tensors.append(anthro_tensor)
#         except Exception as e:
#             print(f"Warning: Could not retrieve item at index {dataset_subset.indices[i]}: {e}. Skipping.")
#             continue

#     if not all_anthro_tensors:
#         raise ValueError("No valid anthropometric data found in the training set.")

#     all_anthro_stacked = torch.stack(all_anthro_tensors, dim=0)
#     mean = torch.mean(all_anthro_stacked, dim=0, dtype=torch.float32)
#     std = torch.std(all_anthro_stacked, dim=0)
#     std = torch.clamp(std, min=1e-8) # Avoid division by zero
#     return mean, std

# def normalize_anthro(anthro_tensor, mean, std):
#     """Applies Z-score normalization."""
#     return (anthro_tensor - mean) / std

# def calculate_lsd(pred_hrtf, true_hrtf):
#     """
#     Calculates the Log-Spectral Distance (LSD) in dB.
#     Assumes inputs are on a log-magnitude scale.
#     """

#     pred_hrtf = pred_hrtf.float()
#     true_hrtf = true_hrtf.float()
#     error_sq = (50*pred_hrtf - 50*true_hrtf)**2
#     lsd_per_point = torch.sqrt(torch.mean(error_sq, dim=-1))
#     return torch.mean(lsd_per_point)

# # --- Baseline evaluation: (1) Find closest-match anthropometry
# # (2) Interpolate between given spatial coordinates for that anthropometry for HRTF value at query location
# #  ---

# def evaluate_baseline(train_subset, val_subset, anthro_mean, anthro_std, device):
#     """
#     Evaluates the baseline nearest-neighbor interpolator.

#     For each validation sample:
#     1. Finds the closest anthropometry in the entire training set.
#     2. Uses that subject's HRTF data.
#     3. Interpolates to find the HRTF at the validation sample's locations.
#     4. Calculates the error (LSD).
#     """
#     anthro_mean = anthro_mean.to(device)
#     anthro_std = anthro_std.to(device)

#     # 1. Cache training data for efficient searching.
#     # We store the normalized anthropometry and the full HRTF data for each training subject.
#     print("Caching training data for nearest-neighbor search...")
#     train_data_cache = []

#     # We create a map to handle subjects with left/right ears to avoid redundant processing
#     processed_train_subjects = {}
#     for i in tqdm(range(len(train_subset)), desc="Caching train data"):
#         idx = train_subset.indices[i]
#         # Get subject key to avoid adding both left and right ears of the same person
#         file_path = train_subset.dataset.file_registry[idx]
#         base_name = os.path.basename(file_path)
#         match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
#         if not match: continue
#         dataset_name, file_num = match.group(1), int(match.group(2))
#         subject_key = (dataset_name, file_num // 2)

#         if subject_key in processed_train_subjects:
#             continue

#         locs, hrtfs, anthro, _ = train_subset.dataset[idx]

#         if locs is not None and anthro is not None and locs.shape[0] > 0:
#             train_data_cache.append({
#                 'norm_anthro': normalize_anthro(anthro, anthro_mean.cpu(), anthro_std.cpu()).to(device),
#                 'locations': locs.numpy(), # Scipy works with numpy arrays
#                 'hrtfs': hrtfs.numpy()
#             })
#             processed_train_subjects[subject_key] = True

#     print(f"Cached {len(train_data_cache)} unique training subjects.")

#     # 2. Iterate through validation set and evaluate
#     total_lsd = 0.0
#     num_val_samples = 0

#     processed_val_subjects = {}
#     print("Evaluating validation samples...")
#     for i in tqdm(range(len(val_subset)), desc="Evaluating validation set"):
#         idx = val_subset.indices[i]

#         # Get subject key to process each validation subject
#         file_path = val_subset.dataset.file_registry[idx]
#         base_name = os.path.basename(file_path)
#         match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
#         if not match: continue
#         dataset_name, file_num = match.group(1), int(match.group(2))
#         subject_key = (dataset_name, file_num // 2)

#         if subject_key in processed_val_subjects:
#             continue

#         val_locs, val_hrtfs, val_anthro, _ = val_subset.dataset[idx]

#         if val_locs is None or val_anthro is None or val_locs.shape[0] == 0:
#             continue

#         val_anthro_norm = normalize_anthro(val_anthro, anthro_mean.cpu(), anthro_std.cpu()).to(device)

#         # Find the closest neighbor in the training cache
#         min_dist = float('inf')
#         closest_neighbor_data = None
#         for train_subject in train_data_cache:
#             dist = torch.linalg.norm(val_anthro_norm - train_subject['norm_anthro'])
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_neighbor_data = train_subject

#         # 2b. Interpolate using the closest neighbor's data
#         # `griddata` performs the interpolation.
#         predicted_hrtfs_np = griddata(
#             points=closest_neighbor_data['locations'], # Known points from train subject
#             values=closest_neighbor_data['hrtfs'],     # Known values from train subject
#             xi=val_locs.numpy(),                       # Points to predict for val subject
#             method='linear',                           # 'linear' is bilinear for 2D
#             fill_value=0.0                             # Fill with 0 for points outside convex hull
#         )


#         predicted_hrtfs = torch.from_numpy(predicted_hrtfs_np)

#         # 2c. Calculate and accumulate the error
#         lsd = calculate_lsd(predicted_hrtfs, val_hrtfs)
#         total_lsd += lsd.item()
#         num_val_samples += 1
#         processed_val_subjects[subject_key] = True

#     # 3. Calculate and return the average LSD
#     average_lsd = total_lsd / num_val_samples if num_val_samples > 0 else float('inf')
#     return average_lsd


# def main(args):

#     DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {DEVICE}")


#     print("Loading dataset...")
#     all_available_datasets = ["hutubs", "cipic", "ari", "scut", "chedar"]
#     datasets_to_load = all_available_datasets if args.include_datasets.lower() == "all" else [name.strip() for name in args.include_datasets.split(',')]

#     merged_dataset = MergedHRTFDataset(
#         dataset_names=datasets_to_load,
#         preprocessed_dir=args.data_path,
#         scale="log"
#     )

#     subjects = {}
#     print("Mapping subjects for train/validation split...")
#     print("Mapping subjects to registry indices for splitting...")
#     for i, file_path in enumerate(merged_dataset.file_registry):
#         base_name = os.path.basename(file_path)
#         match = re.match(r"([a-zA-Z0-9]+)_(\d+).pkl", base_name)
#         if not match: continue
#         dataset_name, file_num = match.group(1), int(match.group(2))
#         subject_id_in_dataset = file_num // 2
#         subject_key = (dataset_name, subject_id_in_dataset)
#         if subject_key not in subjects: subjects[subject_key] = []
#         subjects[subject_key].append(i)

#     # SEED = 0
#     # torch.manual_seed(SEED)
#     # np.random.seed(SEED)
#     # random.seed(SEED)
#     # torch.cuda.manual_seed_all(SEED)

#     ## Not using Chedar in valuation set because it is numerically simulated data that would be easy to perform well on
#     chedar_subjects = []
#     other_subjects = []
#     for subject_key in subjects.keys():
#         if subject_key[0] == 'chedar':
#             chedar_subjects.append(subject_key)
#         else:
#             other_subjects.append(subject_key)

#     print(f"Found {len(chedar_subjects)} 'chedar' subjects and {len(other_subjects)} subjects from other datasets.")

#     # Perform the validation split ONLY on the non-chedar subjects
#     random.shuffle(other_subjects)
#     # Note: val_split is now applied to the count of 'other' subjects, not the total.
#     num_val_subjects = int(len(other_subjects) * args.val_split)
#     val_subject_keys = other_subjects[:num_val_subjects]

#     # The training set includes the remaining 'other' subjects AND all 'chedar' subjects
#     train_other_subject_keys = other_subjects[num_val_subjects:]
#     train_subject_keys = train_other_subject_keys + chedar_subjects

#     train_indices = [idx for key in train_subject_keys for idx in subjects[key]]
#     val_indices = [idx for key in val_subject_keys for idx in subjects[key]]

#     total_subjects = len(chedar_subjects) + len(other_subjects)

#     print(f"Total unique subjects: {total_subjects}, Train: {len(train_subject_keys)}, Val: {len(val_subject_keys)}")

#     train_subset = Subset(merged_dataset, train_indices)
#     val_subset = Subset(merged_dataset, val_indices)

#     # --- Calculate Normalization Stats ---
#     try:
#         anthro_mean, anthro_std = calculate_anthro_stats(train_subset)
#         print("Successfully calculated anthropometry statistics.")
#     except ValueError as e:
#         print(f"Error: {e}")
#         return

#     # --- Run Baseline Evaluation ---
#     avg_lsd_error = evaluate_baseline(train_subset, val_subset, anthro_mean, anthro_std, DEVICE)

#     print("\n=============================================")
#     print("      Baseline Evaluation Finished")
#     print("=============================================")
#     print(f"Method: Nearest-Neighbor Anthropometry + Bilinear Interpolation")
#     print(f"Average Log-Spectral Distance (LSD) on Validation Set: {avg_lsd_error:.4f} dB")
#     print("=============================================")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="HRTF baseline prediction using nearest-neighbor interpolation.")

#     parser.add_argument('--data_path', type=str, default="/export/mkrishn9/hrtf_field/preprocessed_hrirs_common", help='Path to preprocessed data')
#     parser.add_argument('--gpu', type=int, default=5, help='GPU device ID to use')
#     parser.add_argument('--include_datasets', type=str, default="all", help="Comma-separated list of datasets to use. 'all' uses everything.")
#     parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of subjects to use for validation')

#     args = parser.parse_args()
#     main(args)