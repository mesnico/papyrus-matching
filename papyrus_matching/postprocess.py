import os
import h5py
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from skimage import io
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import warnings
import pandas as pd

def compute_positions_and_scores(grid, t_relatives_in, scores_in):
    """
    Calculates the mean score for each translation and interpolates
    onto a grid.
    
    Args:
        grid (np.array): The common (X, 2) grid for interpolation.
        t_relatives_in (np.array): (M, 2) array of *unique* translation positions.
        scores_in (list): List of M arrays, where scores_in[i] contains
                          patch scores for translation t_relatives_in[i].
                          
    Returns:
        np.array: Interpolated scores on the grid.
    """
    pos_l = []
    scores_l = []
    
    # zip is now correct, as both inputs have length M (num_translations)
    for t_rel, scores in zip(t_relatives_in, scores_in):
        # Handle case where a translation might have no patches/scores
        if scores.size > 0:
            score = scores.mean().item()
            pos_l.append(t_rel)
            scores_l.append(score)

    if not pos_l:
        # No data to interpolate, return a grid of zeros
        return np.zeros(grid.shape[0])

    # interpolate
    interp_l_scores = griddata(
        pos_l,    
        scores_l, 
        grid,
        method='linear', 
        fill_value=0)

    return interp_l_scores

def merge_pair(recto_path, verso_path, output_path):
    """
    Merges a single recto/verso HDF5 pair.
    """
    try:
        with h5py.File(recto_path, 'r') as f_recto, h5py.File(verso_path, 'r') as f_verso:
            
            # --- 1. Load Data and Attributes ---
            
            # Recto data
            scores_recto = f_recto['scores'][:]
            t_ids_recto = f_recto['translation_ids'][:]
            t_rel_recto = f_recto['t_relatives'][:]
            a_coords_recto = f_recto['a_coords'][:]
            b_coords_recto = f_recto['b_coords'][:]
            attrs_recto = dict(f_recto.attrs)
            
            # Verso data
            scores_verso = f_verso['scores'][:]
            t_ids_verso = f_verso['translation_ids'][:]
            t_rel_verso = f_verso['t_relatives'][:]
            a_coords_verso = f_verso['a_coords'][:]
            b_coords_verso = f_verso['b_coords'][:]
            attrs_verso = dict(f_verso.attrs)

            # --- 1b. Correctly Group Data by Translation ID ---
            
            # --- RECTO GROUPING ---
            df_recto = pd.DataFrame({
                'score': scores_recto,
                't_id': t_ids_recto,
                't_rel_y': t_rel_recto[:, 0],
                't_rel_x': t_rel_recto[:, 1]
            })
            grouped_recto = df_recto.groupby('t_id')
            
            # Get the list of score arrays
            grouped_scores_list_r = grouped_recto['score'].apply(list).tolist()
            grouped_scores_recto = [np.array(s) for s in grouped_scores_list_r]
            
            # Get the unique translation position for each group
            unique_t_rel_df_r = grouped_recto[['t_rel_y', 't_rel_x']].first()
            unique_t_rel_recto = unique_t_rel_df_r.to_numpy()

            # --- VERSO GROUPING ---
            df_verso = pd.DataFrame({
                'score': scores_verso,
                't_id': t_ids_verso,
                't_rel_y': t_rel_verso[:, 0],
                't_rel_x': t_rel_verso[:, 1]
            })
            grouped_verso = df_verso.groupby('t_id')

            grouped_scores_list_v = grouped_verso['score'].apply(list).tolist()
            grouped_scores_verso = [np.array(s) for s in grouped_scores_list_v]
            
            unique_t_rel_df_v = grouped_verso[['t_rel_y', 't_rel_x']].first()
            unique_t_rel_verso = unique_t_rel_df_v.to_numpy()
            
            # --- 2. Create Common Grid ---
            
            # Get image size from fragment_a attribute
            fragment_a_path = attrs_recto['fragment_a']
            try:
                image = io.imread(fragment_a_path)
            except FileNotFoundError:
                print(f"  Warning: Could not find image {fragment_a_path}. "
                      "Using fragment_b path.")
                try:
                    fragment_b_path = attrs_recto['fragment_b']
                    image = io.imread(fragment_b_path)
                except FileNotFoundError:
                     print(f"  Error: Could not find fragment images. Skipping pair.")
                     return
                     
            image_size = image.shape[:2]
            patch_size = (64, 64) # As per your logic
            samples = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

            # Combine t_relatives to find global min/max
            # Use the unique, grouped t_rel arrays
            if not unique_t_rel_recto.size and not unique_t_rel_verso.size:
                print(f"  Warning: No translation data in either file. Skipping.")
                return
                
            t_relatives_all = np.vstack((unique_t_rel_recto, unique_t_rel_verso))
            
            min_y = np.min(t_relatives_all[:, 0])
            max_y = np.max(t_relatives_all[:, 0])
            min_x = np.min(t_relatives_all[:, 1])
            max_x = np.max(t_relatives_all[:, 1])
            
            # Handle edge case where all translations are at the same point
            if min_y == max_y: max_y += 1e-6
            if min_x == max_x: max_x += 1e-6

            grid_x_coords = np.linspace(min_x, max_x, samples[1])
            grid_y_coords = np.linspace(min_y, max_y, samples[0])
            grid_y, grid_x = np.meshgrid(grid_y_coords, grid_x_coords, indexing='ij')
            grid = np.vstack([grid_y.ravel(), grid_x.ravel()]).T
            
            # --- 4. Interpolate ---
            
            # Pass the unique t_relatives and grouped scores
            scores_l = compute_positions_and_scores(grid, unique_t_rel_recto, grouped_scores_recto)
            scores_grid = scores_l.reshape((samples[0], samples[1]))
            
            scores_l_back = compute_positions_and_scores(grid, unique_t_rel_verso, grouped_scores_verso)
            scores_grid_back = scores_l_back.reshape((samples[0], samples[1]))
            
            # Flip for verso to match recto
            scores_grid_back = scores_grid_back[:, ::-1]
            
            sum_scores_grid = scores_grid + scores_grid_back
            
            # --- 5. Log Discrepancy & Top K ---
            
            discrepancy_recto_verso = np.abs(scores_grid - scores_grid_back)
            # Only average non-zero discrepancy (where interpolation happened)
            non_zero_mask = (scores_grid != 0) | (scores_grid_back != 0)
            if np.any(non_zero_mask):
                discrepancy_mean = discrepancy_recto_verso[non_zero_mask].mean()
            else:
                discrepancy_mean = 0.0
            
            print(f"  Discrepancy: {discrepancy_mean:.4f}")

            # Find top-k
            k = 5
            flat_indices = np.argsort(sum_scores_grid.ravel())[-k:][::-1]
            y_indices, x_indices = np.unravel_index(flat_indices, sum_scores_grid.shape)
            
            print("  Top 5 Scores:")
            for y_idx, x_idx in zip(y_indices, x_indices):
                score = sum_scores_grid[y_idx, x_idx]
                if score == 0: continue # Skip zero-score maximums
                x_pos = grid_x_coords[x_idx]
                y_pos = grid_y_coords[y_idx]
                # Angle calculation from your snippet
                angle = np.degrees(np.arctan2(x_pos, y_pos))
                print(f"    Pos (y,x): ({y_pos:.2f}, {x_pos:.2f}), "
                      f"Score: {score:.4f}, Angle: {angle:.2f}")

            # --- 6. Save Merged HDF5 ---
            
            # Logic 6a: Quantize original t_relatives -> grid (from last request)
            if grid.shape[0] > 0:
                grid_tree = cKDTree(grid)
                
                # Find nearest grid point for each *original* t_relative point
                # (t_rel_recto is the raw, un-grouped array)
                dist_r, indices_r = grid_tree.query(t_rel_recto)
                t_rel_recto_quantized = grid[indices_r]
                
                dist_v, indices_v = grid_tree.query(t_rel_verso)
                t_rel_verso_quantized = grid[indices_v]
            else:
                # Handle case with no grid (e.g., no data)
                t_rel_recto_quantized = np.array([])
                t_rel_verso_quantized = np.array([])
            
            # Map grid -> original t_relatives
            # Create a reverse mapping: for each grid point, find the
            # translation_id of the *nearest* original point.
            
            # Initialize empty arrays
            grid_to_tid_recto = np.zeros(grid.shape[0], dtype=np.int32)
            grid_to_tid_verso = np.zeros(grid.shape[0], dtype=np.int32)
            
            # Build tree from original t_relatives and query with grid
            if t_rel_recto.size > 0 and grid.shape[0] > 0:
                recto_tree = cKDTree(t_rel_recto)
                # Find nearest original t_relative point for each grid point
                dist_r, indices_r = recto_tree.query(grid)
                # Get the translation_id for that original point
                grid_to_tid_recto = t_ids_recto[indices_r]

            if t_rel_verso.size > 0 and grid.shape[0] > 0:
                verso_tree = cKDTree(t_rel_verso)
                # Find nearest original t_relative point for each grid point
                dist_v, indices_v = verso_tree.query(grid)
                # Get the translation_id for that original point
                grid_to_tid_verso = t_ids_verso[indices_v]
            
            with h5py.File(output_path, 'w') as f_out:
                # Add new merged data
                f_out.create_dataset('grid_shape', data=np.array(samples))
                f_out.create_dataset('grid', data=grid)
                f_out.create_dataset('scores_grid', data=scores_grid.ravel())
                f_out.create_dataset('scores_grid_back', data=scores_grid_back.ravel())
                f_out.create_dataset('sum_scores_grid', data=sum_scores_grid.ravel())
                
                # Add the quantized datasets (t_rel -> grid)
                f_out.create_dataset('t_relatives_recto_quantized', data=t_rel_recto_quantized)
                f_out.create_dataset('t_relatives_verso_quantized', data=t_rel_verso_quantized)
                
                # Add the new reverse-mapping datasets (grid -> t_id)
                f_out.create_dataset('grid_to_translation_id_recto', data=grid_to_tid_recto)
                f_out.create_dataset('grid_to_translation_id_verso', data=grid_to_tid_verso)
                
                # Add original recto data
                f_out.create_dataset('scores_recto', data=scores_recto)
                f_out.create_dataset('translation_ids_recto', data=t_ids_recto)
                f_out.create_dataset('t_relatives_recto', data=t_rel_recto)
                f_out.create_dataset('a_coords_recto', data=a_coords_recto)
                f_out.create_dataset('b_coords_recto', data=b_coords_recto)
                
                # Add original verso data
                f_out.create_dataset('scores_verso', data=scores_verso)
                f_out.create_dataset('translation_ids_verso', data=t_ids_verso)
                f_out.create_dataset('t_relatives_verso', data=t_rel_verso)
                f_out.create_dataset('a_coords_verso', data=a_coords_verso)
                f_out.create_dataset('b_coords_verso', data=b_coords_verso)
                
                # Add attributes, renaming to avoid collisions
                for k, v in attrs_recto.items():
                    f_out.attrs[f'{k}_recto'] = v
                for k, v in attrs_verso.items():
                    f_out.attrs[f'{k}_verso'] = v

    except Exception as e:
        print(f"Failed to process pair {Path(recto_path).name}: {e}")

def main():
    BASE_DIR = Path("results")
    RECTO_DIR = BASE_DIR / "recto"
    VERSO_DIR = BASE_DIR / "verso"
    MERGED_DIR = BASE_DIR / "merged"
    
    # Create output directory
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all HDF5 files in the recto directory
    recto_files = glob.glob(str(RECTO_DIR / "*.hdf5"))
    
    # Get all available verso files first
    verso_files_dict = {f.name: f for f in VERSO_DIR.glob("*.hdf5")}
    
    if not recto_files:
        print(f"Error: No .hdf5 files found in {RECTO_DIR}")
        return
    
    if not verso_files_dict:
        print(f"Error: No .hdf5 files found in {VERSO_DIR}")
        return

    print(f"Found {len(recto_files)} recto files. Starting merge...")
    
    # Suppress warnings from skimage about EXIF data
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
    
    for recto_path_str in tqdm(recto_files, desc="Merging pairs"):
        recto_path = Path(recto_path_str)
        filename = recto_path.name
        base_name = filename.replace('.hdf5', '')
        
        # Parse the recto filename
        parts = base_name.split('__')
        if len(parts) != 2:
            print(f"Warning: Skipping {filename} (malformed name, expected 'a__b').")
            continue
        frag_a, frag_b = parts
        
        # Search for the matching verso file
        verso_path = None
        found_v_name = None
        for v_name, v_path in verso_files_dict.items():
            v_base = v_name.replace('.hdf5', '')
            v_parts = v_base.split('__')
            if len(v_parts) != 2: continue
            v_frag_a, v_frag_b = v_parts
            
            # Check for the pattern: fragB...__fragA...
            if v_frag_a.startswith(frag_a) and v_frag_b.startswith(frag_b):
                verso_path = v_path
                found_v_name = v_name
                break # Found our match
        
        if verso_path is None:
            print(f"Warning: No matching verso file found for {filename}. Skipping.")
            continue
            
        # Use the recto filename for the output
        output_path = MERGED_DIR / filename
        
        print(f"\nProcessing {filename} (Recto) with {found_v_name} (Verso)...")
        merge_pair(recto_path, verso_path, output_path)
        
    print("\nMerge complete.")

if __name__ == "__main__":
    main()