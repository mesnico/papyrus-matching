import os
import h5py
import torch
import numpy as np
import itertools
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from torchvision import transforms as T
from papyrus_matching.train import LitPapyrusTR
from loader.inference import InferenceDataset


class FragmentMatcher:
    """
    Computes and saves alignment scores for all unique pairs of fragment images
    found in a given directory.
    """
    
    def __init__(self, model_path, output_dir, pad=25, perimeter_points=100, base_device="cuda"):
        """
        Initializes the matcher with model and dataset parameters.

        Args:
            model_path (str): Path to the .ckpt model file.
            output_dir (str): Directory to save the output .hdf5 files.
            pad (int): Padding to use for InferenceDataset.
            perimeter_points (int): Perimeter points for InferenceDataset.
            base_device (str): "cuda" or "cpu". Worker processes will be
                               assigned specific CUDA devices if available.
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.pad = pad
        self.perimeter_points = perimeter_points
        self.base_device = base_device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        self.transf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
        ])
        self.mask_transf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
        ])

    def _process_pair(self, fragment_a_path, fragment_b_path, output_hdf5_path, device, encoder):
        """
        Internal worker function to process a single pair of fragments.
        This function is designed to be run in a separate process.
        """
        torch.set_num_threads(8)  # Avoid thread contention in multiprocessing
        try:            
            # 2. Create Dataset
            dset = InferenceDataset(
                fragment_a_path, 
                fragment_b_path, 
                pad=self.pad, 
                perimeter_points=self.perimeter_points, 
                transform=self.transf, 
                mask_transform=self.mask_transf
            )
            
            if len(dset) == 0:
                return f"Skipped (empty dataset): {os.path.basename(fragment_a_path)}-{os.path.basename(fragment_b_path)}"
                
            # **MODIFIED**: Store all individual patch scores and their translation IDs
            all_patch_scores = []
            all_t_relatives = []
            all_a_coords = []
            all_b_coords = []
            translation_ids = []
            
            # 3. Run Inference Loop for all translations
            for i in range(len(dset)):
                try:
                    patches = dset[i]
                    patches_imgs, _, t_relatives, _, a_coords, b_coords = zip(*patches)
                    
                    if not patches_imgs: # Handle case with zero patches
                        continue 
                        
                    patches_imgs = torch.stack(patches_imgs).to(device)
                
                except Exception as e:
                    print(f"Error loading data index {i} for pair {fragment_a_path}-{fragment_b_path}: {e}")
                    # Skip this translation if data loading fails
                    continue

                # Get scores for all patches in this translation
                with torch.no_grad():
                    scores = encoder(patches_imgs)
                    scores = torch.sigmoid(scores).squeeze(-1).cpu() # (N, 1) -> (N,)
                
                # **MODIFIED**: Store individual scores and their IDs
                if scores.numel() > 0:
                    scores_np = scores.numpy()
                    all_patch_scores.extend(scores_np.tolist())
                    # Add the translation ID 'i' for each patch score
                    translation_ids.extend([i] * len(scores_np))
                    # store t_relatives
                    all_t_relatives.extend(t_relatives)
                    # store a_coords and b_coords
                    all_a_coords.extend(a_coords)
                    all_b_coords.extend(b_coords)

            if not all_patch_scores:
                 return f"Skipped (no valid patches found): {os.path.basename(fragment_a_path)}-{os.path.basename(fragment_b_path)}"

            # 4. Save all scores and IDs to HDF5
            with h5py.File(output_hdf5_path, 'w') as f:
                f.create_dataset('scores', data=np.array(all_patch_scores))
                f.create_dataset('translation_ids', data=np.array(translation_ids, dtype=np.int32))
                f.create_dataset('t_relatives', data=np.array(all_t_relatives))
                f.create_dataset('a_coords', data=np.array(all_a_coords))
                f.create_dataset('b_coords', data=np.array(all_b_coords))
                f.attrs['fragment_a'] = str(fragment_a_path)
                f.attrs['fragment_b'] = str(fragment_b_path)
                f.attrs['model_path'] = str(self.model_path)

            return f"Processed: {os.path.basename(fragment_a_path)}-{os.path.basename(fragment_b_path)}"

        except Exception as e:
            return f"Failed: {os.path.basename(fragment_a_path)}-{os.path.basename(fragment_b_path)} with error: {e}"

    def run_all_pairs(self, fragment_paths, num_workers=4, skip_existing=True):
        """
        Finds all unique pairs from the list of fragment_paths and processes them
        in parallel, saving results to HDF5 files.

        Args:
            fragment_paths (list[str]): A list of file paths to the fragment images.
            num_workers (int): The number of parallel processes to spawn.
            skip_existing (bool): If True, skip pairs where the output HDF5
                                  file already exists.
        """
        
        pairs = list(itertools.combinations(fragment_paths, 2))
        if not pairs:
            print("No fragment pairs to process.")
            return
        
        # Limit workers
        if self.base_device == "cuda":
            max_workers = min(num_workers, len(pairs))
            print(f"Starting GPU processing with {max_workers} workers.")
        else:
            max_workers = min(num_workers, len(pairs), os.cpu_count())
            print(f"Starting CPU processing with {max_workers} workers.")

        mp_context = torch.multiprocessing.get_context('spawn')

        device_name = self.base_device
        device = torch.device(device_name)
            
        # Load Model (must be done in the subprocess)
        model = LitPapyrusTR.load_from_checkpoint(self.model_path)
        encoder = model.eval().to(device)
        
        # **MODIFIED**: Build the list of tasks to run *after* checking for skips
        tasks_to_run = []
        skipped_count = 0
        
        for i, (path_a, path_b) in enumerate(pairs):
            base_a = Path(path_a).stem
            base_b = Path(path_b).stem
            output_path = self.output_dir / f"{base_a}__{base_b}.hdf5"
            
            if skip_existing and output_path.exists():
                skipped_count += 1
                continue
                
            tasks_to_run.append((path_a, path_b, output_path))

        if skipped_count > 0:
            print(f"Skipping {skipped_count} pairs that already exist.")
            
        if not tasks_to_run:
            print("No new pairs to process.")
            return

        print(f"Total new pairs to process: {len(tasks_to_run)}")

        # Run the processing pool
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = []
            for path_a, path_b, output_path in tasks_to_run:
                futures.append(executor.submit(
                    self._process_pair, 
                    path_a, 
                    path_b, 
                    output_path, 
                    device_name,
                    encoder
                ))
            
            # Collect results with TQDM
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
                result = future.result()
                if "Failed" in result or "Skipped" in result:
                    print(result)
                            
        print(f"\nAll fragment pairs processed. Results saved to {self.output_dir}")


if __name__ == "__main__":
    import glob
    # This block is essential for multiprocessing
    
    # 1. Define your parameters
    FRAGMENT_DIR = "data/for_inference"
    MODEL_PATH = "runs/lightning_logs/version_0/checkpoints/epoch=3-step=2576.ckpt"
    OUTPUT_DIR = "results/fragment_scores"
    NUM_WORKERS = 4 # Adjust based on your available GPUs/CPU cores
    
    # 2. Find all your fragment images
    # Assumes fragments are PNG files. Adjust glob pattern as needed.
    fragment_files = [f for f in glob.glob(f"{FRAGMENT_DIR}/*.png") if "_back" not in f.lower()]
    
    if len(fragment_files) < 2:
        print(f"Error: Found fewer than 2 fragments in {FRAGMENT_DIR}. Need at least 2 to form a pair.")
    else:
        print(f"Found {len(fragment_files)} fragments.")
        
        # 3. Initialize the matcher
        matcher = FragmentMatcher(
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            pad=25,
            perimeter_points=100,
            base_device="cuda", # Use "cuda" or "cpu"
        )
        
        # 4. Run the process
        matcher.run_all_pairs(
            fragment_paths=fragment_files,
            num_workers=NUM_WORKERS,
            skip_existing=True
        )