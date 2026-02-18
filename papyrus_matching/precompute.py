import os
import h5py
import torch
import numpy as np
import itertools
import tqdm
import glob
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .train import LitPapyrusTR
from loader.inference import InferenceDataset

MODELS={
    'patch-encoder-v0-2-0.ckpt': 'https://github.com/mesnico/papyrus-matching/releases/download/v0.2.0/patch-encoder-v0-2-0.ckpt',
}

# ---------------------------------------------------------
# 1. The Worker Class (CPU Heavy Lifting)
# ---------------------------------------------------------
class FragmentPairDataset(Dataset):
    """
    This dataset takes a list of pairs. 
    In __getitem__, it does ALL the heavy CPU work:
    1. Instantiates the InferenceDataset
    2. Loops through it to extract all patches and coordinates
    3. Stacks them into Tensors to send to the main GPU process
    """
    def __init__(self, pairs, pad, perimeter_points_distance, transf, mask_transf):
        self.pairs = pairs
        self.pad = pad
        self.perimeter_points_distance = perimeter_points_distance
        self.transf = transf
        self.mask_transf = mask_transf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b = self.pairs[idx]
        
        try:
            # Heavy CPU Operation: Create the inner dataset
            dset = InferenceDataset(
                path_a, path_b,
                pad=self.pad,
                perimeter_points_distance=self.perimeter_points_distance,
                transform=self.transf,
                mask_transform=self.mask_transf
            )
            
            if len(dset) == 0:
                return None # Signal to skip

            # Containers for this specific pair
            all_patches = []
            all_t_relatives = []
            all_a_coords = []
            all_b_coords = []
            translation_ids = []

            # Heavy CPU Loop: Extract patches for all translations
            for i in range(len(dset)):
                try:
                    patches = dset[i]
                    # Unzip the data
                    patches_imgs, _, t_rels, _, a_co, b_co = zip(*patches)
                    
                    if not patches_imgs:
                        continue

                    # We stack images here to send a single tensor block per translation
                    # or we can flatten everything. Let's flatten everything for this pair.
                    for j, img in enumerate(patches_imgs):
                        all_patches.append(img)
                        translation_ids.append(i)
                        all_t_relatives.append(t_rels[j])
                        all_a_coords.append(a_co[j])
                        all_b_coords.append(b_co[j])
                        
                except Exception as e:
                    # Log error inside worker if necessary, but keep moving
                    print(f"Worker Error on translation {i} for {Path(path_a).name}: {e}")
                    continue
            
            if not all_patches:
                return None

            # Stack everything into Tensors/Numpy arrays for efficient transport
            return {
                "path_a": str(path_a),
                "path_b": str(path_b),
                "patches": torch.stack(all_patches), # (N_total_patches, C, H, W)
                "translation_ids": np.array(translation_ids, dtype=np.int32),
                "t_relatives": np.array(all_t_relatives),
                "a_coords": np.array(all_a_coords),
                "b_coords": np.array(all_b_coords)
            }

        except Exception as e:
            print(f"Failed processing pair {Path(path_a).name}-{Path(path_b).name}: {e}")
            return None

def collate_fn(batch):
    # Batch size is 1, so we just unwrap the list
    # If None was returned (skipped), filtering happens in the main loop or here
    return batch[0]

# ---------------------------------------------------------
# 2. The Matcher Class (Main Process / GPU)
# ---------------------------------------------------------
class FragmentMatcher:
    def __init__(self, model_name, output_dir, pad=25, perimeter_points_distance=32, base_device="cuda"):
        self.model_path = Path(model_name)
        self.output_dir = Path(output_dir)
        self.pad = pad
        self.perimeter_points_distance = perimeter_points_distance
        self.base_device = base_device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transf = T.Compose([T.ToTensor(), T.Resize((224, 224))])
        self.mask_transf = T.Compose([T.ToTensor(), T.Resize((224, 224))])

        # download the model if not provided
        if not self.model_path.exists():
            torch.hub.download_url_to_file(MODELS[model_name], self.model_path)


    def run_all_pairs(self, fragment_paths, num_workers=4, skip_existing=True):
        # 1. Prepare Pairs
        all_pairs = list(itertools.combinations(fragment_paths, 2))
        
        # 2. Filter existing files BEFORE creating the dataset
        tasks_to_run = []
        for path_a, path_b in all_pairs:
            base_a = Path(path_a).stem
            base_b = Path(path_b).stem
            output_path = self.output_dir / f"{base_a}__{base_b}.hdf5"
            
            if skip_existing and output_path.exists():
                continue
            tasks_to_run.append((path_a, path_b))

        if not tasks_to_run:
            print("No new pairs to process.")
            return

        print(f"Processing {len(tasks_to_run)} pairs using {num_workers} CPU workers...")

        # 3. Initialize Model ONCE (Main Process)
        device = torch.device(self.base_device)
        print(f"Loading model to {device}...")
        model = LitPapyrusTR.load_from_checkpoint(self.model_path)
        encoder = model.eval().to(device)

        # 4. Create Dataset & DataLoader
        # The DataLoader manages the Multiprocessing pool automatically.
        dataset = FragmentPairDataset(
            tasks_to_run, 
            self.pad, 
            self.perimeter_points_distance, 
            self.transf, 
            self.mask_transf
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=1,        # Process one pair at a time
            shuffle=False, 
            num_workers=num_workers, 
            collate_fn=collate_fn,
            pin_memory=(self.base_device == "cuda") # Faster transfer to GPU
        )

        # 5. Main Inference Loop
        # We use tqdm on the dataloader. The workers are pre-fetching the next items in the background.
        with torch.no_grad():
            for batch_data in tqdm.tqdm(dataloader, desc="Inference"):
                if batch_data is None:
                    continue # Skipped or failed in worker
                
                path_a = batch_data['path_a']
                path_b = batch_data['path_b']
                patches = batch_data['patches'].to(device) # Move batch to GPU
                
                # -- INFERENCE --
                # If a pair has massive amount of patches, you might need to mini-batch this
                # to avoid OOM. e.g. chunks of 512.
                # Here we assume one pair's patches fit in VRAM.
                
                # Optional: Batching huge patch lists
                scores_list = []
                chunk_size = 256
                for i in range(0, len(patches), chunk_size):
                    chunk = patches[i : i + chunk_size]
                    out = encoder(chunk)
                    out = torch.sigmoid(out).squeeze(-1).cpu()
                    scores_list.append(out)
                
                scores = torch.cat(scores_list).numpy()

                # -- SAVING --
                base_a = Path(path_a).stem
                base_b = Path(path_b).stem
                output_hdf5_path = self.output_dir / f"{base_a}__{base_b}.hdf5"

                with h5py.File(output_hdf5_path, 'w') as f:
                    f.create_dataset('scores', data=scores)
                    f.create_dataset('translation_ids', data=batch_data['translation_ids'])
                    f.create_dataset('t_relatives', data=batch_data['t_relatives'])
                    f.create_dataset('a_coords', data=batch_data['a_coords'])
                    f.create_dataset('b_coords', data=batch_data['b_coords'])
                    f.attrs['fragment_a'] = path_a
                    f.attrs['fragment_b'] = path_b
                    f.attrs['model_path'] = str(self.model_path)

        print(f"\nAll fragment pairs processed. Results saved to {self.output_dir}")


if __name__ == "__main__":
    import glob
    import argparse
    # This block is essential for multiprocessing
    
    # Define your parameters using argparse
    parser = argparse.ArgumentParser(description="Fragment Matcher Parameters")
    parser.add_argument('fragment_dir', type=str, help='Directory containing fragment images')
    parser.add_argument('--model_name', type=str, default="patch-encoder-v0-2-0.ckpt", help='Path to the model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save output results')
    parser.add_argument('--skip_existing', type=bool, default=True, help='Skip existing output files')
    parser.add_argument('--num-workers', type=int, default=24, help="Num workers to use")
    args = parser.parse_args()

    FRAGMENT_DIR = args.fragment_dir
    MODEL_NAME = args.model_name
    OUTPUT_DIR = Path(args.output_dir) / (Path(FRAGMENT_DIR).stem)
    SKIP_EXISTING = args.skip_existing

    NUM_WORKERS = args.num_workers # Adjust based on your available GPUs/CPU cores
    
    for side in ["recto", "verso"]:
        # Assumes fragments are PNG files. Adjust glob pattern as needed.
        if side == "verso":
            fragment_files = [f for f in glob.glob(f"{FRAGMENT_DIR}/*.png") if "_back" in Path(f).stem.lower()]
        else:
            fragment_files = [f for f in glob.glob(f"{FRAGMENT_DIR}/*.png") if "_back" not in Path(f).stem.lower()]

        fragment_files.sort()
        
        if len(fragment_files) < 2:
            print(f"Error: Found fewer than 2 fragments in {FRAGMENT_DIR}. Need at least 2 to form a pair.")
        else:
            print(f"Found {len(fragment_files)} fragments.")
            
            # 3. Initialize the matcher
            matcher = FragmentMatcher(
                model_name=MODEL_NAME,
                output_dir=OUTPUT_DIR / side,
                pad=20,
                perimeter_points_distance=100,
                base_device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 4. Run the process
            matcher.run_all_pairs(
                fragment_paths=fragment_files,
                num_workers=NUM_WORKERS,
                skip_existing=True
            )