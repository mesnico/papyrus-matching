from multiprocessing import Pool
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import io

from base import BaseMatchDataset
import utils


class PositiveRealMatchDataset(BaseMatchDataset):
    def __init__(self, root_dir: Path, side=384, transform=None):
        super().__init__(root_dir, side, transform)

        self.image_ids = sorted(p.stem for p in (self.root_dir / 'rgba').glob('*.png'))  # Adjust the glob pattern as needed
        self.image_ids = [i for i in self.image_ids if self.check_image(i)]

        self.contact_points = [self.get_contact_points(i) for i in tqdm(self.image_ids)]
        self.contact_points = [self.filter_contact_points(c, i) for c, i in zip(tqdm(self.contact_points), self.image_ids)]

        self.lengths = [len(contacts) for contacts in self.contact_points]
        self.lengths_cumsum = np.cumsum(self.lengths)

    def get_contact_points(self, image_id: str):
        mask_path = self.root_dir / 'mask' / (image_id + '_mask.png')
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            if 'contacts' in f:
                return f['contacts'][:]

        contours = self.get_contours(image_id)
        contacts = utils.find_all_contact_points(contours)

        with h5py.File(cache, 'a') as f:
            f.create_dataset('contacts', data=contacts)

        return contacts

    def filter_contact_points(self, contact_points, image_id):
        """ Filter contact points to remove those too close to the image border or too near masked areas (e.g., tape). """
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            if f'filtered_contacts/{self.side}' in f:
                return f[f'filtered_contacts/{self.side}'][:]

            if len(contact_points) > 0:
                mask_path = self.root_dir / 'mask' / (image_id + '_mask.png')
                mask = io.imread(mask_path)
                hw = np.array(mask.shape[:2])

                half_side = self.side // 2
                valid = (half_side <= contact_points).all(axis=1) & (contact_points < hw - half_side).all(axis=1)
                contact_points = contact_points[valid]

                binary_mask = np.all(mask == (0, 0, 0, 255), axis=-1)  # black pixel in the mask are to avoid
                integral = binary_mask.astype(np.float32).cumsum(axis=0).cumsum(axis=1)

                y0, x0 = (contact_points - half_side).T
                y1, x1 = y0 + self.side, x0 + self.side
                valid = (integral[y1, x1] - integral[y1, x0] - integral[y0, x1] + integral[y0, x0]) == 0
                contact_points = contact_points[valid]

                f.create_dataset(f'filtered_contacts/{self.side}', data=contact_points)

            return contact_points

    def __len__(self):
        return self.lengths_cumsum[-1]

    def __getitem__(self, idx):
        image_idx = np.searchsorted(self.lengths_cumsum, idx, side='right')
        contact_idx = idx - (self.lengths_cumsum[image_idx - 1] if image_idx > 0 else 0)

        contact_point = self.contact_points[image_idx][contact_idx]

        y, x = contact_point
        half_side = self.side // 2
        y0 = y - half_side
        x0 = x - half_side
        y1 = y0 + self.side
        x1 = x0 + self.side

        image_id = self.image_ids[image_idx]

        image_path = self.root_dir / 'rgba' / (image_id + '.png')
        image = io.imread(image_path)
        crop_rgba = image[y0:y1, x0:x1]
        assert crop_rgba.shape[:2] == (self.side, self.side), f"Crop shape mismatch: {crop_rgba.shape} != ({self.side}, {self.side})"

        mask_path = self.root_dir / 'mask' / (image_id + '_mask.png')
        mask = io.imread(mask_path)
        crop_mask = mask[y0:y1, x0:x1]

        # Extract the contact point region
        if self.transform:
            crop_rgba = self.transform(crop_rgba)
            crop_mask = self.transform(crop_mask)

        return crop_rgba, crop_mask


if __name__ == "__main__":

    for root in ['data/organized', 'data/organized_test']:
        dset = PositiveRealMatchDataset(root)
        print(f"[{root=}] Dataset length:", len(dset))

    exit()

    from skimage import draw
    root = 'data/organized'
    dset = PositiveRealMatchDataset(root)
    print(len(dset))
    sample_rgba, sample_mask = dset[0]
    print(sample_rgba.shape)

    for i in np.random.choice(len(dset), 10, replace=False):
        sample_rgba, sample_mask = dset[i]
        print(f"Sample {i}: RGBA shape {sample_rgba.shape}, Mask shape {sample_mask.shape}")

        # draw a red dot at the center of the mask and rgba
        center = (sample_mask.shape[0] // 2, sample_mask.shape[1] // 2)
        rr, cc = draw.disk(center, 5, shape=sample_mask.shape[:2])
        sample_mask[rr, cc] = [255, 0, 0, 255]
        sample_rgba[rr, cc] = [255, 0, 0, 255]

        # Save the sample images
        io.imsave(f'figures/real_pos_samples/sample_{i:02d}rgba.png', sample_rgba)
        io.imsave(f'figures/real_pos_samples/sample_{i:02d}mask.png', sample_mask)