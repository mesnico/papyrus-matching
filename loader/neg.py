import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from skimage import io, draw, transform
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from loader.base import BaseMatchDataset
from loader import utils


class NegativeMatchDataset(BaseMatchDataset):
    def __init__(self, root_dir: Path, side=384, transform=None, mask_transform=None, perimeter_points=32, pad=15):
        super().__init__(root_dir, side, transform, perimeter_points, pad)
        self.mask_transform = mask_transform
        
        self.image_ids = sorted(p.stem for p in (self.root_dir / 'rgba').glob('*.png'))  # Adjust the glob pattern as needed
        self.image_ids = [i for i in self.image_ids if self.check_image(i)]

        if False:
            self.image_ids = self.image_ids[:5]
            print("[INFO] Using only first 5 images for testing purposes.")

        self.contours = [self.get_contours(i) for i in tqdm(self.image_ids, desc="Getting contours")]
        self.touch_points = [self.get_all_touch_points(i, c, pad=self.pad, force=False) for i, c in zip(tqdm(self.image_ids, desc="Getting touch points"), self.contours)]
        self.touch_points = [self.filter_touch_points(i, t, self.side, force=True) for i, t in zip(tqdm(self.image_ids, desc="Filtering touch points"), self.touch_points)]

        lengths = [len(t) for t in self.touch_points]
        self.lengths_cumsum = np.cumsum(lengths)

    def __len__(self):
        return self.lengths_cumsum[-1] if len(self.lengths_cumsum) else 0

    def __getitem__(self, idx, pedantic=False):
        image_idx = np.searchsorted(self.lengths_cumsum, idx, side='right')
        idx -= self.lengths_cumsum[image_idx - 1] if image_idx > 0 else 0

        # load image
        image_id = self.image_ids[image_idx]
        rgba_path = self.root_dir / 'rgba' / f'{image_id}.png'
        rgba = io.imread(rgba_path)

        # get contours and touch points
        contours = self.contours[image_idx]
        (a_idx, b_idx), a_contact, b_contact, delta = self.touch_points[image_idx][idx]
        a_idx, b_idx = int(a_idx), int(b_idx)
        a_contour, b_contour = contours[a_idx], contours[b_idx]

        # make an empty transparent image
        shape = (self.side, self.side, 4)
        crop_rgba = np.zeros(shape, dtype=np.uint8)

        # find the coordinates of pixels to copy from the original image
        half_side = self.side // 2
        half_delta = delta / 2

        # move the contours such that they are centered in the crop
        t_a = - a_contact + half_side - self.pad * half_delta
        t_b = - b_contact + (self.side - half_side) + self.pad * (delta - half_delta)
        moved_a_contour = a_contour + t_a
        moved_b_contour = b_contour + t_b

        a_row_indices, a_col_indices = draw.polygon(moved_a_contour[:, 0], moved_a_contour[:, 1], shape=shape)
        b_row_indices, b_col_indices = draw.polygon(moved_b_contour[:, 0], moved_b_contour[:, 1], shape=shape)

        a_dst_y = a_row_indices.astype(int)
        a_dst_x = a_col_indices.astype(int)
        a_src_y = (a_row_indices - t_a[0]).astype(int)
        a_src_x = (a_col_indices - t_a[1]).astype(int)

        b_dst_y = b_row_indices.astype(int)
        b_dst_x = b_col_indices.astype(int)
        b_src_y = (b_row_indices - t_b[0]).astype(int)
        b_src_x = (b_col_indices - t_b[1]).astype(int)

        # print warning if xy are out of limits, and remove them
        if pedantic:
            rgba_h, rgba_w = rgba.shape[:2]
            valid_a = (a_src_y >= 0) & (a_src_y < rgba_h) & (a_src_x >= 0) & (a_src_x < rgba_w)
            valid_b = (b_src_y >= 0) & (b_src_y < rgba_h) & (b_src_x >= 0) & (b_src_x < rgba_w)

            if not np.all(valid_a):
                print(f"[WARNING] Invalid A coordinates found for image {image_id}: {np.argwhere(~valid_a)}")
                a_src_y = a_src_y[valid_a]
                a_src_x = a_src_x[valid_a]
                a_dst_y = a_dst_y[valid_a]
                a_dst_x = a_dst_x[valid_a]

            if not np.all(valid_b):
                print(f"[WARNING] Invalid B coordinates found for image {image_id}: {np.argwhere(~valid_b)}")
                b_src_y = b_src_y[valid_b]
                b_src_x = b_src_x[valid_b]
                b_dst_y = b_dst_y[valid_b]
                b_dst_x = b_dst_x[valid_b]

        crop_rgba[a_dst_y, a_dst_x] = rgba[a_src_y, a_src_x]
        crop_rgba[b_dst_y, b_dst_x] = rgba[b_src_y, b_src_x]

        if False:  # debug
            print(f'{image_id=}, {a_contact=}')

            # draw moved contours as debug
            yy, xx = draw.polygon_perimeter(moved_a_contour[:, 0], moved_a_contour[:, 1], shape=shape)
            crop_rgba[yy, xx] = [255, 0, 0, 255]
            yy, xx = draw.polygon_perimeter(moved_b_contour[:, 0], moved_b_contour[:, 1], shape=shape)
            crop_rgba[yy, xx] = [0, 255, 0, 255]

        if self.transform:
            mask_a = np.zeros(shape[:2], dtype=np.float32)
            mask_b = np.zeros(shape[:2], dtype=np.float32)
            mask_a[a_dst_y, a_dst_x] = 1.0
            mask_b[b_dst_y, b_dst_x] = 1.0
            mask_a = self.mask_transform(mask_a[:, :, None])
            mask_b = self.mask_transform(mask_b[:, :, None])
            crop_a = self.transform(crop_rgba) * mask_a
            crop_b = self.transform(crop_rgba) * mask_b
            crop_rgba = crop_a + crop_b
            
            # crop_mask = self.transform(crop_mask)

        # ensure when the alpha is zero, the rgb is also zero
        zero_alpha = crop_rgba[3, :, :] == 0
        crop_rgba[:3, zero_alpha] = 0

        return crop_rgba

if __name__ == "__main__":

    # for root in ['data/organized', 'data/organized_test']:
    #     for pad in (0,5,10,15):
    #         dset = NegativeMatchDataset(root, pad=pad)
    #         print(f"[{root=} {pad=}] Dataset length:", len(dset))

    # exit()
    
    root = 'data/organized'
    from torchvision import transforms as T
    transf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        utils.ApplyToRGB(
            T.RandomGrayscale(p=0.2)
        ),
        utils.ApplyToRGB(
            T.ColorJitter(0.5, 0.5, 0.3, 0.0)
        ),
    ])
    mask_transf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
    ])
    dset = NegativeMatchDataset(root, pad=15, transform=transf, mask_transform=mask_transf)
    sample_rgba = dset[0]
    print(sample_rgba.shape)

    rng = np.random.default_rng(42)
    for i in rng.choice(len(dset), 10, replace=False):
        sample_rgba = dset[i]
        sample_rgba = (sample_rgba.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        print(f"Sample {i}: RGBA shape {sample_rgba.shape}")

        # draw a red dot at the center of the mask and rgba
        center = (sample_rgba.shape[0] // 2, sample_rgba.shape[1] // 2)
        rr, cc = draw.disk(center, 5, shape=sample_rgba.shape[:2])
        sample_rgba[rr, cc] = [255, 0, 0, 255]

        # Save the sample images
        io.imsave(f'figures/neg_samples/sample_{i:02d}rgba.png', sample_rgba)