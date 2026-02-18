from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import io, draw, transform
from torchvision.transforms import functional as F

from .base import BaseMatchDataset


class NegativeSyntheticMatchDataset(BaseMatchDataset):
    def __init__(self, root_dir: Path, image_ids: list[str], side=384, stride=32, transform=None, mask_transform=None, post_transform=None, perimeter_points=32, pad=15, max_shift=0):
        super().__init__(root_dir, side, transform, perimeter_points, pad)
        self.stride = stride
        self.max_shift = max_shift
        self.mask_transform = mask_transform
        self.post_transform = post_transform

        self.image_ids = image_ids
        self.image_ids = [i for i in self.image_ids if self.check_image(i)]

        if False:
            self.image_ids = self.image_ids[:5]
            print("[INFO] Using only first 5 images for testing purposes.")

        # Precompute high availability patches and contours for all images
        self.high_availability_patches = [self.get_high_availability_patches(i) for i in tqdm(self.image_ids, desc="Getting high availability patches")]

        self.contours = [self.get_contours(i) for i in tqdm(self.image_ids, desc="Getting contours")]
        self.touch_points = [self.get_all_touch_points(i, c, pad=self.pad, force=False) for i, c in zip(tqdm(self.image_ids, desc="Getting touch points"), self.contours)]

        self.patches_lengths = [len(p) for p in self.high_availability_patches]
        self.touch_lengths = [len(t) for t in self.touch_points]

        self.patches_cumsum = np.cumsum(self.patches_lengths)
        self.touch_cumsum = np.cumsum(self.touch_lengths)
    
    def get_high_availability_patches(self, image_id, force=False):
        """ Get top left corner of patches with high availability in the mask. """
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            path = f'available_patches/side-{self.side}-stride-{self.stride}/all'
            if not force and path in f:
                return f[path][:]

            mask_path = self.root_dir / 'mask' / f'{image_id}_mask.png'
            mask = io.imread(mask_path)
            availability = mask[:, :, 3] > 128  # consider only alpha channel
            availability = transform.integral_image(availability.astype(np.uint8))

            # we are interested in the black regions
            invalidity = (mask == [0, 0, 0, 255]).all(axis=-1)
            invalidity = transform.integral_image(invalidity.astype(np.uint8))

            h, w = mask.shape[:2]
            y_indices = np.arange(0, h - self.side + 1, self.stride)
            x_indices = np.arange(0, w - self.side + 1, self.stride)
            y0, x0 = np.meshgrid(y_indices, x_indices, indexing='ij')
            y1 = y0 + self.side - 1
            x1 = x0 + self.side - 1

            available_area = (availability[y1, x1] - availability[y0, x1] - availability[y1, x0] + availability[y0, x0]) / (self.side * self.side)
            valid_area = (invalidity[y1, x1] - invalidity[y0, x1] - invalidity[y1, x0] + invalidity[y0, x0])

            valid = (available_area > 0.90) & (valid_area == 0)
            points = np.stack([y0[valid], x0[valid]], axis=-1)

            if path in f:
                del f[path]
            f.create_dataset(path, data=points)
            return points

    def __len__(self):
        return self.patches_cumsum[-1] * self.touch_cumsum[-1]

    def __getitem__(self, idx, pedantic=False):
        patch_idx_1, touch_idx = divmod(idx, self.touch_cumsum[-1])
        patch_image_idx_1 = np.searchsorted(self.patches_cumsum, patch_idx_1, side='right')
        patch_idx_1 -= self.patches_cumsum[patch_image_idx_1 - 1] if patch_image_idx_1 > 0 else 0

        touch_image_idx = np.searchsorted(self.touch_cumsum, touch_idx, side='right')
        touch_idx -= self.touch_cumsum[touch_image_idx - 1] if touch_image_idx > 0 else 0

        patch_idx_2, _ = divmod(int(idx + len(self) / 2) % len(self), self.touch_cumsum[-1])
        patch_image_idx_2 = np.searchsorted(self.patches_cumsum, patch_idx_2, side='right')
        patch_idx_2 -= self.patches_cumsum[patch_image_idx_2 - 1] if patch_image_idx_2 > 0 else 0

        # load image
        image_id_1 = self.image_ids[patch_image_idx_1]
        rgba_path = self.root_dir / 'rgba' / f'{image_id_1}.png'
        rgba_1 = io.imread(rgba_path)

        # load another image
        image_id_2 = self.image_ids[patch_image_idx_2]
        rgba_path = self.root_dir / 'rgba' / f'{image_id_2}.png'
        rgba_2 = io.imread(rgba_path)

        # get contours and touch points
        contours = self.contours[touch_image_idx]
        (a_idx, b_idx), a_contact, b_contact, delta = self.touch_points[touch_image_idx][touch_idx]
        a_idx, b_idx = int(a_idx), int(b_idx)
        a_contour, b_contour = contours[a_idx], contours[b_idx]

        # get first patch
        origin = self.high_availability_patches[patch_image_idx_1][patch_idx_1]
        y0, x0 = origin
        crop_rgba_1 = rgba_1[y0:y0 + self.side, x0:x0 + self.side]

        # get second patch
        origin = self.high_availability_patches[patch_image_idx_2][patch_idx_2]
        y0, x0 = origin
        crop_rgba_2 = rgba_2[y0:y0 + self.side, x0:x0 + self.side]

        # move contours and draw them on the patch
        shape = crop_rgba_1.shape
        half_side = self.side // 2
        half_delta = delta / 2
        shift_magnitude = ((np.random.rand() * 2) - 1) * self.max_shift

        # move the contours such that they are centered in the crop
        t_a = - a_contact + half_side - (self.pad + shift_magnitude) * half_delta
        t_b = - b_contact + (self.side - half_side) + (self.pad + shift_magnitude) * (delta - half_delta)
        moved_a_contour = a_contour + t_a
        moved_b_contour = b_contour + t_b

        a_row_indices, a_col_indices = draw.polygon(moved_a_contour[:, 0], moved_a_contour[:, 1], shape=shape)
        b_row_indices, b_col_indices = draw.polygon(moved_b_contour[:, 0], moved_b_contour[:, 1], shape=shape)

        a_dst_y = a_row_indices.astype(int)
        a_dst_x = a_col_indices.astype(int)

        b_dst_y = b_row_indices.astype(int)
        b_dst_x = b_col_indices.astype(int)

        mask_a = np.zeros(shape[:2], dtype=bool)
        mask_b = np.zeros(shape[:2], dtype=bool)
        mask_a[a_dst_y, a_dst_x] = True
        mask_b[b_dst_y, b_dst_x] = True

        if False:  # debug
            print(f'{image_id_1=} {patch_image_idx_1=} {touch_image_idx=} {patch_idx=} {touch_idx=} {a_idx=} {b_idx=}')
            print(f'  {y0=} {x0=} {a_contact=} {b_contact=} {delta=} {t_a=} {t_b=}')

            # draw moved contours as debug
            yy, xx = draw.polygon_perimeter(moved_a_contour[:, 0], moved_a_contour[:, 1], shape=shape)
            crop_rgba_1[yy, xx] = [255, 0, 0, 255]
            yy, xx = draw.polygon_perimeter(moved_b_contour[:, 0], moved_b_contour[:, 1], shape=shape)
            crop_rgba_1[yy, xx] = [0, 255, 0, 255]

        if self.transform:
            transformed_rgba_1 = self.transform(crop_rgba_1)
            transformed_rgba_2 = self.transform(crop_rgba_2)

            # 4. Transform the masks (as before)
            mask_a = self.mask_transform(mask_a[:, :, None].astype(np.float32))
            mask_b = self.mask_transform(mask_b[:, :, None].astype(np.float32))

            # 5. Apply masks to the *different* image versions
            #    Piece A uses the original, un-shifted image
            crop_a = transformed_rgba_1 * mask_a
            #    Piece B uses the new, shifted image
            crop_b = transformed_rgba_2 * mask_b

            # 6. Combine the two pieces
            crop_rgba_1 = torch.maximum(crop_a, crop_b)
        else:
            raise NotImplementedError("Transform must be provided for PositiveSyntheticMatchDataset")
            mask = mask_a | mask_b
            crop_rgba_1 *= mask[:, :, None]
            crop_rgba_1 = torch.tensor(crop_rgba_1).permute(2, 0, 1).float() / 255.0

            # crop_mask = self.transform(crop_mask)

        # ensure when the alpha is zero, the rgb is also zero
        zero_alpha = crop_rgba_1[3, :, :] == 0
        crop_rgba_1[:3, zero_alpha] = 0

        if self.post_transform:
            crop_rgba_1 = self.post_transform(crop_rgba_1)

        return crop_rgba_1

if __name__ == "__main__":

    # for root in ['data/organized', 'data/organized_test']:
    #     for pad in (0, 5, 10, 15):
    #         dset = PositiveSyntheticMatchDataset(root, pad=pad)
    #         print(f"[{root=} {pad=}] Dataset length:", len(dset))

    # exit()
    
    root = 'data/organized'
    from torchvision import transforms as T
    from torchvision.transforms import v2 as T2
    from loader import utils
    transf = T.Compose([
        T2.ToImage(),
        T2.Resize((224, 224)),
        utils.ApplyToRGB(
            T2.JPEG(quality=(20, 100)),
        ),
        utils.ApplyToRGB(
            T2.RandomPosterize(bits=4, p=0.2),
        ),
        utils.ApplyToRGB(
            T.RandomGrayscale(p=0.2)
        ),
        utils.ApplyToRGB(
            T.ColorJitter(0.5, 0.5, 0.3, 0.1)
        ),
        T2.ToDtype(torch.float32, scale=True),
    ])
    mask_transf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
    ])
    post_transforms = T2.Compose([
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
    ])
    dset = NegativeSyntheticMatchDataset(root, pad=20, max_shift=0, transform=transf, mask_transform=mask_transf, post_transform=post_transforms)
    sample_rgba = dset[0]
    print(sample_rgba.shape)

    rng = np.random.default_rng(42)
    for i in rng.choice(len(dset), 15, replace=False):
        sample_rgba = dset[i]
        sample_rgba = (sample_rgba.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        print(f"Sample {i}: RGBA shape {sample_rgba.shape}")

        # draw a red dot at the center of the mask and rgba
        center = (sample_rgba.shape[0] // 2, sample_rgba.shape[1] // 2)
        rr, cc = draw.disk(center, 5, shape=sample_rgba.shape[:2])
        sample_rgba[rr, cc] = [255, 0, 0, 255]

        # Save the sample images
        Path('figures/synth_neg_samples').mkdir(parents=True, exist_ok=True)
        io.imsave(f'figures/synth_neg_samples/sample_{i:02d}rgba.png', sample_rgba)

        # Save the alpha channel separately for visualization
        alpha_channel = sample_rgba[:, :, 3]
        # io.imsave(f'figures/synth_neg_samples/sample_{i:02d}alpha.png', alpha_channel)