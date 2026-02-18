from pathlib import Path

import h5py
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform

import itertools
from multiprocessing import Pool
import os

from . import utils


def process_pair(args):
    i, j, a_contour, b_contour, perimeter_points, pad = args

    a_points = np.linspace(0, len(a_contour), perimeter_points, endpoint=False, dtype=int)
    b_points = np.linspace(0, len(b_contour), perimeter_points, endpoint=False, dtype=int)

    a_contact_points = a_contour[a_points]
    b_contact_points = b_contour[b_points]

    a_centroid = utils.contour_centroid(a_contour)
    b_centroid = utils.contour_centroid(b_contour)

    good_contacts = []
    for a_touch, b_touch in itertools.product(a_contact_points, b_contact_points):
        delta = b_centroid + (a_touch - b_touch) - a_centroid
        delta /= np.sqrt(np.sum(delta**2))
        moved_b_contour = b_contour + (a_touch - b_touch) + pad * delta
        if utils.contour_intersect_area(a_contour, moved_b_contour) <= 0:
            good_contacts.append((a_touch, b_touch, delta))

    good_contacts = np.array(good_contacts)
    n_contacts = good_contacts.shape[0]
    ijs = np.full((n_contacts, 1, 2), (i, j), dtype=int)
    contacts = np.concatenate((ijs, good_contacts), axis=1) if n_contacts > 0 else np.empty((0, 4, 2), dtype=float)
    
    return contacts


class BaseMatchDataset(Dataset):

    def __init__(self, root_dir: str, side=384, transform=None, perimeter_points=32, pad=15):
        self.root_dir = Path(root_dir)
        self.side = side
        self.transform = transform
        self.perimeter_points = perimeter_points
        self.pad = pad

    def check_image(self, image_id: str):
        mask_path = self.root_dir / 'mask' / (image_id + '_mask.png')
        valid = mask_path.exists()
        if not valid:
            print(f"[WARNING] Missing mask for {image_id}")
        return valid

    def get_image_size(self, image_id: str):
        image_path = self.root_dir / 'rgba' / (image_id + '.png')
        with Image.open(image_path) as img:
            width, height = img.size
        return np.array((height, width))  # (height, width) for consistency with (y, x) format

    def get_contours(self, image_id: str):
        mask_path = self.root_dir / 'mask' / (image_id + '_mask.png')
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            if 'contours/stacked' in f:
                stacked = np.array(f['contours/stacked'])
                idx = np.array(f['contours/idx'])
                contours = utils.unstack_ragged(stacked, idx, axis=0)
                return contours

            mask = io.imread(mask_path)
            num_colors_guess = utils.guess_num_colors(mask, mask_path)
            palette = utils.get_dominant_mask_colors(mask, num_colors=num_colors_guess)
            contours = utils.find_main_contours(mask, palette)

            stacked, idx = utils.stack_ragged(contours, axis=0)
            f.create_dataset('contours/stacked', data=stacked)
            f.create_dataset('contours/idx', data=idx)

            return contours

    def get_all_touch_points(self, image_id, contours, pad=15, force=False):
        """ Get all touch points between pairs of contours in the image. """
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            path = f'touch_points/pad-{pad}/all'
            if not force and path in f:
                return f[path][:]

            n = len(contours)
            if n < 2:
                print("[WARNING] Not enough contours to find contact points.")
                return np.empty((0, 4, 2), dtype=float)

            pairs = [(i, j, contours[i], contours[j], self.perimeter_points, pad) for i in range(n) for j in range(i + 1, n)]
            with Pool(os.cpu_count()) as pool:
                all_contacts_list = pool.map(process_pair, pairs)

            all_contacts = np.vstack([c for c in all_contacts_list]) if all_contacts_list else np.empty((0, 4, 2), dtype=float)
            if path in f:
                del f[path]
            f.create_dataset(path, data=all_contacts)
            return all_contacts

    def filter_touch_points(self, image_id, touch_points, side, pad=15, force=False):
        """ Filter touch points based on the mask. Skip the ones near invalid (e.g., tape) regions
            that are marked as black pixels in the mask. """
        cache = self.root_dir / 'cache' / f'{image_id}.h5'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache, 'a') as f:
            path = f'touch_points/pad-{pad}/side-{side}'
            if not force and path in f:
                return f[path][:]

            mask_path = self.root_dir / 'mask' / f'{image_id}_mask.png'
            mask = io.imread(mask_path)
            height, width = mask.shape[:2]

            # we are interested in the black regions
            mask = (mask == [0, 0, 0, 255]).all(axis=-1)
            mask_int = transform.integral_image(mask)

            valid = []
            for i, (_, a_touch, b_touch, delta) in enumerate(touch_points):
                y0, x0 = (a_touch - pad * delta - side // 2).astype(int)
                y0, x0 = max(y0, 0), max(x0, 0)
                y1, x1 = min(y0 + side, height - 1), min(x0 + side, width - 1)

                black_area = mask_int[y1, x1] - mask_int[y0, x1] - mask_int[y1, x0] + mask_int[y0, x0]
                if black_area > 0:
                    print(f"[INFO] Skipping touch point #{i} in {image_id}.")
                    continue

                y0, x0 = (b_touch + pad * delta - side // 2).astype(int)
                y0, x0 = max(y0, 0), max(x0, 0)
                y1, x1 = min(y0 + side, height - 1), min(x0 + side, width - 1)

                black_area = mask_int[y1, x1] - mask_int[y0, x1] - mask_int[y1, x0] + mask_int[y0, x0]
                if black_area > 0:
                    print(f"[INFO] Skipping touch point #{i} in {image_id}.")
                    continue

                valid.append(i)

            filtered_touch_points = touch_points[valid]
            if path in f:
                del f[path]
            f.create_dataset(path, data=filtered_touch_points)

        return filtered_touch_points