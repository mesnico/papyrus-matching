import numpy as np
from skimage import io, draw, measure, morphology
from matplotlib import pyplot as plt
import torch
import os
import itertools
from multiprocessing import Pool
from scipy.spatial.distance import cdist

from . import utils

def process_pair_extended(args):
    i, j, a_contour, b_contour, perimeter_points_distance, pad = args
    other_valid_contacts_threshold = 100

    # segment lenghts
    def get_sampling_indexes(contour):
        subsample = 3
        contour = contour[::subsample]  # to avoid discretization issues
        lenghts = np.linalg.norm(contour[1:] - contour[:-1], axis=1)
        lenghts = np.insert(lenghts, 0, 0.0)
        cumsum_length = np.cumsum(lenghts)
        samples = range(0, int(cumsum_length[-1]), perimeter_points_distance)
        indexes = np.searchsorted(cumsum_length, samples) * subsample
        return indexes
    
    a_indexes = get_sampling_indexes(a_contour)
    b_indexes = get_sampling_indexes(b_contour)

    a_contact_points = a_contour[a_indexes]
    b_contact_points = b_contour[b_indexes]

    a_centroid = utils.contour_centroid(a_contour)
    b_centroid = utils.contour_centroid(b_contour)

    good_contacts_main = []
    # This will be a list [N_main_contacts] of (list [len(a_contour)] of (arrays [N_b_pts, 2]))
    other_good_contacts_idxs = []

    for a_touch, b_touch in itertools.product(a_contact_points, b_contact_points):
        delta = b_centroid + (a_touch - b_touch) - a_centroid
        delta /= np.sqrt(np.sum(delta**2))
        moved_b_contour = b_contour + (a_touch - b_touch) + pad * delta
        moved_b_contact_points = b_contact_points + (a_touch - b_touch) + pad * delta
        
        if utils.contour_intersect_area(a_contour, moved_b_contour) <= 0:
            # 1. Store the main "good contact"
            good_contacts_main.append((a_touch, b_touch, delta))
            
            # 2. Find and store all "other" contacts for this pose
            dist_matrix = cdist(a_contact_points, moved_b_contact_points)
            other_valid_contacts_mask = dist_matrix < other_valid_contacts_threshold

            a_idx, b_idx = np.where(other_valid_contacts_mask)
            other_good_contacts_idxs.append((a_idx, b_idx))

    # --- Process the main contacts as before ---
    good_contacts_array = np.array(good_contacts_main)
    n_contacts = good_contacts_array.shape[0]
    
    if n_contacts == 0:
        return np.empty((0, 4, 2), dtype=float), [] 

    ijs = np.full((n_contacts, 1, 2), (i, j), dtype=int)
    contacts = np.concatenate((ijs, good_contacts_array), axis=1)

    other_contacts = np.empty((len(other_good_contacts_idxs), a_contact_points.shape[0], b_contact_points.shape[0], 4), dtype=float)
    other_contacts[:] = np.nan  # Initialize with NaNs
    for idx in range(len(other_good_contacts_idxs)):
        a_idx, b_idx = other_good_contacts_idxs[idx]
        for x, y in zip(a_idx, b_idx):
            other_contacts[idx, x, y] = np.concatenate((a_contact_points[x], b_contact_points[y]))
    
    return contacts, other_contacts


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, rgba_path_a, rgba_path_b, side=384, transform=None, mask_transform=None, perimeter_points_distance=64, pad=15, return_final_image=False):
        self.mask_transform = mask_transform
        self.transform = transform
        self.side = side
        self.perimeter_points_distance = perimeter_points_distance
        self.pad = pad
        self.rgba_path_a = rgba_path_a
        self.rgba_path_b = rgba_path_b
        self.return_final_image = return_final_image

        # Extract one contour from each image's alpha channel
        self.contour_a = self.get_contour_from_alpha(rgba_path_a)
        self.contour_b = self.get_contour_from_alpha(rgba_path_b)

        # Get touch points between the two specific contours
        self.touch_points, self.other_touch_points = self.get_all_touch_points(
            self.contour_a, self.contour_b, 
            pad=self.pad, force=False
        )
        # Note: self.filter_touch_points call is removed as it's no longer applicable.

    def get_contour_from_alpha(self, image_path, pad_amount=(10, 10)):
        """
        Loads an RGBA image, extracts the alpha channel, and finds the 
        main exterior contour using the robust logic from find_main_contours.
        """
        try:
            rgba = io.imread(image_path)
        except FileNotFoundError:
            print(f"[ERROR] Image not found at {image_path}")
            return np.empty((0, 2), dtype=float)

        if rgba.shape[2] != 4:
            print(f"[ERROR] Image at {image_path} is not RGBA.")
            return np.empty((0, 2), dtype=float)
            
        # 1. Binarize from alpha channel (alpha > 0)
        binary = rgba[:, :, 3] > 0
        
        # 2. Pad
        binary = np.pad(binary, pad_amount, mode='constant', constant_values=0)
        
        # 3. Morphological closing
        binary = morphology.binary_closing(binary, morphology.disk(1))

        # 4. Label regions
        labels = measure.label(binary)
        region_props = measure.regionprops(labels, intensity_image=binary)

        if not region_props:
            print(f"[WARNING] No regions found in alpha channel of {image_path}")
            return np.empty((0, 2), dtype=float)

        # 5. Get largest region prop
        prop = max(region_props, key=lambda x: x.area)

        # 6. Fill holes
        # find perimeter contour of the region prop, ignoring holes
        binary_filled = morphology.remove_small_holes(labels == prop.label, area_threshold=1000)
        
        # 7. Find contours
        contours = measure.find_contours(binary_filled, 0.5)

        if not contours:
            print(f"[WARNING] No contours found after processing {image_path}")
            return np.empty((0, 2), dtype=float)

        # 8. Get the longest contour (this is the exterior one)
        contour = max(contours, key=len)
        
        # 9. Remove padding
        contour -= np.array(pad_amount)

        return contour
    
    def get_all_touch_points(self, contour_a, contour_b, pad=15, force=False):
        """ Get all touch points between two specific contours. """

        if contour_a.shape[0] < 2 or contour_b.shape[0] < 2:
            print("[WARNING] Not enough contour data to find contact points.")
            return np.empty((0, 4, 2), dtype=float)

        # Create a single pair to process: (idx_a, idx_b, contour_a, contour_b, ...)
        # We use 0 and 1 as the dummy indices
        pairs = [(0, 1, contour_a, contour_b, self.perimeter_points_distance, pad)]
        
        # with Pool(os.cpu_count()) as pool:
        data = [process_pair_extended(p) for p in pairs] #pool.map(process_pair_extended, pairs)
        all_contacts_list, other_contacts_list = zip(*data)

        all_contacts = np.vstack([c for c in all_contacts_list])
        other_contacts = np.vstack([c for c in other_contacts_list])

        return all_contacts, other_contacts

    def __len__(self):
        return len(self.touch_points)

    def __getitem__(self, idx):
        # load both images
        rgba_a = io.imread(self.rgba_path_a)
        rgba_b = io.imread(self.rgba_path_b)

        # get contours and touch points
        data = self.touch_points[idx]
        a_contact = data[1]
        b_contact = data[2]
        delta = data[3]
        
        other_touch_points = self.other_touch_points[idx]
        other_touch_points = other_touch_points.reshape(-1, 4)
        
        a_contour = self.contour_a
        b_contour = self.contour_b

        def _get_blit_slices(t_vec, src_shape, dst_shape):
            """
            Helper function to calculate clipped source and destination slices for blitting.
            """
            src_h, src_w = src_shape[:2]
            dst_h, dst_w = dst_shape[:2]
            t_y, t_x = np.floor(t_vec).astype(int)

            # Calculate source slice (from rgba_a or rgba_b)
            src_y_start = max(0, -t_y)
            src_x_start = max(0, -t_x)
            src_y_end = min(src_h, dst_h - t_y)
            src_x_end = min(src_w, dst_w - t_x)
            
            # Calculate destination slice (on crop_rgba)
            dst_y_start = max(0, t_y)
            dst_x_start = max(0, t_x)
            dst_y_end = min(dst_h, t_y + src_h)
            dst_x_end = min(dst_w, t_x + src_w)

            # Check for non-overlapping (empty) slices
            if src_y_start >= src_y_end or src_x_start >= src_x_end or \
               dst_y_start >= dst_y_end or dst_x_start >= dst_x_end:
                return None, None

            src_slice = (slice(src_y_start, src_y_end), slice(src_x_start, src_x_end))
            dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))
            
            return src_slice, dst_slice

        # find the coordinates of pixels to copy from the original image
        def _prepare_patch(cropped, a_coords=None, b_coords=None):

            if cropped:
                side = self.side
            else:
                width_a = a_contour[:, 1].max() - a_contour[:, 1].min()
                width_b = b_contour[:, 1].max() - b_contour[:, 1].min()
                height_a = a_contour[:, 0].max() - a_contour[:, 0].min()
                height_b = b_contour[:, 0].max() - b_contour[:, 0].min()
                side = int(np.ceil(max(width_a + width_b, height_a + height_b))) + 2 * self.pad

            # make empty transparent images
            shape = (side, side, 4)
            shape_mask = (side, side)
            crop_a_only = np.zeros(shape, dtype=np.uint8)
            crop_b_only = np.zeros(shape, dtype=np.uint8)

            half_side = side // 2
            half_delta = delta / 2

            # move the contours such that they are centered in the crop
            t_relative = (a_contact - b_contact) + self.pad * delta
            if a_coords is not None and b_coords is not None:
                moved_b_coords = b_coords + t_relative
                middle_point = (a_coords + moved_b_coords) / 2
                t_a = half_side - middle_point
                t_b = t_relative + t_a
            else:
                t_a = - a_contact + half_side - self.pad * half_delta
                t_b = - b_contact + (side - half_side) + self.pad * (delta - half_delta)
                middle_point = a_contact

            # --- OPTIMIZATION: Blit piece A ---
            src_a_slice, dst_a_slice = _get_blit_slices(t_a, rgba_a.shape, shape)
            if src_a_slice:
                src_patch = rgba_a[src_a_slice]
                dst_region = crop_a_only[dst_a_slice]
                
                # Composite using alpha channel as mask
                mask = src_patch[:, :, 3] > 0
                dst_region[mask] = src_patch[mask]
                crop_a_only[dst_a_slice] = dst_region

            # --- OPTIMIZATION: Blit piece B ---
            src_b_slice, dst_b_slice = _get_blit_slices(t_b, rgba_b.shape, shape)
            if src_b_slice:
                src_patch = rgba_b[src_b_slice]
                dst_region = crop_b_only[dst_b_slice]

                # Composite using alpha channel as mask
                mask = src_patch[:, :, 3] > 0
                dst_region[mask] = src_patch[mask]
                crop_b_only[dst_b_slice] = dst_region
            
            # --- Combine the two pieces ---
            # Create the final composite image by overlaying B onto A
            crop_rgba = crop_a_only.copy()
            mask_b_full = crop_b_only[:, :, 3] > 0
            crop_rgba[mask_b_full] = crop_b_only[mask_b_full]

            if cropped and self.transform:
                # --- OPTIMIZATION: Get masks from blitted images ---
                # Use the alpha channels of the separate patches as masks
                mask_a = (crop_a_only[:, :, 3] > 0).astype(np.float32)
                mask_b = (crop_b_only[:, :, 3] > 0).astype(np.float32)
                
                mask_a = self.mask_transform(mask_a[:, :, None])
                mask_b = self.mask_transform(mask_b[:, :, None])
                
                # Apply original logic: transform composite, then split
                transformed_composite = self.transform(crop_rgba)
                crop_a = transformed_composite * mask_a
                crop_b = transformed_composite * mask_b
                crop_rgba = crop_a + crop_b
                
            # --- Bug fix (same as before) ---
            if cropped and self.transform:
                zero_alpha = crop_rgba[3, :, :] == 0
                crop_rgba[:3, zero_alpha] = 0
            else:
                zero_alpha = crop_rgba[:, :, 3] == 0
                crop_rgba[zero_alpha, :3] = 0

            return crop_rgba, t_a, t_relative, middle_point

        cropped_patches = []
        valid_coords = other_touch_points[~np.isnan(other_touch_points).all(axis=1)]
        for coord in valid_coords:
            a_coords = coord[0:2]
            b_coords = coord[2:4]
            cropped_patch = _prepare_patch(cropped=True, a_coords=a_coords, b_coords=b_coords)
            cropped_patches.append(cropped_patch + (a_coords, b_coords))  # Append the original contact points as well

        if self.return_final_image:
            full_image = _prepare_patch(cropped=False)
            return cropped_patches, full_image
        else:
            return cropped_patches
    

if __name__ == "__main__":
    root = 'data/toy_fragments'
    image_name_a = root + "/test_fragment_easy1_L.png"
    image_name_b = root + "/test_fragment_easy1_R.png"

    from torchvision import transforms as T
    transf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
    ])
    mask_transf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
    ])
    dset = InferenceDataset(image_name_a, image_name_b, pad=20, perimeter_points_distance=128, transform=transf, mask_transform=mask_transf)

    output_path = 'figures/inference_samples'
    os.makedirs(output_path, exist_ok=True)
    
    for idx in range(len(dset)):
        cropped_patches, full_image = dset[idx]
        
        # Save full image
        full_image_np, t_a_full, _ = full_image
        
        # Save cropped patches
        for p_idx, (crop_rgba, t_a_crop, middle_point) in enumerate(cropped_patches):
            crop_rgba_np = (crop_rgba.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            io.imsave(os.path.join(output_path, f'cropped_patch_{idx}_{p_idx}.png'), crop_rgba_np)

            # draw red_dots at the center of the full_image
            center = middle_point + t_a_full
            rr, cc = draw.disk(center, 30, shape=full_image_np.shape[:2])
            # sample_mask[rr, cc] = [255, 0, 0, 255]
            full_image_np[rr, cc] = [255, 0, 0, 255]

        io.imsave(os.path.join(output_path, f'full_image_{idx}.png'), full_image_np)