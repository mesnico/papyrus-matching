import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.spatial.distance import cdist
import shapely
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import torch
import cv2


def stack_ragged(array_list, axis=0):
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx


def unstack_ragged(stacked, idx, axis=0):
    return np.split(stacked, idx, axis=axis)


def estimate_num_colors(img, k_range=(2, 15), sample_size=5000):
    """ Estimate number of colors using silhouette score on a sample of pixels. """
    if img.shape[0] > sample_size:
        idx = np.random.RandomState(42).choice(img.shape[0], sample_size, replace=False)
        sample = img[idx]
    else:
        sample = img

    best_k = k_range[0]
    best_score = -1
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(sample)
        labels = kmeans.labels_
        score = silhouette_score(sample, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def guess_num_colors(mask, mask_path):
    """ Guess the number of colors from the filename and presence of invalid (black) regions. """
    num_colors_guess = mask_path.as_posix().count('+') + 1
    if (mask == (0,0,0,255)).all(axis=-1).any():
        num_colors_guess += 1
    return num_colors_guess


def get_dominant_mask_colors(mask, num_colors=None):
    mask = mask.reshape(-1, 4)  # Flatten the image to (N, 3)
    mask = mask[mask[..., 3] > 127]

    if num_colors is None:
        num_colors = estimate_num_colors(mask)
        print(f"[INFO] Estimated number of colors: {num_colors}")

    sample = mask
    if len(mask) > 10000:
        sample = np.random.RandomState(42).choice(len(mask), 10000)
        sample = mask[sample]

    palette = KMedoids(n_clusters=num_colors, random_state=42, init='k-medoids++').fit(sample).cluster_centers_
    distances = np.linalg.norm(mask[:, None, :] - palette[None, :, :], axis=2)
    palette = mask[np.argmin(distances, axis=0)]

    non_black = ~(palette == (0,0,0,255)).all(axis=-1)
    palette = palette[non_black]

    return palette


def show_palette_array(palette, size=(8, 2)):
    """ Display colors from an Nx3 array (values in [0,1]) as a horizontal palette. """
    n = palette.shape[0]
    fig, ax = plt.subplots(figsize=size)
    ax.imshow([palette], extent=[0, n, 0, 1])
    ax.set_xticks(range(n))
    ax.set_yticks([])
    ax.set_xticklabels([str(i) for i in range(n)])
    plt.show()


def find_main_contours(mask, palette, pad=(10,10)):
    main_contours = []

    for label in palette:
        if (label == 0).all() or (label == 255).all():
            continue

        binary = (mask == label).all(axis=-1)
        binary = np.pad(binary, (pad, pad), mode='constant', constant_values=0)
        binary = morphology.binary_closing(binary, morphology.disk(1))

        labels = measure.label(binary)
        region_props = measure.regionprops(labels, intensity_image=binary)
        prop = max(region_props, key=lambda x: x.area)

        # find perimeter contour of the region prop, ignoring holes
        binary_filled = morphology.remove_small_holes(labels == prop.label, area_threshold=1000)
        contours = measure.find_contours(binary_filled, 0.5)
        contour = max(contours, key=len)  # Get the longest contour
        contour -= np.array(pad)  # remove padding

        main_contours.append(contour)

    return main_contours


def find_contact_points(a_contour, b_contour, tolerance=25):
    ab = cdist(a_contour, b_contour)

    def _find_contacts(a, b, ab):
        b_closest_to_a_idx = ab.argmin(axis=1, keepdims=True)
        b_closest_to_a_dist = np.take_along_axis(ab, b_closest_to_a_idx, axis=1).squeeze()

        good_idx = np.where(b_closest_to_a_dist < tolerance)[0]
        good_as = a[good_idx]
        good_bs = b[b_closest_to_a_idx[good_idx, 0]]

        good_contacts = (good_as + good_bs) // 2
        good_contacts = good_contacts.astype(int)
        return good_contacts

    ab_contacts = _find_contacts(a_contour, b_contour, ab)
    ba_contacts = _find_contacts(b_contour, a_contour, ab.T)

    contacts = np.vstack((ab_contacts, ba_contacts))
    contacts = np.unique(contacts, axis=0)

    return contacts


def find_all_contact_points(contours, tolerance=5):
    n = len(contours)
    if n < 2:
        print("[WARNING] Not enough contours to find contact points.")
        return np.empty((0, 2), dtype=int)

    contacts = [find_contact_points(contours[i], contours[j], tolerance=tolerance) for i in range(n) for j in range(i + 1, n)]
    contacts = np.vstack(contacts)
    contacts = np.unique(contacts, axis=0)
    return contacts


def show_contours_and_contacts(mask, contours, contacts, palette):
    plt.figure(figsize=(10,10))
    plt.imshow(mask, alpha=0.5)
    for c, p in zip(contours, palette):
        plt.plot(c[:,1], c[:,0], c=0.5*p/255)
    plt.scatter(contacts[:,1], contacts[:,0], s=6, c='r', marker='+', zorder=100)
    plt.axis('off')


def contour_centroid(points):
    """
    points: (N, 2) numpy array of polygon vertices
    """
    y = points[:, 0]
    x = points[:, 1]
    # Wrap-around
    y1 = np.roll(y, -1)
    x1 = np.roll(x, -1)

    cross = x * y1 - x1 * y
    area = np.sum(cross) / 2.0

    Cy = np.sum((y + y1) * cross) / (6.0 * area)
    Cx = np.sum((x + x1) * cross) / (6.0 * area)

    return Cy, Cx


def contour_intersect_area(a, b):
    """ a and b are contours found by find_contours()."""
    a_poly = shapely.geometry.Polygon(a[:,::-1])
    b_poly = shapely.geometry.Polygon(b[:,::-1])
    if a_poly.intersects(b_poly):
        intersection = a_poly.intersection(b_poly)
        return intersection.area
    return -1


class ApplyToRGB:
    """
    A wrapper that applies a given transform to the RGB channels of an RGBA image tensor.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (Tensor): An RGBA image tensor of shape [4, H, W].

        Returns:
            Tensor: The transformed RGBA image tensor.
        """
        # Ensure the input is a 4-channel tensor
        if img_tensor.shape[0] != 4:
            # If not 4 channels, apply transform to the whole image (e.g., for RGB)
            return self.transform(img_tensor)

        # 1. Separate the RGB and Alpha channels
        rgb_channels = img_tensor[:3, :, :]
        alpha_channel = img_tensor[3:, :, :] # Shape [1, H, W]

        # 2. Apply the wrapped transform to the RGB channels
        transformed_rgb = self.transform(rgb_channels)

        # 3. Recombine the transformed RGB with the original Alpha
        transformed_rgba = torch.cat([transformed_rgb, alpha_channel], dim=0)

        return transformed_rgba
    

def create_prominent_color_masks(image_rgba, area_cutoff=0.01):
    """
    Creates binary masks for the most prominent colors in an RGBA image.

    It ignores transparent pixels and generates masks only for the most frequent
    colors that account for a given percentile of the total pixels.

    Args:
        image: Input RGBA image as a numpy array.
        area_cutoff (float): Percentile cutoff (between 0 and 1) to determine prominent colors.
        output_dir (str): Directory to save the output masks.
    """

    # 2. Filter out transparent pixels (where alpha=0)
    alpha_channel = image_rgba[:, :, 3]
    foreground_mask = (alpha_channel > 0) & (image_rgba[:, :, :3].sum(axis=2) > 0)  # Non-transparent and non-black pixels
    image_bgr = image_rgba[:, :, :3]
    
    # Get a flat list of BGR colors for only the visible pixels
    foreground_pixels = image_bgr[foreground_mask]

    # 3. Get the color histogram
    # np.unique with return_counts is a fast way to get a histogram of unique items
    unique_colors, counts = np.unique(foreground_pixels, axis=0, return_counts=True)

    # 4. Get colors whose frequency is above the area_cutoff percentile
    total_pixels = foreground_pixels.shape[0]
    valid_pixels = counts > total_pixels * area_cutoff

    # These are the colors we will generate masks for
    prominent_colors = unique_colors[valid_pixels]

    # 5. Generate a mask for each prominent color
    masks = []
    for color in prominent_colors:

        # Use cv2.inRange for a fast, exact match
        mask = cv2.inRange(image_bgr, lowerb=color, upperb=color)
        masks.append(mask)

    return masks


def erosion(src, erosion_size, erosion_shape=cv2.MORPH_RECT):    
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
    erosion_dst = cv2.erode(src, element)
    return erosion_dst