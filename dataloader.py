from dataclasses import dataclass

import numpy as np
from skimage import draw
from skimage.io import imread
from torch.utils.data import Dataset


@dataclass
class Match:
    y: int
    x: int
    aspect: float
    patch_size: int
    gap: float
    direction: str = 'horizontal'

    def region_A(self):
        # smallest side is always the patch size, the other side is computed from the aspect ratio
        height = round(max(1, self.aspect) * self.patch_size)
        width = round(max(1, 1 / self.aspect) * self.patch_size)

        # swap height and width if direction is vertical such that the aspect ratio always refers to the matching side
        if self.direction == 'vertical':
            height, width = width, height

        start = (self.y, self.x)
        end = (self.y + height, self.x + width)
        return start, end

    def region_B(self):
        if self.direction == 'horizontal':
            height = round(max(1, self.aspect) * self.patch_size)
            width = round(max(1, 1 / self.aspect) * self.patch_size)

            offset = width + round(self.gap * self.patch_size)
            start = (self.y, self.x + offset)
            end = (self.y + height, self.x + width + offset)
        else:
            width = round(max(1, self.aspect) * self.patch_size)
            height = round(max(1, 1 / self.aspect) * self.patch_size)

            offset = height + round(self.gap * self.patch_size)
            start = (self.y + offset, self.x)
            end = (self.y + height + offset, self.x + width)

        return start, end

    @staticmethod
    def availability(im, start, end):
        top, left = start
        bottom, right = end

        integral = im[bottom, right] - im[bottom, left] - im[top, right] + im[top, left]
        height, width = bottom - top + 1, right - left + 1

        return integral / (height * width)

    def is_good(self, im, mask=None, threshold=0.7):
        start_A, end_A = self.region_A()
        start_B, end_B = self.region_B()

        (y0_A, x0_A), (y1_A, x1_A) = start_A, end_A
        (y0_B, x0_B), (y1_B, x1_B) = start_B, end_B

        height, width = im.shape
        if not (
            (0 <= y0_A < height) and (0 <= y1_A < height) and
            (0 <= x0_A < width) and (0 <= x1_A < width) and
            (0 <= y0_B < height) and (0 <= y1_B < height) and
            (0 <= x0_B < width) and (0 <= x1_B < width)
        ):
            return False

        available_1 = self.availability(im, start_A, end_A)
        available_2 = self.availability(im, start_B, end_B)

        if (available_1 < threshold) or (available_2 < threshold):
            return False

        if mask is not None:
            invalid_A = self.availability(mask, start_A, end_A) > 0
            invalid_B = self.availability(mask, start_B, end_B) > 0

            if invalid_A or invalid_B:
                return False

        return True

    def draw(self, im, color):
        start_A, end_A = self.region_A()
        start_B, end_B = self.region_B()

        coords = draw.rectangle(start_A, end_A, shape=im.shape)
        draw.set_color(im, coords, color, 0.3)

        coords = draw.rectangle(start_B, end_B, shape=im.shape)
        draw.set_color(im, coords, color, 0.3)


class PapyrMatchesDataset(Dataset):
    def __init__(
        self,
        image_path,
        mask_path=None,
        patch_size=64,
        stride=None,
        aspect=0.5,  # aspect ratio of left/top and right/bottom parts
                     # (length of matching side / length of non-matching side,
                     # considering that the smallest side is always the patch size)
        min_gap=0.125,  # minimum gap between left/top and right/bottom parts in fraction of patch size
        max_gap=1.500,  # maximum gap between left/top and right/bottom parts in fraction of patch size
        step_gap=0.125, # step for exploring gap values
        min_availability=0.7,  # minimum fraction of non-missing papyrus to consider a region usable
        transform=None
    ):
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.aspect = aspect
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.step_gap = step_gap
        self.min_availability = min_availability
        self.transform = transform

        # load image data
        self.image = imread(image_path)  # RGBA image
        self.image[:, :, -1] = np.where(self.image[:, :, -1] >= 127, np.uint8(255), np.uint8(0))  # binarize alpha

        # load mask data
        if mask_path is None:
            mask_path = image_path.parent / 'mask' / (image_path.stem + '_mask.png')
            mask_path = mask_path if mask_path.exists() else None

        self.mask = imread(mask_path) if mask_path else None

        # matches data
        self.matches = self._find_matches()

    def _find_matches(self):
        # compute integral image of alpha channel to compute availability efficiently
        self.integral_alpha = self.image[:, :, -1].astype(np.float32).cumsum(axis=0).cumsum(axis=1) / 255.
        self.integral_mask = None
        if self.mask is not None:
            self.integral_mask = np.all(self.mask == (0, 0, 0, 255), axis=-1).astype(np.float32).cumsum(axis=0).cumsum(axis=1)

        matches = []
        height, width = self.image.shape[:2]
        for gap in np.arange(self.min_gap, self.max_gap + 0.001, self.step_gap):
            for y in range(0, height, self.stride):
                for x in range(0, width, self.stride):
                    for direction in ('horizontal', 'vertical'):
                        match = Match(y, x, self.aspect, self.patch_size, gap, direction)
                        if match.is_good(self.integral_alpha, mask=self.integral_mask, threshold=self.min_availability):
                            matches.append(match)

        return matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        match = self.matches[index]

        (y0_A, x0_A), (y1_A, x1_A) = match.region_A()
        (y0_B, x0_B), (y1_B, x1_B) = match.region_B()

        region_A = self.image[y0_A:y1_A, x0_A:x1_A, :]
        region_B = self.image[y0_B:y1_B, x0_B:x1_B, :]

        if self.transform:
            region_A = self.transform(region_A)
            region_B = self.transform(region_B)

        return region_A, region_B, match