from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


MODELS={
    'patch-encoder.pth': 'https://github.com/fabiocarrara/papyrus-matching/releases/download/v0.1.0/patch-encoder.pth',
}


class FragmentMatcher(object):
    """Matches two fragments."""
    def __init__(
        self,
        model='patch-encoder.pth',
        device='cpu',
        patch_size=64,
        stride=32,
        min_availability=0.7,
    ):
        super(FragmentMatcher, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.min_availability = min_availability

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # RGBA
        ])

        model_path = Path(model)
        if not model_path.exists():
            torch.hub.download_url_to_file(MODELS[model], model_path)

        self.encoder = torch.load(model_path).eval().to(self.device)

    @staticmethod
    def availability(im, top, left, bottom, right):
        integral = im[bottom, right] - im[bottom, left] - im[top, right] + im[top, left]
        height, width = bottom - top + 1, right - left + 1

        return integral / (height * width)

    def find_patches(self, image, side='right'):
        alpha = image[:, :, 3]
        height, width = alpha.shape

        integral_alpha = alpha.astype(np.float32).cumsum(axis=0).cumsum(axis=1) / 255.

        patches = []
        patches_pos = []

        def _check_availability(y, x):
            av = self.availability(integral_alpha, y, x, y + self.patch_size - 1, x + self.patch_size - 1)
            if av >= self.min_availability:
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch = self.transform(patch)
                patch_pos = (y, x)

                patches.append(patch)
                patches_pos.append(patch_pos)
                return True
            return False

        if side in ('left', 'right'):
            y_range = range(0, height - self.patch_size, self.stride)
            x_range = range(width - self.patch_size, 0, -self.stride) if side == 'right' else range(0, width - self.patch_size, self.stride)

            for y in y_range:
                for x in x_range:
                    if _check_availability(y, x):
                        break

        elif side in ('top', 'bottom'):
            x_range = range(0, width - self.patch_size, self.stride)
            y_range = range(height - self.patch_size, 0, -self.stride) if side == 'bottom' else range(0, height - self.patch_size, self.stride)

            for x in x_range:
                for y in y_range:
                    if _check_availability(y, x):
                        break
        else:
            raise ValueError(f"Invalid side: {side}, must be one of 'left', 'right', 'top', 'bottom'.")

        patches = torch.stack(patches)
        patches_pos = np.array(patches_pos)
        return patches, patches_pos

    @torch.no_grad()
    def match(self, fragmentA, fragmentB, direction='horizontal'):

        sideA = 'right' if direction == 'horizontal' else 'bottom'
        sideB = 'left'  if direction == 'horizontal' else 'top'

        patchesA, positionsA = self.find_patches(fragmentA, side=sideA)
        patchesB, positionsB = self.find_patches(fragmentB, side=sideB)

        numA = len(patchesA)
        numB = len(patchesB)

        patchesA = patchesA.to(self.device)
        patchesB = patchesB.to(self.device)

        codesA = self.encoder(patchesA).flatten(start_dim=1)
        codesB = self.encoder(patchesB).flatten(start_dim=1)

        codesA = F.normalize(codesA)
        codesB = F.normalize(codesB)

        scoresAB = torch.matmul(codesA, codesB.T)

        # each diagonal of scoresAB scores a different vertical dispacement of the two fragments
        scored_displacements = [(d, scoresAB.diag(d).mean().item()) for d in range(- numA + 1, numB)]
        # ranked_displacements = sorted(scored_displacements, key=lambda x: x[1], reverse=True)

        return positionsA, positionsB, scored_displacements, scoresAB


if __name__ == "__main__":
    from skimage import draw
    from skimage.io import imread, imsave
    from skimage.transform import rescale
    import matplotlib.pyplot as plt
    import pandas as pd

    fragL = imread('data/test_fragments/test_fragment_easy1_L_back.png')
    fragR = imread('data/test_fragments/test_fragment_easy1_R_back.png')

    # fragL = imread('data/test_fragments/test_fragment_hard1_L_back.png')
    # fragR = imread('data/test_fragments/test_fragment_hard1_R_back.png')

    matcher = FragmentMatcher(min_availability=0.6)
    posL, posR, scored_displacements, scoresLR = matcher.match(fragL, fragR, direction='horizontal')
    ranked_displacements = sorted(scored_displacements, key=lambda x: x[1], reverse=True)

    scored_displacements = pd.DataFrame(scored_displacements, columns=['displacement', 'score'])
    scored_displacements.plot(x='displacement', y='score')
    plt.savefig('dy.png')

    Lh, Lw = fragL.shape[:2]
    Rh, Rw = fragR.shape[:2]

    rng = np.random.default_rng(7)

    for i, (dy, score) in enumerate(ranked_displacements[:5]):
    # for i, (dy, score) in enumerate(((43, 1.),)):
        print(dy, score)
        Ly0, Lx0 = 0, 0
        Ly1, Lx1 = Lh, Lw
        Ry0, Rx0 = - dy * matcher.stride, Lw
        Ry1, Rx1= Ry0 + Rh, Rx0 + Rw

        h = max(Ly1, Ry1) - min(Ly0, Ry0)
        w = max(Lx1, Rx1) - min(Lx0, Rx0)
        preview = np.full((h, w, 4), (0, 0, 0, 1), dtype=fragL.dtype)

        Oy, Ox = min(Ly0, Ry0), min(Lx0, Rx0)
        Ly0, Lx0 = Ly0 - Oy, Lx0 - Ox
        Ly1, Lx1 = Ly1 - Oy, Lx1 - Ox
        Ry0, Rx0 = Ry0 - Oy, Rx0 - Ox
        Ry1, Rx1 = Ry1 - Oy, Rx1 - Ox

        preview[Ly0:Ly1, Lx0:Lx1] = fragL
        preview[Ry0:Ry1, Rx0:Rx1] = fragR

        numL = len(posL)
        numR = len(posR)

        # maxMatches = min(numL, numR) - abs(dy)
        maxMatches = min(numL, numR, numL+dy, numR-dy)
        matchesL = posL[min(dy, 0):min(dy, 0) + maxMatches] + np.array([Ly0, Lx0])
        matchesR = posR[max(dy, 0):max(dy, 0) + maxMatches] + np.array([Ry0, Rx0])
        scores = scoresLR.diag(dy).tolist()

        for left_start, right_start, lr_score in zip(matchesL, matchesR, scores):
            left_end = left_start + matcher.patch_size - 1
            right_end = right_start + matcher.patch_size - 1

            color = rng.integers(0, 256, 4)
            color[3] = 255

            coords = draw.rectangle(left_start, left_end, shape=preview.shape)
            draw.set_color(preview, coords, color, lr_score)
            coords = draw.rectangle(right_start, right_end, shape=preview.shape)
            draw.set_color(preview, coords, color, lr_score)

        preview = rescale(preview, 0.25, channel_axis=2, anti_aliasing=True)
        imsave(f'match_preview_r{i}.png', preview)