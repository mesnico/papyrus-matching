#!/usr/bin/env python3

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from papyrus_matching.loader import utils


BLACK_RGBA = np.array([0, 0, 0, 255], dtype=np.uint8)
TRANSPARENT_RGBA = np.array([0, 0, 0, 0], dtype=np.uint8)


@dataclass
class FragmentDraft:
    source_id: str
    mask_color_rgba: list[int]
    mask: np.ndarray
    contour: np.ndarray
    bbox_yxhw: list[int]
    sort_key: tuple[int, int]


def read_rgba_8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def save_rgba(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGBA").save(path)


def resolve_mask_path(data_root: Path, source_id: str) -> Path:
    mask_dir = data_root / "mask"
    candidates = [
        mask_dir / f"{source_id}_mask.png",
        mask_dir / f"{source_id} mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    suffixes = (f"{source_id}_mask.png", f"{source_id} mask.png")
    matches = sorted(path for path in mask_dir.glob("*.png") if path.name in suffixes)
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"No mask found for '{source_id}' in {mask_dir}")


def normalize_source_id(line: str) -> str:
    source_id = line.strip()
    if source_id.lower().endswith(".png"):
        source_id = source_id[:-4]
    return source_id


def foreground_mask(mask_rgba: np.ndarray) -> np.ndarray:
    alpha = mask_rgba[..., 3] > 0
    black = np.all(mask_rgba[..., :3] == 0, axis=-1)
    return alpha & ~black


def dedupe_palette(mask_rgba: np.ndarray, palette: np.ndarray, min_distance: float = 12.0) -> np.ndarray:
    if len(palette) == 0:
        return palette.astype(np.uint8)

    fg = foreground_mask(mask_rgba)
    pixels = mask_rgba[fg, :3].astype(np.float32)
    colors = palette[:, :3].astype(np.float32)
    if len(pixels) == 0:
        return palette.astype(np.uint8)

    nearest = np.argmin(np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2), axis=1)
    counts = np.bincount(nearest, minlength=len(palette))
    order = np.argsort(counts)[::-1]

    kept = []
    for idx in order:
        color = palette[idx].astype(np.uint8)
        if np.all(color[:3] == 0):
            continue
        if all(np.linalg.norm(color[:3].astype(float) - other[:3].astype(float)) >= min_distance for other in kept):
            color[3] = 255
            kept.append(color)

    if not kept:
        return np.empty((0, 4), dtype=np.uint8)
    return np.array(kept, dtype=np.uint8)


def fallback_palette(mask_rgba: np.ndarray, max_colors: int, min_fragment_area: int) -> np.ndarray:
    fg = foreground_mask(mask_rgba)
    pixels = mask_rgba[fg].copy()
    if len(pixels) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    pixels[:, 3] = 255
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    valid = counts >= min_fragment_area
    colors = colors[valid]
    counts = counts[valid]
    if len(colors) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    order = np.argsort(counts)[::-1][:max_colors]
    return colors[order].astype(np.uint8)


def get_palette(mask_rgba: np.ndarray, mask_path: Path, min_fragment_area: int) -> np.ndarray:
    fg = foreground_mask(mask_rgba)
    visible = mask_rgba[fg]
    if len(visible) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    guessed_colors = max(1, utils.guess_num_colors(mask_rgba, mask_path))
    unique_visible = np.unique(visible, axis=0)
    num_colors = min(guessed_colors, len(unique_visible))

    try:
        palette = utils.get_dominant_mask_colors(mask_rgba, num_colors=num_colors)
    except Exception as exc:
        print(f"[WARNING] Falling back to histogram palette for {mask_path.name}: {exc}")
        palette = fallback_palette(mask_rgba, max_colors=guessed_colors, min_fragment_area=min_fragment_area)

    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 4:
        return np.empty((0, 4), dtype=np.uint8)

    palette[:, 3] = 255
    palette = palette[~np.all(palette[:, :3] == 0, axis=1)]
    return dedupe_palette(mask_rgba, palette)


def quantize_mask(mask_rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    quantized = np.zeros_like(mask_rgba, dtype=np.uint8)
    if len(palette) == 0:
        return quantized

    alpha = mask_rgba[..., 3] > 0
    black = alpha & np.all(mask_rgba[..., :3] == 0, axis=-1)
    quantized[black] = BLACK_RGBA

    fg = alpha & ~black
    if not np.any(fg):
        return quantized

    pixels = mask_rgba[fg, :3].astype(np.float32)
    colors = palette[:, :3].astype(np.float32)
    nearest = np.argmin(np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2), axis=1)
    quantized[fg] = palette[nearest]
    quantized[fg, 3] = 255
    return quantized


def largest_component(binary: np.ndarray) -> np.ndarray:
    binary_u8 = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=bool)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_idx


def bbox_yxhw(binary: np.ndarray) -> list[int] | None:
    coords = np.argwhere(binary)
    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return [int(y0), int(x0), int(y1 - y0), int(x1 - x0)]


def build_fragment_drafts(
    source_id: str,
    quantized_mask: np.ndarray,
    palette: np.ndarray,
    min_fragment_area: int,
) -> list[FragmentDraft]:
    valid_palette = []
    component_masks = []

    for color in palette:
        color_mask = np.all(quantized_mask == color, axis=-1)
        component = largest_component(color_mask)
        area = int(component.sum())
        if area < min_fragment_area:
            continue
        valid_palette.append(color)
        component_masks.append(component)

    if not valid_palette:
        return []

    valid_palette_array = np.asarray(valid_palette, dtype=np.uint8)
    contour_mask = np.zeros_like(quantized_mask, dtype=np.uint8)
    for color, component in zip(valid_palette_array, component_masks):
        contour_mask[component] = color

    contours = utils.find_main_contours(contour_mask, valid_palette_array)

    drafts = []
    for color, component, contour in zip(valid_palette_array, component_masks, contours):
        bbox = bbox_yxhw(component)
        if bbox is None:
            continue
        y0, x0, _, _ = bbox
        drafts.append(
            FragmentDraft(
                source_id=source_id,
                mask_color_rgba=[int(v) for v in color],
                mask=component,
                contour=contour,
                bbox_yxhw=bbox,
                sort_key=(int(x0), int(y0)),
            )
        )

    drafts.sort(key=lambda item: item.sort_key)
    return drafts


def contour_min_distance(a_contour: np.ndarray, b_contour: np.ndarray, tolerance: float, max_cdist_pairs: int) -> float:
    if len(a_contour) == 0 or len(b_contour) == 0:
        return float("inf")

    if len(a_contour) * len(b_contour) <= max_cdist_pairs:
        # Reuse the existing contact heuristic on moderate contour sizes.
        utils.find_contact_points(a_contour, b_contour, tolerance=tolerance)

    tree = cKDTree(b_contour)
    distances, _ = tree.query(a_contour, k=1)
    return float(np.min(distances)) if len(distances) else float("inf")


def extract_fragment_image(rgba: np.ndarray, mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    y0, x0, height, width = bbox
    crop_rgba = rgba[y0 : y0 + height, x0 : x0 + width].copy()
    crop_mask = mask[y0 : y0 + height, x0 : x0 + width]
    crop_rgba[~crop_mask] = TRANSPARENT_RGBA
    crop_rgba[crop_rgba[..., 3] == 0, :3] = 0
    return crop_rgba


def prepare_source(
    data_root: Path,
    output_dir: Path,
    source_id: str,
    adjacency_tolerance: float,
    min_fragment_area: int,
    max_cdist_pairs: int,
) -> tuple[list[dict], list[dict], dict]:
    rgba_path = data_root / "rgba" / f"{source_id}.png"
    mask_path = resolve_mask_path(data_root, source_id)

    if not rgba_path.exists():
        raise FileNotFoundError(f"No RGBA image found for '{source_id}' at {rgba_path}")

    rgba = read_rgba_8(rgba_path)
    mask_rgba = read_rgba_8(mask_path)
    if rgba.shape[:2] != mask_rgba.shape[:2]:
        raise ValueError(f"RGBA and mask sizes differ for '{source_id}': {rgba.shape[:2]} != {mask_rgba.shape[:2]}")

    palette = get_palette(mask_rgba, mask_path, min_fragment_area=min_fragment_area)
    quantized_mask = quantize_mask(mask_rgba, palette)
    drafts = build_fragment_drafts(
        source_id=source_id,
        quantized_mask=quantized_mask,
        palette=palette,
        min_fragment_area=min_fragment_area,
    )

    fragments = []
    for index, draft in enumerate(drafts, start=1):
        fragment_id = f"{source_id}_{index}"
        output_path = output_dir / f"{fragment_id}.png"
        crop = extract_fragment_image(rgba, draft.mask, draft.bbox_yxhw)
        save_rgba(output_path, crop)

        y0, x0, _, _ = draft.bbox_yxhw
        fragments.append(
            {
                "id": fragment_id,
                "source_id": source_id,
                "path": output_path.as_posix(),
                "index": index,
                "origin_yx": [int(y0), int(x0)],
                "bbox_yxhw": draft.bbox_yxhw,
                "mask_color_rgba": draft.mask_color_rgba,
            }
        )

    pairs = []
    for (idx_a, draft_a), (idx_b, draft_b) in itertools.combinations(enumerate(drafts), 2):
        distance = contour_min_distance(
            draft_a.contour,
            draft_b.contour,
            tolerance=adjacency_tolerance,
            max_cdist_pairs=max_cdist_pairs,
        )
        if distance > adjacency_tolerance:
            continue

        fragment_a = fragments[idx_a]
        fragment_b = fragments[idx_b]
        origin_a = np.asarray(fragment_a["origin_yx"], dtype=float)
        origin_b = np.asarray(fragment_b["origin_yx"], dtype=float)
        pairs.append(
            {
                "fragment_a": fragment_a["id"],
                "fragment_b": fragment_b["id"],
                "source_id": source_id,
                "offset_b_on_a_yx": [float(v) for v in (origin_b - origin_a)],
                "distance_px": float(distance),
            }
        )

    source_record = {
        "id": source_id,
        "rgba_path": rgba_path.as_posix(),
        "mask_path": mask_path.as_posix(),
        "num_fragments": len(fragments),
        "num_pairs": len(pairs),
    }
    return fragments, pairs, source_record


def load_split(data_root: Path, split: str) -> list[str]:
    split_path = Path(split)
    if not split_path.is_absolute():
        split_path = data_root / split

    with open(split_path, "r", encoding="utf-8") as handle:
        return [normalize_source_id(line) for line in handle if normalize_source_id(line)]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract validation fragments and GT joins from colored masks.")
    parser.add_argument("--data-root", type=Path, default=Path("data/unified"), help="Root containing rgba/, mask/, and split file.")
    parser.add_argument("--split", type=str, default="val.txt", help="Split file path or name relative to data-root.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/unified/val_fragments"), help="Directory for extracted fragment PNGs.")
    parser.add_argument("--gt-json", type=Path, default=Path("data/unified/val_ground_truth.json"), help="Output ground-truth JSON path.")
    parser.add_argument("--adjacency-tolerance", type=float, default=25.0, help="Contour distance threshold for GT adjacency.")
    parser.add_argument("--min-fragment-area", type=int, default=100, help="Minimum connected mask area to keep as a fragment.")
    parser.add_argument("--max-cdist-pairs", type=int, default=10_000_000, help="Maximum contour pair products before using KD-tree distance only.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    gt_json = args.gt_json

    source_ids = load_split(data_root, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_json.parent.mkdir(parents=True, exist_ok=True)

    all_fragments = []
    all_pairs = []
    sources = []
    errors = []

    for source_id in source_ids:
        print(f"[INFO] Preparing {source_id}")
        try:
            fragments, pairs, source_record = prepare_source(
                data_root=data_root,
                output_dir=output_dir,
                source_id=source_id,
                adjacency_tolerance=args.adjacency_tolerance,
                min_fragment_area=args.min_fragment_area,
                max_cdist_pairs=args.max_cdist_pairs,
            )
        except Exception as exc:
            print(f"[ERROR] Failed {source_id}: {exc}")
            errors.append({"source_id": source_id, "error": str(exc)})
            continue

        all_fragments.extend(fragments)
        all_pairs.extend(pairs)
        sources.append(source_record)
        print(f"[INFO]   wrote {len(fragments)} fragments and {len(pairs)} adjacent GT pairs")

    payload = {
        "version": 1,
        "coordinate_system": "yx",
        "data_root": data_root.as_posix(),
        "split": str(args.split),
        "output_dir": output_dir.as_posix(),
        "adjacency_tolerance": float(args.adjacency_tolerance),
        "min_fragment_area": int(args.min_fragment_area),
        "sources": sources,
        "fragments": all_fragments,
        "pairs": all_pairs,
        "errors": errors,
    }

    with open(gt_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[INFO] Saved {len(all_fragments)} fragments and {len(all_pairs)} GT pairs to {gt_json}")
    if errors:
        print(f"[WARNING] {len(errors)} source(s) failed; see errors in the GT JSON.")


if __name__ == "__main__":
    main()
