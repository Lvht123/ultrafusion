#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate RGB-T low-light simulation triplets with a robust visible-image
reinforcement step based on synthetic multi-exposure fusion (MEF).

Main pipeline
-------------
1) Load normal visible image and paired thermal image
2) Reinforce the normal visible image first using synthetic MEF
3) Simulate low-light degradation only on the reinforced visible image
4) Save pairs:
      gt/       : reinforced visible image (MEF enhanced)
      low/      : simulated low-light visible image

   (Original visible and SWIR images already exist in the dataset, no need to duplicate.)

This version contains several safety protections to avoid the "all-black gt/low"
problem:
  - milder default exposure synthesis for MEF,
  - Mertens fusion uses a non-zero exposure weight by default,
  - post-fusion brightness sanity check and automatic fallback/rescaling,
  - low-light simulation retries if the sampled degradation is too aggressive.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required for the multi-exposure fusion step. "
        "Please install opencv-python or opencv-python-headless."
    ) from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def set_seed(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)



def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS



def collect_image_files(path: Path) -> List[Path]:
    if path.is_file():
        if not is_image_file(path):
            raise ValueError(f"Not a supported image file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = [p for p in sorted(path.rglob("*")) if is_image_file(p)]
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {path}")
    return files



def build_pairs(visible_path: Path, thermal_path: Path) -> List[Tuple[Path, Path]]:
    """Build visible-thermal pairs for either file-file or dir-dir inputs."""
    if visible_path.is_file() and thermal_path.is_file():
        return [(visible_path, thermal_path)]

    if visible_path.is_dir() and thermal_path.is_dir():
        visible_files = collect_image_files(visible_path)
        thermal_files = collect_image_files(thermal_path)

        thermal_by_name = {p.name: p for p in thermal_files}
        thermal_by_stem = {}
        for p in thermal_files:
            thermal_by_stem.setdefault(p.stem, p)

        pairs: List[Tuple[Path, Path]] = []
        missing: List[Path] = []
        for vp in visible_files:
            tp = thermal_by_name.get(vp.name, None)
            if tp is None:
                tp = thermal_by_stem.get(vp.stem, None)
            if tp is None:
                missing.append(vp)
            else:
                pairs.append((vp, tp))

        if len(pairs) == 0:
            raise FileNotFoundError(
                "No visible-thermal pairs found. Please make sure the two folders use "
                "the same filenames or at least the same filename stems."
            )
        if missing:
            print(f"[Warning] {len(missing)} visible images have no matched thermal image and will be skipped.")
        return pairs

    raise ValueError(
        "visible_path and thermal_path must both be files or both be directories.\n"
        f"visible_path={visible_path}\nthermal_path={thermal_path}"
    )



def pil_to_rgb_np(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)



def pil_to_thermal_np(path: Path, thermal_mode: str = "keep") -> np.ndarray:
    img = Image.open(path)
    if thermal_mode == "rgb":
        img = img.convert("RGB")
    elif thermal_mode == "gray":
        img = img.convert("L")
    elif thermal_mode == "keep":
        if img.mode in {"RGBA", "LA", "P"}:
            img = img.convert("RGB")
    else:
        raise ValueError(f"Unsupported thermal_mode: {thermal_mode}")
    return np.asarray(img)



def resize_np_like_thermal(thermal: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    img = Image.fromarray(thermal)
    img = img.resize((target_w, target_h), resample=Image.BICUBIC)
    return np.asarray(img)



def make_positions(length: int, patch_size: int, stride: int, keep_edge: bool) -> List[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    if keep_edge and positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions



def iter_patch_coords(h: int, w: int, patch_size: int, stride: int, keep_edge: bool) -> Iterable[Tuple[int, int]]:
    ys = make_positions(h, patch_size, stride, keep_edge)
    xs = make_positions(w, patch_size, stride, keep_edge)
    for y in ys:
        for x in xs:
            yield y, x



def crop_patch(img: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    return img[y:y + patch_size, x:x + patch_size].copy()





# -----------------------------------------------------------------------------
# Image statistics / safeguards
# -----------------------------------------------------------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)



def luma_mean_rgb(rgb_u8: np.ndarray) -> float:
    x = rgb_u8.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return float(y.mean())



def safe_gain_rescale(rgb_u8: np.ndarray, target_mean: float, min_gain: float = 0.8, max_gain: float = 2.5) -> np.ndarray:
    cur_mean = luma_mean_rgb(rgb_u8)
    if cur_mean < 1e-6:
        return rgb_u8.copy()
    gain = np.clip(target_mean / cur_mean, min_gain, max_gain)
    out = np.clip(rgb_u8.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
    return out


# -----------------------------------------------------------------------------
# Multi-exposure fusion for visible-image reinforcement
# -----------------------------------------------------------------------------

def apply_ev_and_gamma(rgb_u8: np.ndarray, ev: float, gamma: float) -> np.ndarray:
    """
    Generate a synthetic exposure variant from a single RGB image.

    ev    : exposure value shift. Positive => brighter, negative => darker.
    gamma : tone adjustment. gamma < 1 brightens shadows, gamma > 1 darkens.
    """
    x = rgb_u8.astype(np.float32) / 255.0
    x = x * (2.0 ** ev)
    x = clamp01(x)
    x = np.power(np.clip(x, 1e-4, 1.0), gamma)
    x = clamp01(x)
    return (x * 255.0 + 0.5).astype(np.uint8)



def build_synthetic_exposure_stack(
    rgb_u8: np.ndarray,
    ev_list: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    gamma_list: Optional[Sequence[float]] = None,
) -> List[np.ndarray]:
    """
    Create a synthetic exposure stack from a single RGB image.

    Defaults are intentionally mild to avoid generating overly dark variants.
    """
    if gamma_list is None:
        gamma_list = (1.10, 1.05, 1.00, 0.95, 0.90)

    if len(ev_list) != len(gamma_list):
        raise ValueError("ev_list and gamma_list must have the same length.")

    variants: List[np.ndarray] = []
    for ev, gamma in zip(ev_list, gamma_list):
        variants.append(apply_ev_and_gamma(rgb_u8, ev=ev, gamma=gamma))

    # Ensure the original image is present in the stack.
    variants.append(rgb_u8.copy())
    return variants



def mild_clahe_on_luminance(rgb_u8: np.ndarray, clip_limit: float = 1.0, tile_grid_size: int = 16) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb



def reinforce_visible_with_mef(
    rgb_u8: np.ndarray,
    ev_list: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    gamma_list: Optional[Sequence[float]] = None,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
    use_clahe_post: bool = False,
    clahe_clip_limit: float = 1.0,
    clahe_tile_grid_size: int = 16,
    blend_with_original: float = 0.85,
) -> np.ndarray:
    """
    Reinforce a normal visible image using synthetic MEF.

    Safety measures:
      - uses mild synthetic exposure variants,
      - fuses with non-zero exposure weighting by default,
      - checks the fused result brightness,
      - rescales or falls back to the original image if necessary.
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {rgb_u8.shape}")

    original = rgb_u8.copy()
    orig_mean = luma_mean_rgb(original)

    stack_rgb = build_synthetic_exposure_stack(rgb_u8, ev_list=ev_list, gamma_list=gamma_list)
    stack_bgr_u8 = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in stack_rgb]

    merge_mertens = cv2.createMergeMertens(
        contrast_weight=float(contrast_weight),
        saturation_weight=float(saturation_weight),
        exposure_weight=float(exposure_weight),
    )

    fused_bgr = merge_mertens.process(stack_bgr_u8)
    fused_bgr = clamp01(fused_bgr)
    fused_bgr_u8 = (fused_bgr * 255.0 + 0.5).astype(np.uint8)
    fused_rgb = cv2.cvtColor(fused_bgr_u8, cv2.COLOR_BGR2RGB)

    # If fusion accidentally becomes too dark, fall back to the original.
    fused_mean = luma_mean_rgb(fused_rgb)
    if fused_rgb.max() <= 5 or fused_mean < max(5.0, 0.15 * orig_mean):
        fused_rgb = original.copy()
        fused_mean = orig_mean

    # Bring the fused image back to a sensible brightness range.
    target_mean = max(orig_mean * 0.98, 45.0)
    fused_rgb = safe_gain_rescale(fused_rgb, target_mean=target_mean, min_gain=0.9, max_gain=2.2)

    if use_clahe_post:
        fused_rgb = mild_clahe_on_luminance(
            fused_rgb,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

    # A small blend with the original image improves stability and naturalness.
    alpha = float(np.clip(blend_with_original, 0.0, 1.0))
    fused_rgb = np.clip(alpha * fused_rgb.astype(np.float32) + (1.0 - alpha) * original.astype(np.float32),
                        0.0, 255.0).astype(np.uint8)
    fused_rgb = cv2.bilateralFilter(
    fused_rgb,
    d=5,
    sigmaColor=20,
    sigmaSpace=20
)
    # Final guard.
    final_mean = luma_mean_rgb(fused_rgb)
    if fused_rgb.max() <= 5 or final_mean < 5.0:
        fused_rgb = original.copy()

    return fused_rgb


# -----------------------------------------------------------------------------
# VIIS-style low-light simulation with safety retry
# -----------------------------------------------------------------------------

def adjust_contrast_uint8(image: np.ndarray, factor: float) -> np.ndarray:
    img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    img = ImageEnhance.Contrast(img).enhance(factor)
    return np.asarray(img, dtype=np.uint8)



def gamma_darkening_uint8(image: np.ndarray, gamma: float) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)



def poisson_noise_viis_uint8(image: np.ndarray, noise_level: float = 5.0) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = np.random.poisson(x * noise_level).astype(np.float32) / float(noise_level)
    noisy = np.clip(x + noise, 0.0, 1.0) * 255.0
    return noisy.astype(np.uint8)



def gaussian_noise_uint8(image: np.ndarray, sigma: float = 10.0, mean: float = 0.0) -> np.ndarray:
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0.0, 255.0)
    return noisy.astype(np.uint8)



def salt_pepper_noise_uint8(image: np.ndarray, salt_prob: float, pepper_prob: float) -> np.ndarray:
    noisy = image.copy()
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    return noisy.astype(np.uint8)



def _simulate_low_light_once(
    gt_patch: np.ndarray,
    contrast_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    poisson_level: float,
    gaussian_sigma: float,
    use_salt_pepper: bool,
) -> np.ndarray:
    contrast_factor = random.uniform(*contrast_range)
    gamma = random.uniform(*gamma_range)

    low = adjust_contrast_uint8(gt_patch, contrast_factor)
    low = gamma_darkening_uint8(low, gamma)
    low = poisson_noise_viis_uint8(low, noise_level=poisson_level)
    low = gaussian_noise_uint8(low, sigma=gaussian_sigma)

    if use_salt_pepper:
        salt_prob = 0.2 * random.random()
        pepper_prob = 0.2 * random.random()
        low = salt_pepper_noise_uint8(low, salt_prob=salt_prob, pepper_prob=pepper_prob)

    return low



def simulate_low_light_viis(
    gt_patch: np.ndarray,
    contrast_range: Tuple[float, float] = (0.15, 0.95),
    gamma_range: Tuple[float, float] = (2.2, 6.0),
    poisson_level: float = 5.0,
    gaussian_sigma: float = 10.0,
    use_salt_pepper: bool = False,
    num_retry: int = 3,
) -> np.ndarray:
    """
    Simulate low-light visible patch from a normal-light RGB patch.

    This keeps the VIIS degradation order, but retries with new random samples
    if the result becomes nearly black.
    """
    if gt_patch.ndim != 3 or gt_patch.shape[2] != 3:
        raise ValueError(f"gt_patch must be RGB [H,W,3], got shape={gt_patch.shape}")

    gt_patch = np.clip(gt_patch, 0, 255).astype(np.uint8)
    best = None
    best_mean = -1.0

    for _ in range(max(1, num_retry)):
        low = _simulate_low_light_once(
            gt_patch,
            contrast_range=contrast_range,
            gamma_range=gamma_range,
            poisson_level=poisson_level,
            gaussian_sigma=gaussian_sigma,
            use_salt_pepper=use_salt_pepper,
        )
        cur_mean = luma_mean_rgb(low)
        if cur_mean > best_mean:
            best = low
            best_mean = cur_mean
        if low.max() > 10 and cur_mean >= 4.0:
            return low

    # Safe fallback: use milder degradation if the randomly sampled results are too dark.
    low = adjust_contrast_uint8(gt_patch, factor=max(0.35, contrast_range[0]))
    low = gamma_darkening_uint8(low, gamma=min(3.5, gamma_range[1]))
    low = poisson_noise_viis_uint8(low, noise_level=poisson_level)
    low = gaussian_noise_uint8(low, sigma=gaussian_sigma)

    if low.max() <= 10 or luma_mean_rgb(low) < 4.0:
        # Final emergency fallback: do not let the sample collapse to black.
        low = safe_gain_rescale(best if best is not None else gt_patch, target_mean=12.0, min_gain=1.0, max_gain=4.0)

    return low.astype(np.uint8)


# -----------------------------------------------------------------------------
# I/O helpers and argument parsing
# -----------------------------------------------------------------------------

def save_np_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)



def parse_range(text: str, name: str) -> Tuple[float, float]:
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 2 or vals[0] > vals[1]:
        raise argparse.ArgumentTypeError(f"{name} must be formatted as min,max and min <= max. Got: {text}")
    return vals[0], vals[1]



def parse_float_list(text: str) -> Tuple[float, ...]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
    if len(vals) == 0:
        raise argparse.ArgumentTypeError("The list must contain at least one numeric value.")
    return tuple(vals)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate whole-image RGB-T low-light simulation triplets without cropping."
    )
    parser.add_argument("--visible-path", type=Path, required=True,
                        help="Path to a normal-light visible image or a folder of visible images.")
    parser.add_argument("--thermal-path", type=Path, required=True,
                        help="Path to the paired thermal image or a folder of thermal images.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory. It will contain gt/, thermal/, and low/.")

    # 移除了 --patch-size, --stride 和 --keep-edge，因为不再需要裁剪
    parser.add_argument("--num-aug", type=int, default=1,
                        help="Number of low-light variants to generate per image (different random degradation each time). Default: 1.")
    parser.add_argument("--resize-thermal", action="store_true",
                        help="Resize thermal image to visible image size if their sizes are different.")
    parser.add_argument("--thermal-mode", choices=["keep", "rgb", "gray"], default="keep",
                        help="How to load/save thermal patches. Default: keep. Use rgb if your model expects 3-channel TIR.")

    # Visible reinforcement via synthetic MEF
    parser.add_argument("--disable-visible-reinforce", action="store_true",
                        help="Disable the visible-image reinforcement step. By default it is enabled.")
    parser.add_argument("--mef-ev-list", type=parse_float_list, default=(-1.0, -0.5, 0.0, 0.5, 1.0),
                        help="Comma-separated EV list for synthetic exposure stack. Default: -1,-0.5,0,0.5,1")
    parser.add_argument("--mef-gamma-list", type=parse_float_list, default=(1.10, 1.05, 1.00, 0.95, 0.90),
                        help="Comma-separated gamma list aligned with --mef-ev-list. Default: 1.10,1.05,1.0,0.95,0.90")
    parser.add_argument("--mef-contrast-weight", type=float, default=1.0,
                        help="MergeMertens contrast weight. Default: 1.0")
    parser.add_argument("--mef-saturation-weight", type=float, default=1.0,
                        help="MergeMertens saturation weight. Default: 1.0")
    parser.add_argument("--mef-exposure-weight", type=float, default=1.0,
                        help="MergeMertens well-exposedness weight. Default: 1.0")
    parser.add_argument("--mef-clahe-post", action="store_true",
                        help="Apply mild CLAHE on luminance after exposure fusion.")
    parser.add_argument("--mef-clahe-clip", type=float, default=1.0,
                        help="CLAHE clip limit for post-processing. Default: 1.0")
    parser.add_argument("--mef-clahe-grid", type=int, default=16,
                        help="CLAHE tile grid size for post-processing. Default: 16")
    parser.add_argument("--mef-blend-alpha", type=float, default=0.85,
                        help="Blend weight for fused result when mixing with the original image. Default: 0.85")

    # Low-light simulation
    parser.add_argument("--contrast-range", type=lambda s: parse_range(s, "contrast-range"), default=(0.15, 0.95),
                        help="Contrast factor range, formatted as min,max. Default: 0.15,0.95.")
    parser.add_argument("--gamma-range", type=lambda s: parse_range(s, "gamma-range"), default=(2.2, 6.0),
                        help="Gamma darkening range, formatted as min,max. Default: 2.2,6.0.")
    parser.add_argument("--poisson-level", type=float, default=5.0,
                        help="Poisson noise level used by VIIS-style implementation. Default: 5.")
    parser.add_argument("--gaussian-sigma", type=float, default=10.0,
                        help="Gaussian noise sigma. Default: 10.")
    parser.add_argument("--use-salt-pepper", action="store_true",
                        help="Add salt-and-pepper noise. Disabled by default.")
    parser.add_argument("--low-retry", type=int, default=3,
                        help="Retry times for low-light simulation if the result becomes too dark. Default: 3")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Default: None.")
    parser.add_argument("--save-ext", type=str, default="png", choices=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                        help="Output image extension. Default: png.")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.num_aug <= 0:
        raise ValueError("num_aug must be a positive integer.")
    if len(args.mef_ev_list) != len(args.mef_gamma_list):
        raise ValueError("--mef-ev-list and --mef-gamma-list must have the same length.")

    pairs = build_pairs(args.visible_path, args.thermal_path)
    print(f"Found {len(pairs)} visible-thermal pair(s).")

    gt_dir = args.out_dir / "gt"
    low_dir = args.out_dir / "low"
    gt_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for pair_idx, (vis_file, th_file) in enumerate(pairs):
        original_img = pil_to_rgb_np(vis_file)
        th_img = pil_to_thermal_np(th_file, thermal_mode=args.thermal_mode)

        # Step 0: 增强整张可见光原图
        gt_img = original_img
        if not args.disable_visible_reinforce:
            gt_img = reinforce_visible_with_mef(
                gt_img,
                ev_list=args.mef_ev_list,
                gamma_list=args.mef_gamma_list,
                contrast_weight=args.mef_contrast_weight,
                saturation_weight=args.mef_saturation_weight,
                exposure_weight=args.mef_exposure_weight,
                use_clahe_post=args.mef_clahe_post,
                clahe_clip_limit=args.mef_clahe_clip,
                clahe_tile_grid_size=args.mef_clahe_grid,
                blend_with_original=args.mef_blend_alpha,
            )

        h, w = gt_img.shape[:2]
        th_h, th_w = th_img.shape[:2]
        if (th_h, th_w) != (h, w):
            if args.resize_thermal:
                th_img = resize_np_like_thermal(th_img, (h, w))
            else:
                raise ValueError(
                    f"Size mismatch for pair:\n"
                    f"  visible={vis_file}, size={(h, w)}\n"
                    f"  thermal={th_file}, size={(th_h, th_w)}\n"
                    f"Please pre-align the pair or add --resize-thermal."
                )

        base = vis_file.stem
        
        # 不再进行切块和几何变换，直接对整图做多次随机暗光模拟
        for aug_idx in range(args.num_aug):
            low_aug = simulate_low_light_viis(
                gt_img,
                contrast_range=args.contrast_range,
                gamma_range=args.gamma_range,
                poisson_level=args.poisson_level,
                gaussian_sigma=args.gaussian_sigma,
                use_salt_pepper=args.use_salt_pepper,
                num_retry=args.low_retry,
            )

            if args.num_aug == 1:
                name = f"{base}.{args.save_ext}"
            else:
                name = f"{base}_a{aug_idx:02d}.{args.save_ext}"
    
            save_np_image(gt_dir / name, gt_img)
            save_np_image(low_dir / name, low_aug)
            total += 1

        if (pair_idx + 1) % 10 == 0 or pair_idx + 1 == len(pairs):
            print(f"Processed {pair_idx + 1}/{len(pairs)} pairs, saved {total} pairs.")

    print("Done.")
    print(f"Saved whole-image pairs: {total}")
    print(f"Output gt  : {gt_dir}")
    print(f"Output low : {low_dir}")

if __name__ == "__main__":
    main()