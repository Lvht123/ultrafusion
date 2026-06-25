from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LOW_LIGHT_ALIASES = ("low", "low_light", "input", "lq")
TARGET_ALIASES = ("normal", "target", "gt", "high", "enhanced")
THERMAL_ALIASES = ("thermal", "ir", "infrared", "tir")


def _scan_images(root_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(root_dir).with_suffix("").as_posix()
        files[key] = path
    return files


def _first_existing_dir(base_dir: Path, preferred: str | None, aliases: Iterable[str]) -> Path:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(alias for alias in aliases if alias not in candidates)

    for name in candidates:
        path = base_dir / name
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not find any of {candidates} under {base_dir}. "
        "Expected paired low-light/target/thermal folders."
    )


def _resolve_base_dir(data_root: str | Path, split: str | None) -> Path:
    root = Path(data_root)
    if split:
        split_dir = root / split
        if split_dir.is_dir():
            return split_dir
    return root


def _target_hw(img_size) -> tuple[int, int]:
    if img_size is None:
        raise ValueError("img_size must be set when resizing/cropping is enabled")
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise ValueError("img_size must be an int or a two-item [height, width] sequence")


def _resize_if_needed(image: Image.Image, min_h: int, min_w: int) -> Image.Image:
    width, height = image.size
    if height >= min_h and width >= min_w:
        return image
    scale = max(min_h / max(height, 1), min_w / max(width, 1))
    new_size = (max(int(round(width * scale)), min_w), max(int(round(height * scale)), min_h))
    return image.resize(new_size, Image.BICUBIC)


def _aligned_crop(
    images: tuple[Image.Image, ...],
    crop_h: int,
    crop_w: int,
    crop_mode: str,
) -> tuple[Image.Image, ...]:
    images = tuple(_resize_if_needed(image, crop_h, crop_w) for image in images)
    width, height = images[0].size

    if crop_mode == "resize":
        return tuple(image.resize((crop_w, crop_h), Image.BICUBIC) for image in images)

    if width < crop_w or height < crop_h:
        raise ValueError(f"Image is smaller than requested crop: {(height, width)} vs {(crop_h, crop_w)}")

    if crop_mode == "random":
        top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
    elif crop_mode == "center":
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
    else:
        raise ValueError("crop_mode must be one of: random, center, resize")

    box = (left, top, left + crop_w, top + crop_h)
    return tuple(image.crop(box) for image in images)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array / 127.5 - 1.0).permute(2, 0, 1).contiguous()


def _resize_exact(images: tuple[Image.Image, ...], height: int, width: int) -> tuple[Image.Image, ...]:
    return tuple(image.resize((width, height), Image.BICUBIC) for image in images)


def _as_pair(value, name: str) -> tuple[float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        value = [float(part) for part in parts]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item sequence")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return lo, hi


def _as_float_tuple(value, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(float(item) for item in value)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Online low-light simulation requires OpenCV. "
            "Please install opencv-python or opencv-python-headless."
        ) from exc
    return cv2


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _luma_mean_rgb(rgb_u8: np.ndarray) -> float:
    x = rgb_u8.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return float(y.mean())


def _safe_gain_rescale(
    rgb_u8: np.ndarray,
    target_mean: float,
    min_gain: float = 0.8,
    max_gain: float = 2.5,
) -> np.ndarray:
    cur_mean = _luma_mean_rgb(rgb_u8)
    if cur_mean < 1e-6:
        return rgb_u8.copy()
    gain = np.clip(target_mean / cur_mean, min_gain, max_gain)
    return np.clip(rgb_u8.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)


def _apply_ev_and_gamma(rgb_u8: np.ndarray, ev: float, gamma: float) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    x = _clamp01(x * (2.0 ** ev))
    x = np.power(np.clip(x, 1e-4, 1.0), gamma)
    return (_clamp01(x) * 255.0 + 0.5).astype(np.uint8)


def _build_synthetic_exposure_stack(
    rgb_u8: np.ndarray,
    ev_list: tuple[float, ...],
    gamma_list: tuple[float, ...],
) -> list[np.ndarray]:
    if len(ev_list) != len(gamma_list):
        raise ValueError("mef_ev_list and mef_gamma_list must have the same length")
    variants = [_apply_ev_and_gamma(rgb_u8, ev=ev, gamma=gamma) for ev, gamma in zip(ev_list, gamma_list)]
    variants.append(rgb_u8.copy())
    return variants


def _mild_clahe_on_luminance(
    rgb_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    cv2 = _require_cv2()
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _reinforce_visible_with_mef(
    rgb_u8: np.ndarray,
    ev_list: tuple[float, ...],
    gamma_list: tuple[float, ...],
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
    use_clahe_post: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    blend_with_original: float = 0.85,
) -> np.ndarray:
    cv2 = _require_cv2()
    original = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    orig_mean = _luma_mean_rgb(original)

    stack_rgb = _build_synthetic_exposure_stack(original, ev_list=ev_list, gamma_list=gamma_list)
    stack_bgr_u8 = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in stack_rgb]
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=float(contrast_weight),
        saturation_weight=float(saturation_weight),
        exposure_weight=float(exposure_weight),
    )

    fused_bgr = _clamp01(merge_mertens.process(stack_bgr_u8))
    fused_bgr_u8 = (fused_bgr * 255.0 + 0.5).astype(np.uint8)
    fused_rgb = cv2.cvtColor(fused_bgr_u8, cv2.COLOR_BGR2RGB)

    fused_mean = _luma_mean_rgb(fused_rgb)
    if fused_rgb.max() <= 5 or fused_mean < max(5.0, 0.15 * orig_mean):
        fused_rgb = original.copy()

    target_mean = max(orig_mean * 0.98, 45.0)
    fused_rgb = _safe_gain_rescale(fused_rgb, target_mean=target_mean, min_gain=0.9, max_gain=2.2)

    if use_clahe_post:
        fused_rgb = _mild_clahe_on_luminance(
            fused_rgb,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

    alpha = float(np.clip(blend_with_original, 0.0, 1.0))
    fused_rgb = np.clip(
        alpha * fused_rgb.astype(np.float32) + (1.0 - alpha) * original.astype(np.float32),
        0.0,
        255.0,
    ).astype(np.uint8)

    if fused_rgb.max() <= 5 or _luma_mean_rgb(fused_rgb) < 5.0:
        return original.copy()
    return fused_rgb


def _adjust_contrast_uint8(image: np.ndarray, factor: float) -> np.ndarray:
    img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(ImageEnhance.Contrast(img).enhance(factor), dtype=np.uint8)


def _gamma_darkening_uint8(image: np.ndarray, gamma: float) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def _poisson_noise_viis_uint8(image: np.ndarray, noise_level: float = 5.0) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = np.random.poisson(x * noise_level).astype(np.float32) / float(noise_level)
    return (np.clip(x + noise, 0.0, 1.0) * 255.0).astype(np.uint8)


def _gaussian_noise_uint8(image: np.ndarray, sigma: float = 10.0, mean: float = 0.0) -> np.ndarray:
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def _salt_pepper_noise_uint8(image: np.ndarray, salt_prob: float, pepper_prob: float) -> np.ndarray:
    noisy = image.copy()
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    return noisy.astype(np.uint8)


def _simulate_low_light_once(
    gt_patch: np.ndarray,
    contrast_range: tuple[float, float],
    gamma_range: tuple[float, float],
    poisson_level: float,
    gaussian_sigma: float,
    use_salt_pepper: bool,
) -> np.ndarray:
    low = _adjust_contrast_uint8(gt_patch, random.uniform(*contrast_range))
    low = _gamma_darkening_uint8(low, random.uniform(*gamma_range))
    low = _poisson_noise_viis_uint8(low, noise_level=poisson_level)
    low = _gaussian_noise_uint8(low, sigma=gaussian_sigma)

    if use_salt_pepper:
        low = _salt_pepper_noise_uint8(
            low,
            salt_prob=0.2 * random.random(),
            pepper_prob=0.2 * random.random(),
        )
    return low


def _simulate_low_light_viis(
    gt_patch: np.ndarray,
    contrast_range: tuple[float, float] = (0.15, 0.95),
    gamma_range: tuple[float, float] = (2.2, 6.0),
    poisson_level: float = 5.0,
    gaussian_sigma: float = 10.0,
    use_salt_pepper: bool = False,
    num_retry: int = 3,
) -> np.ndarray:
    gt_patch = np.clip(gt_patch, 0, 255).astype(np.uint8)
    best = None
    best_mean = -1.0

    for _ in range(max(1, int(num_retry))):
        low = _simulate_low_light_once(
            gt_patch,
            contrast_range=contrast_range,
            gamma_range=gamma_range,
            poisson_level=poisson_level,
            gaussian_sigma=gaussian_sigma,
            use_salt_pepper=use_salt_pepper,
        )
        cur_mean = _luma_mean_rgb(low)
        if cur_mean > best_mean:
            best = low
            best_mean = cur_mean
        if low.max() > 10 and cur_mean >= 4.0:
            return low

    low = _adjust_contrast_uint8(gt_patch, factor=max(0.35, contrast_range[0]))
    low = _gamma_darkening_uint8(low, gamma=min(3.5, gamma_range[1]))
    low = _poisson_noise_viis_uint8(low, noise_level=poisson_level)
    low = _gaussian_noise_uint8(low, sigma=gaussian_sigma)

    if low.max() <= 10 or _luma_mean_rgb(low) < 4.0:
        low = _safe_gain_rescale(best if best is not None else gt_patch, target_mean=12.0, min_gain=1.0, max_gain=4.0)
    return low.astype(np.uint8)


class OnlineLowLightRGBTDataset(Dataset):
    def __init__(
        self,
        data_root,
        img_size=640,
        split: str | None = None,
        target_dir: str | None = None,
        thermal_dir: str | None = None,
        random_flip: bool = True,
        disable_visible_reinforce: bool = False,
        mef_ev_list=(-1.0, -0.5, 0.0, 0.5, 1.0),
        mef_gamma_list=(1.10, 1.05, 1.00, 0.95, 0.90),
        mef_contrast_weight: float = 1.0,
        mef_saturation_weight: float = 1.0,
        mef_exposure_weight: float = 1.0,
        mef_clahe_post: bool = False,
        mef_clahe_clip: float = 2.0,
        mef_clahe_grid: int = 8,
        mef_blend_alpha: float = 0.85,
        contrast_range=(0.15, 0.95),
        gamma_range=(2.2, 6.0),
        poisson_level: float = 5.0,
        gaussian_sigma: float = 10.0,
        use_salt_pepper: bool = False,
        low_retry: int = 3,
    ):
        if data_root is None:
            raise ValueError("data_root is required")

        base_dir = _resolve_base_dir(data_root, split)
        target_root = _first_existing_dir(base_dir, target_dir, TARGET_ALIASES)
        thermal_root = _first_existing_dir(base_dir, thermal_dir, THERMAL_ALIASES)
        target_files = _scan_images(target_root)
        thermal_files = _scan_images(thermal_root)
        keys = sorted(set(target_files) & set(thermal_files))
        if not keys:
            raise FileNotFoundError(f"No paired target/thermal images found under {target_root} and {thermal_root}")

        self.pairs = [(target_files[key], thermal_files[key]) for key in keys]
        self.resize_h, self.resize_w = _target_hw(img_size)
        self.random_flip = bool(random_flip)
        self.reinforce_visible = not bool(disable_visible_reinforce)
        self.mef_ev_list = _as_float_tuple(mef_ev_list, "mef_ev_list")
        self.mef_gamma_list = _as_float_tuple(mef_gamma_list, "mef_gamma_list")
        if len(self.mef_ev_list) != len(self.mef_gamma_list):
            raise ValueError("mef_ev_list and mef_gamma_list must have the same length")
        self.mef_contrast_weight = float(mef_contrast_weight)
        self.mef_saturation_weight = float(mef_saturation_weight)
        self.mef_exposure_weight = float(mef_exposure_weight)
        self.mef_clahe_post = bool(mef_clahe_post)
        self.mef_clahe_clip = float(mef_clahe_clip)
        self.mef_clahe_grid = int(mef_clahe_grid)
        self.mef_blend_alpha = float(mef_blend_alpha)
        self.contrast_range = _as_pair(contrast_range, "contrast_range")
        self.gamma_range = _as_pair(gamma_range, "gamma_range")
        self.poisson_level = float(poisson_level)
        self.gaussian_sigma = float(gaussian_sigma)
        self.use_salt_pepper = bool(use_salt_pepper)
        self.low_retry = int(low_retry)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target_path, thermal_path = self.pairs[idx]
        target = Image.open(target_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("RGB")
        target, thermal = _resize_exact((target, thermal), self.resize_h, self.resize_w)

        if self.random_flip:
            if torch.rand(()) < 0.5:
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
                thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(()) < 0.5:
                target = target.transpose(Image.FLIP_TOP_BOTTOM)
                thermal = thermal.transpose(Image.FLIP_TOP_BOTTOM)

        target_array = np.asarray(target, dtype=np.uint8)
        if self.reinforce_visible:
            target_array = _reinforce_visible_with_mef(
                target_array,
                ev_list=self.mef_ev_list,
                gamma_list=self.mef_gamma_list,
                contrast_weight=self.mef_contrast_weight,
                saturation_weight=self.mef_saturation_weight,
                exposure_weight=self.mef_exposure_weight,
                use_clahe_post=self.mef_clahe_post,
                clahe_clip_limit=self.mef_clahe_clip,
                clahe_tile_grid_size=self.mef_clahe_grid,
                blend_with_original=self.mef_blend_alpha,
            )
        low_light_array = _simulate_low_light_viis(
            target_array,
            contrast_range=self.contrast_range,
            gamma_range=self.gamma_range,
            poisson_level=self.poisson_level,
            gaussian_sigma=self.gaussian_sigma,
            use_salt_pepper=self.use_salt_pepper,
            num_retry=self.low_retry,
        )

        target = Image.fromarray(target_array, mode="RGB")
        low_light = Image.fromarray(low_light_array, mode="RGB")
        return _to_tensor(target), _to_tensor(low_light), _to_tensor(thermal)


class PairedLowLightRGBTDataset(Dataset):
    def __init__(
        self,
        data_root,
        split: str | None = None,
        low_light_dir: str | None = None,
        target_dir: str | None = None,
        thermal_dir: str | None = None,
        strict_pair_size: bool = False,
    ):
        if data_root is None:
            raise ValueError("data_root is required")

        base_dir = _resolve_base_dir(data_root, split)
        low_light_root = _first_existing_dir(base_dir, low_light_dir, LOW_LIGHT_ALIASES)
        target_root = _first_existing_dir(base_dir, target_dir, TARGET_ALIASES)
        thermal_root = _first_existing_dir(base_dir, thermal_dir, THERMAL_ALIASES)
        low_files = _scan_images(low_light_root)
        target_files = _scan_images(target_root)
        thermal_files = _scan_images(thermal_root)
        keys = sorted(set(low_files) & set(target_files) & set(thermal_files))
        if not keys:
            raise FileNotFoundError(
                "No paired low-light/target/thermal images found under "
                f"{low_light_root}, {target_root}, and {thermal_root}"
            )

        self.pairs = [(target_files[key], low_files[key], thermal_files[key]) for key in keys]
        self.strict_pair_size = bool(strict_pair_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target_path, low_light_path, thermal_path = self.pairs[idx]
        target = Image.open(target_path).convert("RGB")
        low_light = Image.open(low_light_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("RGB")

        if self.strict_pair_size and (target.size != low_light.size or target.size != thermal.size):
            raise ValueError(
                f"Pair sizes differ for {low_light_path}, {thermal_path}, and {target_path}: "
                f"{low_light.size}, {thermal.size}, {target.size}"
            )

        return _to_tensor(target), _to_tensor(low_light), _to_tensor(thermal)


LowLightPairedDataset = PairedLowLightRGBTDataset


def loader(
    train_batch_size,
    num_workers,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_type: str = "paired",
    **args,
):
    dataset_type = dataset_type.lower().replace("-", "_")
    if dataset_type in {"online", "online_simulation", "online_lowlight", "train"}:
        dataset = OnlineLowLightRGBTDataset(**args)
    elif dataset_type in {"paired", "triplet", "eval", "val", "test"}:
        dataset = PairedLowLightRGBTDataset(**args)
    else:
        raise ValueError(
            "dataset_type must be one of: online_simulation, paired, triplet, train, val, test"
        )
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
