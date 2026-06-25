import os, random, glob, time
import tqdm
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, CenterCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset.lowlight_pair_dataset_v2 import _simulate_low_light_viis, _reinforce_visible_with_mef


def get_color_and_struct(isrgb, input_img: torch.Tensor, ksize, sigmaX, c):  #input an RGB image

    input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)

    if isrgb==True:
        yuv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV).astype(np.float32)
        y = np.expand_dims(yuv_img[:,:,0], axis=-1).astype(np.float64)
        u = np.expand_dims(yuv_img[:,:,1], axis=-1).astype(np.float32)
        v = np.expand_dims(yuv_img[:,:,2], axis=-1).astype(np.float32)
    else:
        y = input_img.astype(np.float64)
    #mu = gaussian_filter(y, ksize, ksize/6)
    mu = cv2.GaussianBlur(y, (ksize,ksize), sigmaX).astype(np.float64)
    mu_sq = mu * mu
    sigma = np.sqrt(np.absolute(cv2.GaussianBlur(y*y, (ksize,ksize), sigmaX) - mu_sq)).astype(np.float64)
    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    dividend = y.astype(np.float64) - mu
    divisor = sigma + c
    struct = dividend / divisor
    struct = struct.astype(np.float32)
    struct_norm = (struct - struct.min()) / (struct.max() - struct.min() + 1e-6)
    struct_norm = torch.from_numpy(struct_norm).permute(2, 0, 1)
    u = torch.from_numpy(u).permute(2, 0, 1)
    v = torch.from_numpy(v).permute(2, 0, 1)
    img_uv = torch.cat([u, v], dim=0)
    return struct_norm, img_uv


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


class MEFDataset(data.Dataset):
    def __init__(self, img_dir, swir_img_dir,
                 random_crop=True, random_resize=True, rotate=True, flip=True,
                 disable_visible_reinforce=True,
                 mef_ev_list=(-1.0, -0.5, 0.0, 0.5, 1.0),
                 mef_gamma_list=(1.10, 1.05, 1.00, 0.95, 0.90),
                 mef_contrast_weight=1.0,
                 mef_saturation_weight=1.0,
                 mef_exposure_weight=1.0,
                 mef_clahe_post=False,
                 mef_clahe_clip=2.0,
                 mef_clahe_grid=8,
                 mef_blend_alpha=0.85,
                 contrast_range=(0.15, 0.95),
                 gamma_range=(2.2, 6.0),
                 poisson_level=5.0,
                 gaussian_sigma=10.0,
                 use_salt_pepper=False,
                 low_retry=3):
        super(MEFDataset, self).__init__()
        self.img_dir = img_dir
        self.swir = swir_img_dir
        self.random_crop = random_crop
        self.random_resize = random_resize
        self.rotate = rotate
        self.flip = flip
        self.to_tensor = ToTensor()
        self.swir_list = []
        self.gt_list = []
        lq1_dir = os.path.join(self.img_dir, 'swir')
        gt_dir = os.path.join(self.img_dir, 'rgb')

        self.swir_list = sorted([os.path.join(lq1_dir, f) for f in os.listdir(lq1_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        # Low-light simulation parameters
        self.reinforce_visible = not bool(disable_visible_reinforce)
        self.mef_ev_list = tuple(float(v) for v in mef_ev_list)
        self.mef_gamma_list = tuple(float(v) for v in mef_gamma_list)
        self.mef_contrast_weight = float(mef_contrast_weight)
        self.mef_saturation_weight = float(mef_saturation_weight)
        self.mef_exposure_weight = float(mef_exposure_weight)
        self.mef_clahe_post = bool(mef_clahe_post)
        self.mef_clahe_clip = float(mef_clahe_clip)
        self.mef_clahe_grid = int(mef_clahe_grid)
        self.mef_blend_alpha = float(mef_blend_alpha)
        self.contrast_range = (float(contrast_range[0]), float(contrast_range[1]))
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.poisson_level = float(poisson_level)
        self.gaussian_sigma = float(gaussian_sigma)
        self.use_salt_pepper = bool(use_salt_pepper)
        self.low_retry = int(low_retry)

    def __getitem__(self, index):
        gt_path = self.gt_list[index % len(self.gt_list)]
        swir_path = self.swir_list[index % len(self.gt_list)]
        swir = Image.open(swir_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if 'align' in gt_path:
            W, H = gt.size
            cc = CenterCrop([H - 100, W - 100])
            swir = cc(swir)
            gt = cc(gt)
        if self.random_resize:
            W, H = gt.size
            min_size = 512
            max_size = min(H, W)
            tgt_size = torch.randint(min_size, max_size + 1, (1, )).item()
            swir = Resize(tgt_size)(swir)
            gt = Resize(tgt_size)(gt)
        if self.random_crop:
            crop_params = RandomCrop.get_params(gt, [512, 512])
            swir = crop(swir, *crop_params)
            gt = crop(gt, *crop_params)
        else:
            swir = CenterCrop(512)(swir)
            gt = CenterCrop(512)(gt)
        if self.rotate:
            rotate_params = random.randint(0, 3) * 90
            swir = rotate(swir, rotate_params)
            gt = rotate(gt, rotate_params)
        if self.flip:
            if torch.rand(1) > 0.5:
                swir = RandomHorizontalFlip(1)(swir)
                gt = RandomHorizontalFlip(1)(gt)
            if torch.rand(1) > 0.5:
                swir = RandomVerticalFlip(1)(swir)
                gt = RandomVerticalFlip(1)(gt)

        # Online low-light simulation from GT
        gt_np = np.asarray(gt)
        if self.reinforce_visible:
            gt_np = _reinforce_visible_with_mef(
                gt_np,
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
        low_light_np = _simulate_low_light_viis(
            gt_np,
            contrast_range=self.contrast_range,
            gamma_range=self.gamma_range,
            poisson_level=self.poisson_level,
            gaussian_sigma=self.gaussian_sigma,
            use_salt_pepper=self.use_salt_pepper,
            num_retry=self.low_retry,
        )
        lq2 = Image.fromarray(low_light_np, mode='RGB')
        gt = Image.fromarray(gt_np, mode='RGB')

        swir = self.to_tensor(swir)
        lq2 = self.to_tensor(lq2)
        gt = self.to_tensor(gt)

        lq1_struct, lq1_color = get_color_and_struct(isrgb=True, input_img=swir, ksize=7, sigmaX=0, c=0.0000001)

        # Normalize to [-1, 1]
        gt = gt * 2 - 1
        lq2 = lq2 * 2 - 1

        return {
            'gt': gt,
            'lq1_struct': lq1_struct,
            'lq1_color': lq1_color,
            'lq2': lq2,
            'prompt': '',
        }

    def __len__(self):
        return len(self.gt_list) * 10
        # return 1