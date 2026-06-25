import os, sys, csv
from argparse import ArgumentParser

from omegaconf import OmegaConf
import torch
import pyiqa
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model.V4_CA.cldm import ControlLDM
from model.V4_CA.gaussian_diffusion import Diffusion
from utils.common import instantiate_from_config
from utils.V4_CA.sampler import SpacedSampler


def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def run_evaluation(eval_cfg, ckpt_path, global_step, exp_dir, pure_cldm, diffusion, sampler, device):
    """
    In-process evaluation reusing training-loaded VAE/UNet/CLIP/Diffusion.
    Only swaps controlnet weights — no extra GPU memory needed.
    """
    from dataset.test_dataset import LowLightTestDataset, TestDataset, get_color_and_struct

    # ---- pad helper for MUSIQ ----
    def pad_to_min_size(tensor, min_size=224):
        _, _, h, w = tensor.shape
        if h < min_size or w < min_size:
            pad_h = max(0, min_size - h)
            pad_w = max(0, min_size - w)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor

    # ---- 1. Save current training controlnet weights ----
    train_controlnet_sd = {k: v.cpu().clone() for k, v in pure_cldm.controlnet.state_dict().items()}

    # ---- 2. Load checkpoint controlnet weights ----
    ckpt_sd = torch.load(ckpt_path, map_location=device)
    pure_cldm.controlnet.load_state_dict(ckpt_sd, strict=True)
    pure_cldm.eval()

    # ---- 3. Setup metrics once ----
    psnr_metric = pyiqa.create_metric("psnr", device=device)
    ssim_metric = pyiqa.create_metric("ssim", device=device)
    niqe_metric = pyiqa.create_metric("niqe", device=device)
    musiq_metric = pyiqa.create_metric("musiq", device=device)
    clipiqa_metric = pyiqa.create_metric("clipiqa+", device=device)
    brisque_metric = pyiqa.create_metric("brisque", device=device)

    eval_output_root = os.path.join(exp_dir, "eval", f"{global_step:07d}")

    for dataset_name, ds_opts in eval_cfg.datasets.items():
        eval_output_dir = os.path.join(eval_output_root, dataset_name)
        os.makedirs(eval_output_dir, exist_ok=True)

        print(f"[Eval] step={global_step:07d}  dataset={dataset_name}  (in-process)")

        # ---- 4. Build dataset ----
        if dataset_name == "LowLight":
            dataset = LowLightTestDataset(ds_opts["data_dir"])
        else:
            dataset = TestDataset(dataset_name)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        psnr_list, ssim_list, niqe_list, musiq_list, clipiqa_list, brisque_list = [], [], [], [], [], []
        is_lowlight = (dataset_name == "LowLight")

        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}", disable=False):
            img_name = batch["file_name"][0]

            if is_lowlight:
                swir = batch["swir"].to(device)
                lowlight = batch["lowlight"].to(device)
                label = batch["label"].to(device)

                # Non-tiled: resize to 512x512
                _, _, H_orig, W_orig = lowlight.shape
                swir_512 = F.interpolate(swir, size=(512, 512), mode="bilinear", align_corners=False)
                lowlight_512 = F.interpolate(lowlight, size=(512, 512), mode="bilinear", align_corners=False)

                swir_struct, swir_color = get_color_and_struct(isrgb=True, input_img=swir_512, ksize=7, sigmaX=0, c=0.0000001)
                swir_struct = swir_struct.unsqueeze(0).to(device)
                swir_color = swir_color.unsqueeze(0).to(device)

                cond = pure_cldm.prepare_condition(
                    lq2=lowlight_512 * 2 - 1,   # [0,1] -> [-1,1]
                    lq1_struct=swir_struct,
                    lq1_color=swir_color,
                    txt=""
                )

                with torch.no_grad():
                    z = sampler.sample(
                        model=pure_cldm, device=device, steps=50, batch_size=1,
                        x_size=(4, 64, 64), cond=cond, uncond=None,
                        cfg_scale=1.0, x_T=None, progress=False, progress_leave=False
                    )
                    out = pure_cldm.vae_decode(z)  # [-1, 1]

                out = F.interpolate(out, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
                out_vis = (out + 1) / 2  # [-1,1] -> [0,1]
                out_vis = out_vis.clamp(0.0, 1.0)

                save_image(out_vis, os.path.join(eval_output_dir, f"{img_name}_out.png"))

                # ---- Full-reference ----
                psnr_list.append(psnr_metric(out_vis, label).item())
                ssim_list.append(ssim_metric(out_vis, label).item())

                # ---- No-reference ----
                out_padded = pad_to_min_size(out_vis)
                niqe_list.append(niqe_metric(out_padded).item())
                musiq_list.append(musiq_metric(out_padded).item())
                clipiqa_list.append(clipiqa_metric(out_padded).item())
                brisque_list.append(brisque_metric(out_padded).item())

            else:
                # MEF datasets: ue/oe inputs, no GT
                ue = batch["ue"].to(device)
                oe = batch["oe"].to(device)
                # ... skip for brevity; MEF eval is rare during training
                continue

        # ---- 5. Save per-checkpoint CSV ----
        if niqe_list:
            avg_niqe = float(np.mean(niqe_list))
            avg_musiq = float(np.mean(musiq_list))
            avg_clipiqa = float(np.mean(clipiqa_list))
            avg_brisque = float(np.mean(brisque_list))

            csv_path = os.path.join(eval_output_dir, "metrics_result.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if is_lowlight:
                    avg_psnr = float(np.mean(psnr_list))
                    avg_ssim = float(np.mean(ssim_list))
                    writer.writerow(["image", "psnr", "ssim", "niqe", "musiq", "clipiqa+", "brisque"])
                    for i, name in enumerate(dataset.file_name_list):
                        writer.writerow([name, round(psnr_list[i], 4), round(ssim_list[i], 4),
                                         round(niqe_list[i], 4), round(musiq_list[i], 4),
                                         round(clipiqa_list[i], 4), round(brisque_list[i], 4)])
                    writer.writerow(["AVERAGE", round(avg_psnr, 4), round(avg_ssim, 4),
                                     round(avg_niqe, 4), round(avg_musiq, 4),
                                     round(avg_clipiqa, 4), round(avg_brisque, 4)])
                    print(f"[Eval] OK  PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}  "
                          f"NIQE={avg_niqe:.4f}  BRISQUE={avg_brisque:.4f}")

                    # ---- 5b. Append to global summary CSV for cross-checkpoint comparison ----
                    summary_path = os.path.join(exp_dir, "eval", "eval_summary.csv")
                    write_header = not os.path.exists(summary_path)
                    with open(summary_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["step", "dataset", "psnr", "ssim", "niqe", "musiq", "clipiqa+", "brisque"])
                        writer.writerow([global_step, dataset_name,
                                         round(avg_psnr, 4), round(avg_ssim, 4),
                                         round(avg_niqe, 4), round(avg_musiq, 4),
                                         round(avg_clipiqa, 4), round(avg_brisque, 4)])
                    print(f"[Eval]   appended to {summary_path}")
                else:
                    writer.writerow(["image", "niqe", "musiq", "clipiqa+", "brisque"])
                    for i, name in enumerate(dataset.file_name_list):
                        writer.writerow([name, round(niqe_list[i], 4), round(musiq_list[i], 4),
                                         round(clipiqa_list[i], 4), round(brisque_list[i], 4)])
                    writer.writerow(["AVERAGE", round(avg_niqe, 4), round(avg_musiq, 4),
                                     round(avg_clipiqa, 4), round(avg_brisque, 4)])
                    print(f"[Eval] OK  NIQE={avg_niqe:.4f}  MUSIQ={avg_musiq:.4f}  "
                          f"CLIPIQA+={avg_clipiqa:.4f}  BRISQUE={avg_brisque:.4f}")

                    summary_path = os.path.join(exp_dir, "eval", "eval_summary.csv")
                    write_header = not os.path.exists(summary_path)
                    with open(summary_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["step", "dataset", "niqe", "musiq", "clipiqa+", "brisque"])
                        writer.writerow([global_step, dataset_name,
                                         round(avg_niqe, 4), round(avg_musiq, 4),
                                         round(avg_clipiqa, 4), round(avg_brisque, 4)])
                    print(f"[Eval]   appended to {summary_path}")
            print(f"[Eval]   saved to {csv_path}")

    # ---- 6. Restore training controlnet weights ----
    pure_cldm.controlnet.load_state_dict(train_controlnet_sd, strict=True)
    del train_controlnet_sd
    pure_cldm.train()
    torch.cuda.empty_cache()


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=False)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=False)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=False)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused = cldm.load_pretrained_sd(sd)
    if accelerator.is_local_main_process:
        print(f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
              f"unused weights: {unused}")
    
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from pretrained SD\n"
                  f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                  f"weights initialized from scratch: {init_with_scratch}")
    
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    
    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)
    
    # Setup data:
    dataset1 = instantiate_from_config(cfg.dataset.train1)
    if accelerator.is_local_main_process:
        print(f"Dataset1 contains {len(dataset1):,} images from {dataset1.img_dir}")
    # dataset2 = instantiate_from_config(cfg.dataset.train2)
    # if accelerator.is_local_main_process:
    #     print(f"Dataset2 contains {len(dataset2):,} images from {dataset2.img_dir}")
    # dataset = ConcatDataset([dataset1, dataset2])
    dataset = dataset1
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for sample in loader:
            gt = sample['gt'].to(device) # [-1, 1]
            # lq1 = sample['lq1'].to(device) # [-1, 1]
            lq2 = sample['lq2'].to(device) # [-1, 1]
            lq1_struct = sample['lq1_struct'].to(device) # [0, 1]
            lq1_color = sample['lq1_color'].to(device) # [0, 1]
            prompt = sample['prompt'] # ""
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                # clean = swinir(lq)
                cond = pure_cldm.prepare_condition(lq2=lq2, lq1_struct=lq1_struct, lq1_color=lq1_color, txt=prompt)
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
            
            loss = diffusion.p_losses(cldm, z_0, t, cond)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = os.path.join(ckpt_dir, f"{global_step:07d}.pt")
                    torch.save(checkpoint, ckpt_path)

                    # Automatic evaluation on saved checkpoint
                    eval_cfg = cfg.get("eval", {})
                    if eval_cfg.get("enabled", False) and eval_cfg.get("datasets"):
                        torch.cuda.empty_cache()
                        run_evaluation(eval_cfg, ckpt_path, global_step, exp_dir,
                                       pure_cldm, diffusion, sampler, device)
                        torch.cuda.empty_cache()

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 4
                log_cond = {k:v[:N] for k, v in cond.items()}
                log_gt, log_lq2, log_lq1_struct = gt[:N], lq2[:N], lq1_struct[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=len(log_gt), x_size=z_0.shape[1:],
                        cond=log_cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            # ("image/lq1", (log_lq1 + 1) / 2),
                            ("image/lq2", (log_lq2 + 1) / 2),
                            ("image/lq1_struct", log_lq1_struct),
                            ("image/condition_lq2_decoded", (pure_cldm.vae_decode(log_cond["c_lq2"]) + 1) / 2),
                            ("image/prompt", (log_txt_as_img((512, 512), log_prompt) + 1) / 2)
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)