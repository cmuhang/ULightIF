import os
import time
from pathlib import Path

import torch
from thop import profile
from tqdm import tqdm

from models.base_model import PLIEModel, PLIFModel, PLIDModel
from options import get_opts
from utils.image_utils import ImreadRGB, ImgRGB_save, ImgGray_save, is_image_file, rgb2ycrcb_torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_models(opts):
    for key in ["stage1_ckpt", "stage2_ckpt", "stage3_ckpt"]:
        ckpt = getattr(opts, key)
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    ie_full = PLIEModel.load_from_checkpoint(opts.stage1_ckpt, opts=opts, weights_only=False).eval()
    if_full = PLIFModel.load_from_checkpoint(
        opts.stage2_ckpt,
        ie_ckpt_path=opts.stage1_ckpt,
        opts=opts,
        weights_only=False,
    ).eval()
    id_full = PLIDModel.load_from_checkpoint(
        opts.stage3_ckpt,
        ie_ckpt_path=opts.stage1_ckpt,
        if_ckpt_path=opts.stage2_ckpt,
        opts=opts,
        weights_only=False,
    ).eval()
    return ie_full, if_full, id_full


def slim_models(ie_full, if_full, id_full, device):
    ie_net = ie_full.net.slim().to(device).eval()
    if_net = if_full.net.slim().to(device).eval()
    id_net = id_full.net.slim().to(device).eval()
    return ie_net, if_net, id_net


@torch.no_grad()
def check_fold(ie_full, if_full, id_full, ie_slim, if_slim, id_slim, h=256, w=256):
    device = next(ie_slim.parameters()).device
    ie_input = torch.rand(1, 3, h, w, device=device)
    if_vi = torch.rand(1, 3, h, w, device=device)
    if_vi_en = torch.rand(1, 3, h, w, device=device)
    if_ir = torch.rand(1, 3, h, w, device=device)
    id_input = torch.rand(1, 3, h, w, device=device)

    ie_full_out = ie_full(ie_input)
    ie_slim_out = ie_slim(ie_input)

    if_vi_y, _, _ = rgb2ycrcb_torch(if_vi)
    if_vi_en_y, _, _ = rgb2ycrcb_torch(if_vi_en)
    if_ir_y, _, _ = rgb2ycrcb_torch(if_ir)
    if_fu_full, if_fu_en_full = if_full.net(if_vi_y, if_vi_en_y, if_ir_y)
    if_fu_slim, if_fu_en_slim = if_slim(if_vi, if_vi_en, if_ir)
    if_fu_slim_y, _, _ = rgb2ycrcb_torch(if_fu_slim)
    if_fu_en_slim_y, _, _ = rgb2ycrcb_torch(if_fu_en_slim)

    id_r_full, id_l_full = id_full.net(id_input)
    id_r_slim, id_l_slim = id_slim(id_input)

    print(f"[FoldCheck] simulated input: {h} x {w}")
    for name, a, b in [
        ("IE", ie_full_out, ie_slim_out),
        ("IF_fu", if_fu_full, if_fu_slim_y),
        ("IF_fuEn", if_fu_en_full, if_fu_en_slim_y),
        ("ID_R", id_r_full, id_r_slim),
        ("ID_L", id_l_full, id_l_slim),
    ]:
        diff = (a - b).abs()
        print(f"[FoldCheck] {name}: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")



@torch.no_grad()
def count_flops(ie_net, if_net, id_net, h=640, w=480):
    device = next(ie_net.parameters()).device
    vi = torch.randn(1, 3, h, w, device=device)
    ir = torch.randn(1, 3, h, w, device=device)

    vi_en = ie_net(vi)
    fu, _ = if_net(vi, vi_en, ir)

    ie_macs, _ = profile(ie_net, inputs=(vi,), verbose=False)
    if_macs, _ = profile(if_net, inputs=(vi, vi_en, ir), verbose=False)
    id_macs, _ = profile(id_net, inputs=(fu,), verbose=False)

    ie_flops = ie_macs / 1e9
    if_flops = if_macs / 1e9
    id_flops = id_macs / 1e9
    total_flops = ie_flops + if_flops + id_flops
    print(
        f"[FLOPs] Input: {h} x {w} | "
        f"IE: {ie_flops:.4f} GFLOPs | "
        f"IF: {if_flops:.4f} GFLOPs | "
        f"ID: {id_flops:.4f} GFLOPs | "
        f"Total: {total_flops:.4f} GFLOPs"
    )

def infer_and_save(ie_net, if_net, id_net, root_dir, output_dir, enable_save=True, enable_timing=True):
    vi_path = os.path.join(root_dir, "vi")
    ir_path = os.path.join(root_dir, "ir")
    img_list = sorted([x for x in os.listdir(vi_path) if is_image_file(x)])
    if not img_list:
        print("No images found in test set.")
        return

    save_dirs = {
        "viEn": os.path.join(output_dir, "ULightIF_viEn"),
        "fu": os.path.join(output_dir, "ULightIF_fu"),
        "fuEn": os.path.join(output_dir, "ULightIF_fuEn"),
        "fuR": os.path.join(output_dir, "ULightIF_fuR"),
        "fuL": os.path.join(output_dir, "ULightIF_fuL"),
    }
    if enable_save:
        for path in save_dirs.values():
            ensure_dir(path)

    device = next(ie_net.parameters()).device
    use_cuda = device.type == "cuda"
    total_time = 0.0

    with torch.inference_mode():
        for img_name in tqdm(img_list, desc="Processing images", unit="image"):
            vi = ImreadRGB(os.path.join(vi_path, img_name)).to(device)
            ir = ImreadRGB(os.path.join(ir_path, img_name)).to(device)

            if enable_timing and use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()

            vi_en = ie_net(vi)
            fu, fu_en = if_net(vi, vi_en, ir)
            fu_r, fu_l = id_net(fu)

            if enable_timing and use_cuda:
                torch.cuda.synchronize()
            if enable_timing:
                total_time += time.perf_counter() - start

            if enable_save:
                ImgRGB_save(vi_en, os.path.join(save_dirs["viEn"], img_name))
                ImgRGB_save(fu, os.path.join(save_dirs["fu"], img_name))
                ImgRGB_save(fu_en, os.path.join(save_dirs["fuEn"], img_name))
                ImgRGB_save(fu_r, os.path.join(save_dirs["fuR"], img_name))
                ImgGray_save(fu_l, os.path.join(save_dirs["fuL"], img_name))

    if enable_timing:
        avg_time = total_time / len(img_list)
        print(f"[Timing] Images: {len(img_list)} | Avg: {avg_time * 1000.0:.3f} ms/img | FPS: {1.0 / avg_time:.3f}")


def main(opts):
    root_dir = opts.test_root
    output_dir = ensure_dir(opts.output_dir)

    ie_full, if_full, id_full = load_models(opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ie_full = ie_full.to(device).eval()
    if_full = if_full.to(device).eval()
    id_full = id_full.to(device).eval()
    ie_net, if_net, id_net = slim_models(ie_full, if_full, id_full, device)

    print(
        f"[Params] IE: {count_params(ie_net) / 1e6:.4f} M | "
        f"IF: {count_params(if_net) / 1e6:.4f} M | "
        f"ID: {count_params(id_net) / 1e6:.4f} M | "
        f"Total: {(count_params(ie_net) + count_params(if_net) + count_params(id_net)) / 1e6:.4f} M"
    )

    if opts.check_fold:
        check_fold(ie_full, if_full, id_full, ie_net, if_net, id_net)
    if opts.calc_flops:
        count_flops(ie_net, if_net, id_net, h=640, w=480)

    infer_and_save(
        ie_net,
        if_net,
        id_net,
        root_dir=root_dir,
        output_dir=output_dir,
        enable_save=True,
        enable_timing=True,
    )

"""
常规融合建议采用 `fu` 进行对比，参数量与 FLOPs 仅统计 `if_model`；
低光融合建议采用 `fuR` 进行对比，并统计 `if_model + id_model`。
其他合理的对比方案亦可采用。

For conventional image fusion, `fu` is recommended for comparison, with parameters and FLOPs calculated only for `if_model`. 
For low-light image fusion, `fuR` is recommended, with parameters and FLOPs calculated for `if_model + id_model`. 
Other reasonable comparison schemes are also acceptable.
"""
if __name__ == "__main__":
    opts = get_opts()
    main(opts)
