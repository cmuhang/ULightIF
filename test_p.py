import os
import time
import torch
import torch.nn as nn
from tqdm import  tqdm
from thop import profile
from utils.image_utils import ImreadRGB,ImgRGB_save,ImgGray_save,rgb2ycrcb_torch,ycrcb2rgb_torch,is_image_file

class MobileIENetS(nn.Module):
    def __init__(self, channels=24):    
        super().__init__()
        self.channels = channels
        self.headA = nn.Sequential(nn.Conv2d(3, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)
        self.body1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)
        self.proj = nn.Conv2d(channels, 1, 3, 1, 1, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward_ill(self,inp : torch.Tensor) -> torch.Tensor:
        x0 = self.headA(inp) 
        x1 = self.body1(x0) 
        out = self.sigmoid(self.proj(x1+x0))
        return out

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        Ill = self.forward_ill(img)
        Ref = torch.clamp(img / Ill, 1e-4, 1.)
        return Ref
    
class MobileEncoderS(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, padding_mode='reflect'),
            nn.PReLU(channels),
        )
        self.BlockA = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels))
        self.BlockB = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels))
        self.BlockC = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels))

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.BlockA(x0) + x0
        x2 = self.BlockB(x1) + x1
        x3 = self.BlockC(x2) + x2
        return x3

class MobileDecoderS(nn.Module):
    def __init__(self,channels = 32):
        super().__init__()
        self.BlockA = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels))
        self.BlockB = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels))
        self.BlockC = nn.Sequential(nn.Conv2d(channels, 1, 3, 1, 1, padding_mode='reflect'),nn.Sigmoid())
        
    def forward(self, f1,f2):
        x0 = torch.cat([f1,f2],dim=1)
        x1 = self.BlockA(x0)
        x2 = self.BlockB(x1)
        x3 = self.BlockC(x2)
        return x3

class MobileFNetS(nn.Module):
    def __init__(self, channels = 16):
        super().__init__()
        self.enc = MobileEncoderS(channels=channels)
        self.dec = MobileDecoderS(channels=channels)

    def forward_fu(self, vi: torch.Tensor, viEn : torch.Tensor, ir: torch.Tensor ): #
        vi_feat = self.enc(vi)
        viEn_feat = self.enc(viEn)
        ir_feat = self.enc(ir)
        fu = self.dec(vi_feat, ir_feat)
        fuEn = self.dec(viEn_feat, ir_feat)
        return fu, fuEn

    def forward(self,vi_img,viEn_img,ir_img):
        vi_img,vi_cr,vi_cb = rgb2ycrcb_torch(vi_img)
        viEn_img,viEn_cr,viEn_cb = rgb2ycrcb_torch(viEn_img)
        ir_img,_,_ = rgb2ycrcb_torch(ir_img)
        fu,fuEn = self.forward_fu(vi_img,viEn_img,ir_img)
        fu = ycrcb2rgb_torch(torch.cat([fu,vi_cr,vi_cb],dim=1))
        fuEn = ycrcb2rgb_torch(torch.cat([fuEn,viEn_cr,viEn_cb],dim=1)) 
        return fu,fuEn

class MobileIDNetS(nn.Module):
    def __init__(self, channels = 32):
        super().__init__()
        self.channels = channels
        self.unit1 = nn.Sequential(nn.Conv2d(3, channels, 3, 1,1, padding_mode='reflect'),nn.PReLU(channels),)
        self.unit2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)
        self.unit3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)
        self.unit4 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)  
        self.unit5 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),nn.PReLU(channels),)  
        self.proj = nn.Sequential(nn.Conv2d(channels, 4, 3, 1, 1, padding_mode='reflect'),nn.Sigmoid())

    def forward(self,inp : torch.Tensor) -> torch.Tensor:
        x0 = self.unit1(inp) 
        x1 = self.unit2(x0)
        x2 = self.unit3(x1)
        x4 = self.unit4(x2 + x1)
        x5 = self.unit5(x4 + x0)  
        out = self.proj(x5)
        return out[:,:3,:,:],out[:,3:,:,:]





# -------------------------------------------------------------------------------------------------------------------------------
def count_flops(ie_model, if_model, id_model, h=256, w=256):
    device = next(ie_model.parameters()).device
    vi = torch.randn(1, 3, h, w, device=device)
    ir = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        vi_en = ie_model(vi)
        fu, _ = if_model(vi, vi_en, ir)

    ie_macs, _ = profile(ie_model, inputs=(vi,), verbose=False)
    if_macs, _ = profile(if_model, inputs=(vi, vi_en, ir), verbose=False)
    id_macs, _ = profile(id_model, inputs=(fu,), verbose=False)

    ie_flops = ie_macs / 1e9
    if_flops = if_macs / 1e9
    id_flops = id_macs / 1e9
    total_flops = ie_flops + if_flops + id_flops

    print(
        f"[FLOPs] Input: {h} × {w} | "
        f"IE: {ie_flops:.4f} GFLOPs | "
        f"IF: {if_flops:.4f} GFLOPs | "
        f"ID: {id_flops:.4f} GFLOPs | "
        f"Total: {total_flops:.4f} GFLOPs"
    )
    return ie_flops, if_flops, id_flops, total_flops

def count_params1(model, trainable_only=False):
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)

def count_params(ie_model, if_model, id_model):
    # 计算参数
    ie_params = count_params1(ie_model) / 1e6
    if_params = count_params1(if_model) / 1e6
    id_params = count_params1(id_model) / 1e6
    total_params = ie_params + if_params + id_params
    print(
        f"[Params] IE: {ie_params:.4f} M | "
        f"IF: {if_params:.4f} M | "
        f"ID: {id_params:.4f} M | "
        f"Total: {total_params:.4f} M"
    )
    return ie_params,if_params,id_params 

def eval_fun(
    ie_model,
    if_model,
    id_model,
    root_dir="/root/gpufree-data/datasets/test/msrs",
    enable_save=True,
    enable_timing=True,
    warmup_iters=10,
):
    device = next(ie_model.parameters()).device
    use_cuda = device.type == "cuda"

    ie_model.eval()
    if_model.eval()
    id_model.eval()

    vi_path = os.path.join(root_dir, "vi")
    ir_path = os.path.join(root_dir, "ir")

    save_dirs = {
        "viEn": os.path.join(root_dir, "ULightIF_viEn"),
        "fu":   os.path.join(root_dir, "ULightIF_fu"),
        "fuEn": os.path.join(root_dir, "ULightIF_fuEn"),
        "fuR":  os.path.join(root_dir, "ULightIF_fuR"),
        "fuL":  os.path.join(root_dir, "ULightIF_fuL"),
    }

    if enable_save:
        for name, path in save_dirs.items():
            os.makedirs(path, exist_ok=True)
            print(f"[SaveDir] {name}: {path}")

    img_list = sorted([x for x in os.listdir(vi_path) if is_image_file(x)])
    if not img_list:
        print("未找到可处理图像。")
        return


    if enable_timing and warmup_iters > 0:
        warmup_name = img_list[0]
        warmup_vi = ImreadRGB(os.path.join(vi_path, warmup_name)).to(device)
        warmup_ir = ImreadRGB(os.path.join(ir_path, warmup_name)).to(device)
        print(f"[Warmup] Running {warmup_iters} warmup iterations...")
        with torch.no_grad():
            for _ in range(warmup_iters):
                warmup_viEn = ie_model(warmup_vi)
                warmup_fu, warmup_fuEn = if_model(warmup_vi, warmup_viEn, warmup_ir)
                warmup_fuR, warmup_fuL = id_model(warmup_fu)
        if use_cuda:
            torch.cuda.synchronize()
        del warmup_vi, warmup_ir
        del warmup_viEn, warmup_fu, warmup_fuEn, warmup_fuR, warmup_fuL
        if use_cuda:
            torch.cuda.empty_cache()

    total_infer_time = 0.0
    valid_img_count = 0
    with torch.no_grad():
        for img_name in tqdm(img_list, desc="Processing images", unit="image"):
            img_vi = ImreadRGB(os.path.join(vi_path, img_name)).to(device)
            img_ir = ImreadRGB(os.path.join(ir_path, img_name)).to(device)
            if enable_timing:
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
            img_viEn = ie_model(img_vi)
            fu, fuEn = if_model(img_vi, img_viEn, img_ir)
            fuR, fuL = id_model(fu)
            if enable_timing:
                if use_cuda:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                infer_time = end_time - start_time
                total_infer_time += infer_time
                valid_img_count += 1
            if enable_save:
                ImgRGB_save(img_viEn, os.path.join(save_dirs["viEn"], img_name))
                ImgRGB_save(fu,      os.path.join(save_dirs["fu"], img_name))
                ImgRGB_save(fuEn,    os.path.join(save_dirs["fuEn"], img_name))
                ImgRGB_save(fuR,     os.path.join(save_dirs["fuR"], img_name))
                ImgGray_save(fuL,    os.path.join(save_dirs["fuL"], img_name))


    if enable_timing and valid_img_count > 0:
        avg_time = total_infer_time / valid_img_count
        ms_per_img = avg_time * 1000.0
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        print(
            f"[Timing] Images: {valid_img_count} | "
            f"Warmup: {warmup_iters} iters | "
            f"Total infer time: {total_infer_time:.4f}s | "
            f"Avg: {ms_per_img:.3f} ms/img | "
            f"FPS: {fps:.3f}"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ie_model = MobileIENetS().to(device)
    ie_model.load_state_dict(torch.load(r"weights/ie_weights.pth", weights_only=True, map_location=device))

    if_model = MobileFNetS().to(device)
    if_model.load_state_dict(torch.load(r"weights/if_weights.pth", weights_only=True, map_location=device))

    id_model = MobileIDNetS().to(device)
    id_model.load_state_dict(torch.load(r"weights/id_weights.pth", weights_only=True, map_location=device))

    count_flops(ie_model, if_model, id_model, h=640, w=480)   # 统计FLOPs
    count_params(ie_model, if_model, id_model, )  # 计算参数

    eval_fun(ie_model,if_model,id_model,
        root_dir="/root/gpufree-data/datasets/test/msrs",
        enable_save=True,
        enable_timing=True,
    )