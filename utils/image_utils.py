import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.io import imsave

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def rgb2ycrcb_torch(rgb_image: torch.Tensor) -> torch.Tensor:
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    y = 0.299 * R + 0.587 * G + 0.114 * B
    cr = (R - y) * 0.713 + 0.5
    cb = (B - y) * 0.564 + 0.5
    y = y.clamp(0.0, 1.0)
    cr = cr.clamp(0.0, 1.0).detach()
    cb = cb.clamp(0.0, 1.0).detach()
    return y, cr, cb


def ycrcb2rgb_torch(ycrcb: torch.Tensor) -> torch.Tensor:
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                       ).to(ycrcb.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(ycrcb.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out





def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def ImreadRGB(path):
    Img = image_read_cv2(path, mode='RGB')[np.newaxis, ...] / 255.0
    Img = np.transpose(Img, (0, 3, 1, 2))
    Img = torch.FloatTensor(Img)
    return Img


def ImgRGB_save(out, path):  # cr,cb,
    out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
    out = (out * 255.0).cpu().numpy().squeeze(0).astype('uint8')
    out = np.transpose(out, (1, 2, 0))
    imsave(path, out)


def ImgGray_save(out, path):
    out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
    out = (out * 255.0).cpu().numpy().squeeze(0).squeeze(0).astype('uint8')
    imsave(path, out)


def MultiCrop(vi, ir, length):
    if (ir.shape[-2] % length != 0) or (ir.shape[-1] % length != 0):
        new_h = ir.shape[-2] - ir.shape[-2] % length

        ir = ir[:, :, :new_h, :]
        vi = vi[:, :, :new_h, :]
    if ir.shape[-1] % length != 0:
        new_w = ir.shape[-1] - ir.shape[-1] % length
        ir = ir[:, :, :, :new_w]
        vi = vi[:, :, :, :new_w]
    return vi, ir





def gauss_cdf(x):
    return 0.5*(1+torch.erf(x/torch.sqrt(torch.tensor(2., device=x.device))))

def gauss_kernel(kernlen=21,nsig=3,channels=1,device=None):
    interval=(2*nsig+1.)/(kernlen)
    x=torch.linspace(-nsig-interval/2.,nsig+interval/2.,kernlen+1, device=device)
    kern1d=torch.diff(gauss_cdf(x))
    kernel_raw=torch.sqrt(torch.outer(kern1d,kern1d))
    kernel=kernel_raw/torch.sum(kernel_raw)
    out_filter=kernel.view(1,1,kernlen,kernlen)
    out_filter = out_filter.repeat(channels,1,1,1)
    return  out_filter

def blur(x):
    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1), device=device)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    return torch.nn.functional.conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1,output2


def add_noise(x, noise_min=15, noise_max=15):
    B, C, H, W = x.shape
    noise_level = torch.randint(noise_min, noise_max+1, (B, 1, 1, 1),
                                device=x.device, dtype=x.dtype) / 255.0
    noise = torch.randn_like(x) * noise_level
    noisy = torch.clamp(x + noise, 0.0, 1.0)
    return noisy
