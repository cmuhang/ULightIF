import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.image_utils import pair_downsampler
from math import exp
import torchvision

_reduction_modes = ['none', 'mean', 'sum']

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu5] #h_relu1, h_relu2, h_relu3,h_relu4,
        return out


class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion = 'l2', reduction='mean'):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')
        self.weights = [1.0] #0.0 / 32, 0.0 / 16, 0.0 / 8,
        self.weight = loss_weight

    def getGray(self,x):
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
        x = torch.sum(x * weights, dim=1, keepdim=True)  # [B, 1, H, W]
        return torch.cat([x, x, x], dim=1)


    def forward(self, x, y):
        x,y = self.getGray(x),self.getGray(y)
        x_vgg,y_vgg = self.vgg(x),self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss




class SPAlossf(nn.Module):
    def __init__(self):
        super(SPAlossf, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)
        self.pool = nn.AvgPool2d(4)

    def GetGray(self,Image):
        weights = torch.tensor([0.299, 0.587, 0.114], device=Image.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
        ImageGray = torch.sum(Image * weights, dim=1, keepdim=True)  # [B, 1, H, W]
        return ImageGray
    
    def forward(self, org , enhance ):
        org_mean = self.GetGray(org)
        enhance_mean = self.GetGray(enhance)
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        loss = torch.mean(D_left + D_right + D_up +D_down)
        return loss
    
class SMOOTHlossf(nn.Module):
    def __init__(self, ker_size=5, sigma=2):
        super().__init__()
        self.ker_size = ker_size
        self.sigma = sigma
        half_size = ker_size // 2
        x = torch.arange(-half_size, half_size + 1).float()
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        gauss_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)  # [1, 1, ker_size, ker_size]
        self.register_buffer('gauss_kernel_2d', gauss_2d)
        
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2)
        smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
        self.register_buffer('smooth_kernel_x', smooth_kernel_x)
        self.register_buffer('smooth_kernel_y', smooth_kernel_y)

    def gradient(self, input_tensor, direction):
        if direction == "x":
            return F.conv2d(input_tensor, self.smooth_kernel_x, padding=1, stride=1)
        elif direction == "y":
            return F.conv2d(input_tensor, self.smooth_kernel_y, padding=1, stride=1)
        else:
            raise ValueError("Direction must be 'x' or 'y'")

    def getweight(self, InitialIll):
        eps = 1e-4
        grad_x = self.gradient(InitialIll, 'x')
        grad_y = self.gradient(InitialIll, 'y')
        padding = self.ker_size // 2
        smooth_grad_x = F.conv2d(grad_x,self.gauss_kernel_2d,padding=padding,)
        smooth_grad_y = F.conv2d(grad_y,self.gauss_kernel_2d,padding=padding,)
        W_x = 1.0 / (torch.abs(smooth_grad_x) + eps)
        W_y = 1.0 / (torch.abs(smooth_grad_y) + eps)
        return W_x, W_y

    def forward(self,input_img,Illumination):
        eps = 1e-4
        input_img = torch.max(input_img, dim=1, keepdim=True)[0] # 取最大通道作为预制
        W_x, W_y = self.getweight(input_img)
        input_img_x = self.gradient(input_img, 'x')
        input_img_y = self.gradient(input_img, 'y')
        Illumination_x = self.gradient(Illumination, 'x')
        Illumination_y = self.gradient(Illumination, 'y')

        loss_smooth_x = W_x * (torch.pow(Illumination_x, 2) / (torch.abs(input_img_x) + eps))
        loss_smooth_y = W_y * (torch.pow(Illumination_y, 2) / (torch.abs(input_img_y) + eps))
        loss_smooth = torch.mean(loss_smooth_x) + torch.mean(loss_smooth_y)
        return loss_smooth

class IElossf(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.spa_loss = SPAlossf()
        self.smooth_loss = SMOOTHlossf()  

    def forward(self,input_img,Illumination,Reflection):
        eps = 1e-9
        input_img = input_img + eps
        weights = torch.tensor([0.299, 0.587, 0.114], device=input_img.device).view(1, 3, 1, 1)
        input_Y_mean = torch.mean((input_img.detach() * weights).sum(dim=1), dim=(1, 2))  
        alpha = 0.6 / (input_Y_mean + eps)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        loss_over = self.l2_loss(
                Reflection,
                torch.clamp(input_img.detach() * alpha, eps, 1.)
        )
        loss_spa = self.spa_loss(input_img.detach(),Reflection)  
        loss_smooth = self.smooth_loss(input_img.detach(), Illumination)
        loss = loss_over + 5 * loss_spa + 0.01 * loss_smooth
        return loss



# ----------------------------------------------------------------------------------------------------------------------------------
class Grad_sobel(nn.Module):
    def __init__(self):
        super(Grad_sobel, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)  
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)  
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)  
        sobely = F.conv2d(x, self.weighty, padding=1)  
        return torch.abs(sobelx) + torch.abs(sobely)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    return 1 - ret

class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()
        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class Fulossf(nn.Module):
    def __init__(self):
        super(Fulossf, self).__init__()
        self.sobel = Grad_sobel()
        self.ssim_loss = L_SSIM(window_size=48)

    def forward(self,viGray: torch.Tensor,irGray: torch.Tensor,fuGray: torch.Tensor ) -> torch.Tensor:
        viGray,irGray = viGray.detach(),irGray.detach()
        loss_int = F.l1_loss(torch.max(viGray,irGray),fuGray)
        vi_grad,ir_grad,fu_grad = self.sobel(viGray),self.sobel(irGray),self.sobel(fuGray)
        loss_tex = F.l1_loss(torch.max(vi_grad,ir_grad),fu_grad)
        loss_ssim = self.ssim_loss(irGray,fuGray) + self.ssim_loss(viGray,fuGray)
        loss_total = loss_int + 10 * loss_tex + loss_ssim
        return loss_total

# ----------------------------------------------------------------------------------------------------------------------------------
class IDlossf(nn.Module):
    def __init__(self, patch_size = 5):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)  #
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SMOOTHlossf()
        self.per = VGGLoss()

    def LocalBrightnessLoss(self, pred, gt):
        weights = torch.tensor([0.299, 0.587, 0.114], device=pred.device).view(1, 3, 1, 1) 
        pred_luminance = torch.sum(pred * weights, dim=1, keepdim=True)  
        gt_luminance = torch.sum(gt * weights, dim=1, keepdim=True)  
        pred_patch_mean = self.pool(pred_luminance) 
        gt_patch_mean = self.pool(gt_luminance)
        loss = self.l2_loss(pred_patch_mean,gt_patch_mean)
        return loss

    def forward(self,fu,fuEn,R,L,R1,R2):
        L3 = torch.cat([L, L, L], dim=1)
        loss1 = self.l2_loss(R,fu/L3) + self.l2_loss(R * L3,fu) + 0.15 * self.smooth_loss(fu,L) 
        loss2 = self.LocalBrightnessLoss(R,fuEn) + 0.5 * self.per(R,fuEn)
        R_1,R_2 = pair_downsampler(R)
        loss3 = 2 * (self.l2_loss(R1, R2) + self.l2_loss(R1, R_1) + self.l2_loss(R2, R_2))
        loss = loss1 + loss2 + loss3
        return loss
