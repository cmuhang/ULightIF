import torch
import torch.nn as nn
from utils.image_utils import *

class IE_Network_Slim(nn.Module):
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

class IF_Network_Slim(nn.Module):
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
    
class ID_Network_Slim(nn.Module):
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