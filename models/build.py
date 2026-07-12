import torch
import torch.nn as nn

from utils.losses import IElossf,Fulossf

from models.net import MBRConv5,MBRConv3
from models.build_slim import IE_Network_Slim,IF_Network_Slim,ID_Network_Slim


class IE_Network(nn.Module):
    def __init__(self, channels = 24 , rep_scale=4):
        super().__init__()
        self.channels = channels
        self.lossf_ie = IElossf()
        self.headA = nn.Sequential(
            MBRConv3(3, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.body1 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.proj = MBRConv3(channels, 1, rep_scale=rep_scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inp : torch.Tensor) -> torch.Tensor:
        x0 = self.headA(inp)
        x1 = self.body1(x0)
        out = self.sigmoid(self.proj(x1+x0))
        return out

    def _loss(self, img: torch.Tensor) -> torch.Tensor:
        Ill = self(img)
        Ref = torch.clamp(img / Ill, 1e-4, 1.)
        return self.lossf_ie(img.detach(), Ill, Ref)


    def slim(self):
        net_slim = IE_Network_Slim(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5):
               if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w 
                    weight_slim['%s.bias' % name] = b
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim


















class MobileEncoder(nn.Module):
    def __init__(self, channels=32, rep_scale=4):
        super().__init__()
        self.channels = channels
        # Stage A
        self.stem = nn.Sequential(
            MBRConv3(1, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.BlockA = nn.Sequential(MBRConv3(channels, channels, rep_scale=rep_scale),nn.PReLU(channels))
        self.BlockB = nn.Sequential(MBRConv3(channels, channels, rep_scale=rep_scale),nn.PReLU(channels))
        self.BlockC = nn.Sequential(MBRConv3(channels, channels, rep_scale=rep_scale),nn.PReLU(channels))


    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.BlockA(x0) + x0
        x2 = self.BlockB(x1) + x1
        x3 = self.BlockC(x2) + x2
        return x3


class MobileDecoder(nn.Module):
    def __init__(self,channels = 32,rep_scale = 4):
        super().__init__()
        self.BlockA = nn.Sequential(
            MBRConv3(channels * 2, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        # c2 -> c1 通道变换
        self.BlockB = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.BlockC = nn.Sequential(
            MBRConv3(channels, 1, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        
    def forward(self, f1,f2):
        x0 = torch.cat([f1,f2],dim=1)
        x1 = self.BlockA(x0)
        x2 = self.BlockB(x1)
        x3 = self.BlockC(x2)
        return x3


class IF_Network(nn.Module):
    def __init__(self, channels = 16, rep_scale = 4):
        super().__init__()
        self.enc = MobileEncoder(channels=channels,rep_scale=rep_scale)
        self.dec = MobileDecoder(channels=channels,rep_scale=rep_scale)
        self.lossf_if = Fulossf()

    def forward(self, vi: torch.Tensor, vi_en: torch.Tensor, ir: torch.Tensor):
        vi_feat = self.enc(vi)
        viEn_feat = self.enc(vi_en)
        ir_feat = self.enc(ir)
        fu = self.dec(vi_feat, ir_feat)
        fuEn = self.dec(viEn_feat, ir_feat)
        return fu, fuEn

    def _loss(self,vi: torch.Tensor, vi_en: torch.Tensor, ir: torch.Tensor) ->  torch.Tensor:
        fu, fuEn = self(vi,vi_en,ir)
        loss = 0
        loss += self.lossf_if(vi,    ir, fu)
        loss += self.lossf_if(vi_en,  ir, fuEn)
        return loss


    def slim(self):
        net_slim = IF_Network_Slim() 
        weight_slim = net_slim.state_dict()

        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5):
                key_w = f'{name}.weight'
                if key_w in weight_slim:      
                    w, b = mod.slim()
                    weight_slim[f'{name}.weight'] = w
                    weight_slim[f'{name}.bias']   = b
            elif isinstance(mod, nn.PReLU):
                weight_slim[f'{name}.weight'] = mod.weight

        net_slim.load_state_dict(weight_slim, strict=True)
        return net_slim
    


class ID_Network(nn.Module):
    def __init__(self, channels = 32 , rep_scale=4):
        super().__init__()
        self.channels = channels
        self.unit1 = nn.Sequential(
            MBRConv3(3, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.unit2 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.unit3 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )
        self.unit4 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )  
        self.unit5 = nn.Sequential(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
        )  

        self.proj = nn.Sequential(
            MBRConv3(channels, 4, rep_scale=rep_scale),
            nn.Sigmoid(),
        )

    def forward(self,inp : torch.Tensor) -> torch.Tensor:
        x0 = self.unit1(inp) 
        x1 = self.unit2(x0)
        x2 = self.unit3(x1)

        x4 = self.unit4(x2 + x1)
        x5 = self.unit5(x4 + x0)  
        out = self.proj(x5)
        return out[:,:3,:,:],out[:,3:,:,:]

    def slim(self):
        net_slim = ID_Network_Slim()  
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5):
                key_w = f'{name}.weight'
                if key_w in weight_slim:       
                    w, b = mod.slim()
                    weight_slim[f'{name}.weight'] = w
                    weight_slim[f'{name}.bias']   = b
            elif isinstance(mod, nn.PReLU):
                weight_slim[f'{name}.weight'] = mod.weight
        net_slim.load_state_dict(weight_slim, strict=True)
        return net_slim