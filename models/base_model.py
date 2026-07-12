import torch
import torch.optim as optim
import lightning.pytorch as pl

from models.build import IE_Network, IF_Network, ID_Network
from utils.image_utils import rgb2ycrcb_torch, pair_downsampler, add_noise
from utils.losses import IDlossf


class PLIEModel(pl.LightningModule):
    def __init__(self, opts=None):
        super().__init__()
        self.opts = opts
        self.net = IE_Network()

    def forward(self, img):
        Ill = self.net(img)
        return torch.clamp(img / Ill, 1e-4, 1.)

    def training_step(self, batch, batch_idx):
        vi_img, ir_img, _ = batch
        loss = self.net._loss(vi_img)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        vi_img, _, _ = batch
        loss = self.net._loss(vi_img)
        self.log("loss_val", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(),lr=self.opts.lr,weight_decay=self.opts.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.opts.optim_step,gamma=self.opts.optim_gamma)
        return [optimizer], [scheduler]

class PLIFModel(pl.LightningModule):
    def __init__(self, ie_ckpt_path, opts=None):
        super().__init__()
        self.opts = opts

        ie_pl = PLIEModel.load_from_checkpoint(ie_ckpt_path,opts=opts,weights_only=False)
        self.ie_net = ie_pl.net.slim()
        self.ie_net.eval()
        for p in self.ie_net.parameters():
            p.requires_grad = False

        self.net = IF_Network()

    def on_train_epoch_start(self):
        self.ie_net.eval()

    def on_validation_epoch_start(self):
        self.ie_net.eval()

    def forward(self, vi_img, ir_img):
        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)

        fu_img, fuEn_img = self.net(vi_img, viEn_img, ir_img)
        return fu_img, fuEn_img

    def training_step(self, batch, batch_idx):
        vi_img, ir_img, _ = batch

        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)

        vi_y, _, _ = rgb2ycrcb_torch(vi_img)
        ir_y, _, _ = rgb2ycrcb_torch(ir_img)
        viEn_y, _, _ = rgb2ycrcb_torch(viEn_img)

        loss = self.net._loss(vi_y, viEn_y, ir_y)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        vi_img, ir_img, _ = batch

        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)

        vi_y, _, _ = rgb2ycrcb_torch(vi_img)
        ir_y, _, _ = rgb2ycrcb_torch(ir_img)
        viEn_y, _, _ = rgb2ycrcb_torch(viEn_img)

        loss = self.net._loss(vi_y, viEn_y, ir_y)

        self.log("loss_val", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(),lr=self.opts.lr,weight_decay=self.opts.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.opts.optim_step,gamma=self.opts.optim_gamma)
        return [optimizer], [scheduler]

class PLIDModel(pl.LightningModule):
    def __init__(self, ie_ckpt_path, if_ckpt_path, opts=None):
        super().__init__()
        self.opts = opts
        self.lossf_id = IDlossf()

        ie_pl = PLIEModel.load_from_checkpoint(ie_ckpt_path,opts=opts,weights_only=False)
        self.ie_net = ie_pl.net.slim()
        self.ie_net.eval()
        for p in self.ie_net.parameters():
            p.requires_grad = False

        if_pl = PLIFModel.load_from_checkpoint(if_ckpt_path,ie_ckpt_path=ie_ckpt_path,opts=opts,weights_only=False)
        self.if_net = if_pl.net.slim()
        self.if_net.eval()
        for p in self.if_net.parameters():
            p.requires_grad = False

        # Stage 3 
        self.net = ID_Network()

    def on_train_epoch_start(self):
        self.ie_net.eval()
        self.if_net.eval()

    def on_validation_epoch_start(self):
        self.ie_net.eval()
        self.if_net.eval()

    def forward(self, vi_img, ir_img):
        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)
            fu_img, fuEn_img = self.if_net(vi_img, viEn_img, ir_img)
        R, L = self.net(fu_img)
        return R, L

    def training_step(self, batch, batch_idx):
        vi_img, ir_img, _ = batch

        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)
            fu_img, fuEn_img = self.if_net(vi_img, viEn_img, ir_img)

        fu_img_noisy = add_noise(fu_img)
        fu1, fu2 = pair_downsampler(fu_img_noisy)
        R1, _ = self.net(fu1)
        R2, _ = self.net(fu2)
        R, L = self.net(fu_img_noisy)

        loss = self.lossf_id(fu_img_noisy, fuEn_img, R, L, R1, R2)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss


    def validation_step(self, batch, batch_idx):
        vi_img, ir_img, _ = batch
        with torch.no_grad():
            viEn_img = self.ie_net(vi_img)
            fu_img, fuEn_img = self.if_net(vi_img, viEn_img, ir_img)

        fu_img_noisy = fu_img
        fu1, fu2 = pair_downsampler(fu_img_noisy)

        R1, _ = self.net(fu1)
        R2, _ = self.net(fu2)
        R, L = self.net(fu_img_noisy)

        loss = self.lossf_id(fu_img_noisy, fuEn_img, R, L, R1, R2)
        self.log("loss_val", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=vi_img.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(),lr=self.opts.lr,weight_decay=self.opts.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.opts.optim_step,gamma=self.opts.optim_gamma)
        return [optimizer], [scheduler]