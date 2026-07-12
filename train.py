import os
from datetime import datetime
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from models.base_model import PLIEModel, PLIFModel, PLIDModel
from data.paired_data import get_train_loader
from options import get_opts
from utils.train_utils import seed_everything_strict,ETACallback


def main(opts):
    print("Creating data loaders...")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = opts.save_dir
    opts.save_dir = os.path.join(base_save_dir, run_timestamp)
    os.makedirs(opts.save_dir, exist_ok=True)
    print(f"Training checkpoints will be saved to: {opts.save_dir}")

    train_loader = get_train_loader(
        vi_train_path=opts.vi_train_path,
        ir_train_path=opts.ir_train_path,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        crop_size=opts.crop_size,
    )
    # ==================== Stage 1: IE training ====================
    print("\n" + "=" * 60)
    print("Stage 1: Training IE Model")
    print("=" * 60)
    ie_save_dir = os.path.join(opts.save_dir, "IE")
    os.makedirs(ie_save_dir, exist_ok=True)
    ie_checkpoint_callback = ModelCheckpoint(
        dirpath=ie_save_dir,
        save_top_k=0,
        save_last=True,
    )
    ie_model = PLIEModel(opts=opts)
    ie_trainer = pl.Trainer(
        max_epochs=opts.num_epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[ie_checkpoint_callback, ETACallback()],
        logger=False,
    )
    ie_trainer.fit(ie_model, train_loader)
    ie_last_ckpt = os.path.join(ie_save_dir, "last.ckpt")
    print(f"IE training completed. Last checkpoint: {ie_last_ckpt}")


    # ==================== Stage 2: IF training ====================
    print("\n" + "=" * 60)
    print("Stage 2: Training IF Model")
    print("=" * 60)
    if_save_dir = os.path.join(opts.save_dir, "IF")
    os.makedirs(if_save_dir, exist_ok=True)
    if_checkpoint_callback = ModelCheckpoint(
        dirpath=if_save_dir,
        save_top_k=0,
        save_last=True,
    )
    if_model = PLIFModel(ie_ckpt_path=ie_last_ckpt, opts=opts)
    if_trainer = pl.Trainer(
        max_epochs=opts.num_epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[if_checkpoint_callback, ETACallback()],
        logger=False,
    )
    if_trainer.fit(if_model, train_loader)
    if_last_ckpt = os.path.join(if_save_dir, "last.ckpt")
    print(f"IF training completed. Last checkpoint: {if_last_ckpt}")



    # ==================== Stage 3: ID training ====================
    print("\n" + "=" * 60)
    print("Stage 3: Training ID Model")
    print("=" * 60)
    id_save_dir = os.path.join(opts.save_dir, "ID")
    os.makedirs(id_save_dir, exist_ok=True)
    id_checkpoint_callback = ModelCheckpoint(
        dirpath=id_save_dir,
        save_top_k=0,
        save_last=True,
    )
    id_model = PLIDModel(ie_ckpt_path=ie_last_ckpt, if_ckpt_path=if_last_ckpt, opts=opts)
    id_trainer = pl.Trainer(
        max_epochs=opts.num_epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[id_checkpoint_callback, ETACallback()],
        logger=False,
    )
    id_trainer.fit(id_model, train_loader)
    id_last_ckpt = os.path.join(id_save_dir, "last.ckpt")

    print(f"ID training completed. Last checkpoint: {id_last_ckpt}")
    print("\n" + "=" * 60)
    print("All stages completed successfully!")
    print(f"Run directory: {opts.save_dir}")
    print(f"IE last checkpoint: {ie_last_ckpt}")
    print(f"IF last checkpoint: {if_last_ckpt}")
    print(f"ID last checkpoint: {id_last_ckpt}")
    print("=" * 60)

if __name__ == "__main__":
    opts = get_opts()
    seed_everything_strict(opts.seed)
    main(opts)
