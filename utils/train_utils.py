import os
import time
import torch
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class ETACallback(Callback):
    def __init__(self, print_every_n_steps=50):
        super().__init__()
        self.print_every_n_steps = print_every_n_steps
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step == 0 or global_step % self.print_every_n_steps != 0:
            return

        elapsed = time.time() - self.start_time
        total_steps = trainer.estimated_stepping_batches
        done_steps = global_step

        if done_steps <= 0:
            return

        steps_left = total_steps - done_steps
        sec_per_step = elapsed / done_steps
        eta_sec = steps_left * sec_per_step

        eta_h = int(eta_sec // 3600)
        eta_m = int((eta_sec % 3600) // 60)
        eta_s = int(eta_sec % 60)

        print(f"\n [ETA] step {done_steps}/{total_steps} | elapsed {elapsed/60:.1f} min | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}")

def seed_everything_strict(seed: int = 3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


