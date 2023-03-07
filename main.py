# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf

from contextlib import nullcontext
from types import SimpleNamespace
from copy import deepcopy
from tqdm import tqdm
import math

import wandb
import torch
from torch import autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from consistency_models import karras_boundaries
from consistency_models.utils import get_data, parse_args

from diffusers import UNet2DModel

WANDB_PROJECT = "consistency-model"

config = SimpleNamespace(
    img_size=32,
    batch_size=128,
    num_workers=8,
    dataset="mnist",
    lr=4e-4,
    n_steps=100000,
    sample_every_n_epoch=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mixed_precision=True,
    s_0=2, 
    s_1=150,
    mu_0=0.95,
    ema_rate=0.9,
    n_samples=64,
)


class UNet(UNet2DModel):
    eps: float = 0.002
    def forward(self, x, t): 
        if isinstance(t, float):
            t = (torch.ones(x.shape[0]) * t).long().to(self.device)
        return super().forward(x, t.squeeze()).sample

    @torch.no_grad()
    def sample(self, x, ts):
        x = self(x, ts[0])

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t**2 - self.eps**2) * z
            x = self(x, t)

        return x

def consistency_loss(model, ema_model, x, t1, t2):
    z = torch.randn_like(x)
    x2 = x + z * t2[:, :, None, None]
    x2 = model(x2, t2)

    with torch.no_grad():
        x1 = x + z * t1[:, :, None, None]
        x1 = ema_model(x1, t1)

    return F.mse_loss(x1, x2)

def Nk(step, s_0=config.s_0, s_1=config.s_1):
    return math.ceil(math.sqrt((step * ((s_1+1)**2 - s_0**2) ) + s_0**2) - 1) + 1

class EMA:
    def __init__(self, model, device=config.device):
        self.model = model
        self.ema_model = deepcopy(model).eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def update(self, N):
        mu = math.exp(config.s_0 * math.log(config.mu_0) / N)
        # update \theta_{-}
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_p.mul_(mu).add_(p, alpha=1 - mu)
    
def sample(model, x, values=[5.0, 10.0, 20.0, 40.0, 80.0], n=config.n_samples):
    "Sample images from model"
    x = x[:n]
    model.eval()
    with torch.inference_mode():
        xh = model.sample(
            torch.randn_like(x).to(device=config.device) * 80.0,
            list(reversed(values)),
        )
        xh = (xh * 0.5 + 0.5).clamp(0, 1)
        wandb.log({f"sampled_images_{len(values)}": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in xh]})
    
def save(model, epoch, model_name):
    "Save model weights"
    torch.save(model.state_dict(), f"./ct_{model_name}.pth")
    at = wandb.Artifact("model", type="model", description="Model weights for Consistency Model", metadata={"epoch": epoch})
    at.add_file(f"./ct_{model_name}.pth")
    wandb.log_artifact(at)


def train(config):
    dataloader = get_data(config.dataset, 
                          config.batch_size, 
                          config.num_workers)
    n_channels = 3 if config.dataset=="cifar10" else 1
    model = UNet(in_channels=n_channels, out_channels=n_channels, 
                 block_out_channels=(32, 64, 128, 256), norm_num_groups=8)
    model.to(config.device)
    optim = torch.optim.AdamW(model.parameters(), eps=1e-5)
    scheduler = OneCycleLR(optim, max_lr=config.lr, total_steps=config.n_steps)
    # scheduler = CosineAnnealingLR(optim, T_max=config.n_epochs*len(dataloader), eta_min=0.0)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, device=config.device)
    step_ct = 1
    epoch = 1

    while step_ct < config.n_steps:        
        pbar = tqdm(dataloader)
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(config.device)
            N = Nk(step_ct/config.n_steps)
            boundaries = karras_boundaries(7.0, 0.002, N, 80.0).to(config.device)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=config.device)
            t_1 = boundaries[t]
            t_2 = boundaries[t + 1]
            with autocast("cuda", dtype=torch.float16) if config.mixed_precision else nullcontext():
                loss = consistency_loss(model, ema.ema_model, x, t_1, t_2)
            scaler.scale(loss).backward() if config.mixed_precision else loss.backward()

            if config.mixed_precision:
                scaler.step(optim) 
                scaler.update()
            else:
                optim.step()
            scheduler.step()
            ema.update(N)
            step_ct += 1
            wandb.log({"loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "N": N})    
            pbar.set_description(f"epoch: {epoch}, loss: {loss:.10f}, N: {N:.10f}")
        wandb.log({"epoch": epoch})
        epoch +=1
        if epoch % config.sample_every_n_epoch == 0:
            sample(model, x, epoch, values=[5.0, 10.0, 20.0, 40.0, 80.0])
            sample(model, x, epoch, values=[5.0, 80.0])
    
    # only save model at the end
    save(model, epoch, model_name=config.dataset)

if __name__ == "__main__":
    parse_args(config)
    with wandb.init(project=WANDB_PROJECT, config=config):
        train(config)
