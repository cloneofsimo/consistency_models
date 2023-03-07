# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from copy import deepcopy
from tqdm import tqdm
import math

import wandb
import torch
from torch import autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchvision.utils import save_image, make_grid

from consistency_models import ConsistencyModel, kerras_boundaries
from consistency_models.utils import get_data

WANDB_PROJECT = "consistency-model"

config = SimpleNamespace(
    img_size=32,
    batch_size=128,
    num_workers=8,
    dataset="mnist",
    lr=4e-4,
    n_epochs=10,
    sample_every_n_epoch=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    wandb=True,
    mixed_precision=True,
    s_0=2, 
    s_1=150,
    mu_0=0.9,
    ema_rate=0.999,
)

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
    
def sample(model, x, epoch, values=[5.0, 10.0, 20.0, 40.0, 80.0], n=64):
    "Sample images from model"
    x = x[:n]
    model.eval()
    with torch.inference_mode():
        xh = model.sample(
            torch.randn_like(x).to(device=config.device) * 80.0,
            list(reversed(values)),
        )
        xh = (xh * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(xh, nrow=4)
        save_image(grid, f"./contents/ct_{config.dataset}_sample_{len(values)}step_{epoch}.png")
        if config.wandb:
            wandb.log({f"sampled_images_{len(values)}": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in xh]})
    
def save(model, epoch, model_name):
    "Save model weights"
    torch.save(model.state_dict(), f"./ct_{model_name}.pth")
    if config.wandb:
        at = wandb.Artifact("model", type="model", description="Model weights for Consistency Model", metadata={"epoch": epoch})
        at.add_file(f"./ct_{model_name}.pth")
        wandb.log_artifact(at)


def train(config):
    dataloader = get_data(config.dataset, 
                          config.batch_size, 
                          config.num_workers)
    n_channels = 3 if config.dataset=="cifar10" else 1
    model = ConsistencyModel(n_channels, D=256)
    model.to(config.device)
    optim = torch.optim.AdamW(model.parameters(), eps=1e-5)
    scheduler = OneCycleLR(optim, max_lr=config.lr, 
                           steps_per_epoch=len(dataloader), epochs=config.n_epochs)
    # scheduler = CosineAnnealingLR(optim, T_max=config.n_epochs*len(dataloader), eta_min=0.0)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, device=config.device)
    example_ct = 0
    total_ct = config.n_epochs*len(dataloader)*config.batch_size

    for epoch in range(1, config.n_epochs):
        N = Nk(example_ct/total_ct)
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(config.device)
        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(config.device)
            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=config.device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]
            with autocast("cuda", dtype=torch.float16) if config.mixed_precision else nullcontext():
                loss = model.loss(x, z, t_0, t_1, ema_model=ema.ema_model)
            scaler.scale(loss).backward() if config.mixed_precision else loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = config.ema_rate * loss_ema + (1-config.ema_rate) * loss.item()

            if config.mixed_precision:
                scaler.step(optim) 
                scaler.update()
            else:
                optim.step()
            scheduler.step()
            ema.update(N)
            example_ct += len(x)
            if config.wandb:
                wandb.log({"loss": loss.item(),
                           "loss_ema": loss_ema,
                           "lr": scheduler.get_last_lr()[0],
                           "epoch": config.n_epochs*example_ct/total_ct,
                           "N": N})    
            pbar.set_description(f"epoch: {epoch}, loss: {loss_ema:.4f}, N: {N:.10f}")

        if epoch % config.sample_every_n_epoch == 0:
            sample(model, x, epoch, values=[5.0, 10.0, 20.0, 40.0, 80.0])
            sample(model, x, epoch, values=[5.0, 80.0])
    
    # only save model at the end
    save(model, epoch, model_name=config.dataset)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v), default=v)
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

if __name__ == "__main__":
    parse_args(config)
    if config.wandb:
        wandb.init(project=WANDB_PROJECT, config=config)
    train(config)
