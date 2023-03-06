# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf


from typing import List
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid


blk = lambda ic, oc: nn.Sequential(
    nn.GroupNorm(32, num_channels=ic),
    nn.SiLU(),
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.GroupNorm(32, num_channels=oc),
    nn.SiLU(),
    nn.Conv2d(oc, oc, 3, padding=1),
)


def kerras_boundaries(sigma, eps, N, T):
    # This will be used to generate the boundaries for the time discretization

    return torch.tensor(
        [
            (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
            ** sigma
            for i in range(N)
        ]
    )


class ConsistencyModel(nn.Module):
    """
    This is ridiculous Unet structure, hey but it works!
    """

    def __init__(self, n_channel: int, eps: float = 0.002) -> None:
        super(ConsistencyModel, self).__init__()

        self.eps = eps
        D = 256
        self.freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=D // 2, dtype=torch.float32)
            / (D // 2)
        )

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, 128, 3, padding=1),
                blk(128, 128),
                blk(128, D),
                blk(D, D),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(D, 128),
            nn.Linear(D, 128),
            nn.Linear(D, D),
            nn.Linear(D, D),
        )

        self.mid = blk(D, D)

        self.up = nn.Sequential(
            *[
                blk(D, D),
                blk(2 * D, 128),
                blk(128, 128),
                nn.Conv2d(256, 256, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(256 + n_channel, n_channel, 3, padding=1)

    def forward(self, x, t) -> torch.Tensor:

        if isinstance(t, float):
            t = (
                torch.tensor([t] * x.shape[0], dtype=torch.float32)
                .to(x.device)
                .unsqueeze(1)
            )
        # time embedding
        args = t.float() * self.freqs[None].to(t.device)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        x = self.last(torch.cat([x, x_ori], dim=1))

        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5)

        return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x

    def loss(self, x, z, t1, t2, ema_model):

        x2 = x + z * t2[:, :, None, None]
        x2 = self(x2, t2)
    
        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)
            
        return F.mse_loss(x1, x2)

    @torch.no_grad()
    def sample(self, x, ts: List[float]):

        x = self(x, ts[0])

        for t in ts[1:]:

            z = torch.randn_like(x)
            x = x + math.sqrt(t**2 - self.eps**2) * z
            x = self(x, t)

        return x


def train_mnist(n_epoch: int = 100, device="cuda:0"):

    model = ConsistencyModel(1)
    model.to(device)

    tf = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(model.parameters(), lr=4e-4)
    
    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(1)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())


    for epoch in range(1, n_epoch):

        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
        print(boundaries)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                
            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)
            
            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")
            
                
        model.eval()
        with torch.no_grad():
            xh = model.sample(
                torch.randn(16, 1, 32, 32, device=device) * 80.0,
                list(reversed([5.0, 20.0, 30.0, 40.0, 50.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_sample_{epoch}.png")

            # save model
            torch.save(model.state_dict(), f"./ct_mnist.pth")


if __name__ == "__main__":
    train_mnist()
