import torch


def kerras_boundaries(sigma, eps, N, T):
    # This will be used to generate the boundaries for the time discretization

    return torch.tensor(
        [
            (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
            ** sigma
            for i in range(N)
        ]
    )
