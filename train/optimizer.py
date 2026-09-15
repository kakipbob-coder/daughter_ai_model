import torch


def build_optimizer(model, config):
    lr = config.get("learning_rate", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    return optimizer
