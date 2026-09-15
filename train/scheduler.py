from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, config):
    """
    必要に応じて後で warmup や cosine decay を追加できる骨格
    今は恒常 LR（Lambda=1.0）
    """
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    return scheduler
