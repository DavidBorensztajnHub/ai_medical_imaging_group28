"""Loss functions and loss-level regularizers."""

from losses import CrossEntropy
import torch
import torch.nn.functional as F

class BoundaryRegularizer:
    def __init__(self, **kwargs):
        print(f"Initialized {self.__class__.__name__}")

    def __call__(self, pred_probs, onehot_target):
        # Convert multi-class probabilities to foreground probability.
        pred_fg = pred_probs[:, 1:, ...].sum(dim=1, keepdim=True)
        target_fg = onehot_target[:, 1:, ...].sum(dim=1, keepdim=True).float()

        # Horizontal and vertical Sobel filters.
        sobel_x = pred_probs.new_tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]]
        ).unsqueeze(0)

        sobel_y = pred_probs.new_tensor(
            [[[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]]
        ).unsqueeze(0)

        pred_x = F.conv2d(pred_fg, sobel_x, padding=1)
        pred_y = F.conv2d(pred_fg, sobel_y, padding=1)

        target_x = F.conv2d(target_fg, sobel_x, padding=1)
        target_y = F.conv2d(target_fg, sobel_y, padding=1)

        pred_boundary = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-8)
        target_boundary = torch.sqrt(target_x ** 2 + target_y ** 2 + 1e-8)

        return F.l1_loss(pred_boundary, target_boundary)

class CombinedLoss:
    def __init__(self, base_loss, regularizers):
        self.base_loss = base_loss
        self.regularizers = regularizers

    def __call__(self, pred_probs, target):
        loss = self.base_loss(pred_probs, target)

        for weight, regularizer in self.regularizers:
            loss = loss + weight * regularizer(pred_probs, target)

        return loss

# name -> class with __init__(idk, **params) and __call__(probs, onehot_target) -> scalar
LOSSES: dict = {
    "ce": CrossEntropy,
}

# Not used by build_loss yet
REGULARIZERS: dict = {
    "boundary": BoundaryRegularizer,
}


def build_loss(cfg, K: int):
    params = dict(cfg.loss)
    name = params.pop("name")
    classes = params.pop("classes", "all")

    if name not in LOSSES:
        raise KeyError(f"unknown loss '{name}'. Known: {sorted(LOSSES)}")

    base_loss = LOSSES[name](
        idk=list(range(K)) if classes == "all" else list(classes),
        **params,
    )

    regularizers = []

    for reg_cfg in cfg.get("regularizers", []):
        reg_params = dict(reg_cfg)
        reg_name = reg_params.pop("name")
        weight = reg_params.pop("weight", 1.0)

        if reg_name not in REGULARIZERS:
            raise KeyError(
                f"unknown regularizer '{reg_name}'. "
                f"Known: {sorted(REGULARIZERS)}"
            )

        regularizer = REGULARIZERS[reg_name](**reg_params)
        regularizers.append((weight, regularizer))

    if not regularizers:
        return base_loss

    return CombinedLoss(base_loss, regularizers)