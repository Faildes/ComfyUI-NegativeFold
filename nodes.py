import torch
from typing import Any, Dict, List, Tuple

def _as_list(x):
    return x if isinstance(x, list) else [x]


def _pairwise(pos_list, neg_list):
    if len(neg_list) == 0:
        raise ValueError("negative conditioning is empty")
    if len(neg_list) == 1 and len(pos_list) > 1:
        neg_list = neg_list * len(pos_list)
    return pos_list, neg_list


def _match_batch(x: torch.Tensor, target_b: int) -> torch.Tensor:
    b = x.shape[0]
    if b == target_b:
        return x
    if b == 1:
        return x.expand(target_b, *x.shape[1:])
    if target_b % b == 0:
        rep = target_b // b
        return x.repeat(rep, *([1] * (x.ndim - 1)))
    raise ValueError(f"Batch mismatch: {b} -> {target_b}")


def _align_tokens(neg: torch.Tensor, target_t: int) -> torch.Tensor:
    # neg: (B, Tn, D) -> (B, target_t, D)
    b, tn, d = neg.shape
    if tn == target_t:
        return neg
    if tn > target_t:
        return neg[:, :target_t, :]
    pad = neg.new_zeros((b, target_t - tn, d))
    return torch.cat((neg, pad), dim=1)


def _fold_tensor(
    pos: torch.Tensor,
    neg: torch.Tensor,
    scale: float,
    rescale: float,
    eps: float,
) -> torch.Tensor:
    # pos/neg: (B,T,D) or pooled: (B,D)
    neg = neg.to(device=pos.device, dtype=pos.dtype)

    if pos.ndim == 3:
        neg = _match_batch(neg, pos.shape[0])
        neg = _align_tokens(neg, pos.shape[1])

        pf = pos.float()
        nf = neg.float()

        # fold: x = pos + s*(pos - neg)
        xf = pf + float(scale) * (pf - nf)

        if rescale > 0.0:
            std_p = pf.std(dim=(1, 2), keepdim=True)
            std_x = xf.std(dim=(1, 2), keepdim=True)
            xf = xf * (std_p / (std_x + eps))

            n_p = pf.norm(dim=2, keepdim=True)
            n_x = xf.norm(dim=2, keepdim=True)
            ratio = n_p / (n_x + eps)
            ratio = ratio.clamp(0.25, 4.0)
            xf = xf * ratio

            xf = pf + float(rescale) * (xf - pf)

        return xf.to(dtype=pos.dtype)

    elif pos.ndim == 2:
        neg = _match_batch(neg, pos.shape[0])

        pf = pos.float()
        nf = neg.float()
        xf = pf + float(scale) * (pf - nf)

        if rescale > 0.0:
            std_p = pf.std(dim=1, keepdim=True)
            std_x = xf.std(dim=1, keepdim=True)
            xf = xf * (std_p / (std_x + eps))

            n_p = pf.norm(dim=1, keepdim=True)
            n_x = xf.norm(dim=1, keepdim=True)
            ratio = (n_p / (n_x + eps)).clamp(0.25, 4.0)
            xf = xf * ratio

            xf = pf + float(rescale) * (xf - pf)

        return xf.to(dtype=pos.dtype)

    else:
        raise ValueError(f"Unsupported tensor rank: {pos.ndim}")


def fold_conditioning(
    positive: List,
    negative: List,
    scale: float,
    rescale: float = 0.7,
    eps: float = 1e-6,
    fold_pooled: bool = True,
) -> List:
    pos_list = _as_list(positive)
    neg_list = _as_list(negative)
    pos_list, neg_list = _pairwise(pos_list, neg_list)

    out = []
    for (pos_cond, pos_meta), (neg_cond, neg_meta) in zip(pos_list, neg_list):
        new_cond = _fold_tensor(pos_cond, neg_cond, scale=scale, rescale=rescale, eps=eps)

        new_meta: Dict[str, Any] = dict(pos_meta) if isinstance(pos_meta, dict) else {}

        if fold_pooled and isinstance(pos_meta, dict) and isinstance(neg_meta, dict):
            if "pooled_output" in pos_meta and "pooled_output" in neg_meta:
                ppo = pos_meta["pooled_output"]
                npo = neg_meta["pooled_output"]
                if isinstance(ppo, torch.Tensor) and isinstance(npo, torch.Tensor):
                    new_meta["pooled_output"] = _fold_tensor(ppo, npo, scale=scale, rescale=rescale, eps=eps)

        out.append([new_cond, new_meta])

    return out


class FoldNegativeIntoPositiveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "scale": ("FLOAT", {"default": 5.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "fold_pooled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning"

    def apply(self, positive, negative, scale, rescale, fold_pooled=True):
        folded = fold_conditioning(
            positive=positive,
            negative=negative,
            scale=float(scale),
            rescale=float(rescale),
            fold_pooled=bool(fold_pooled),
        )
        return (folded,)


NODE_CLASS_MAPPINGS = {
    "FoldNegativeIntoPositiveConditioning": FoldNegativeIntoPositiveConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoldNegativeIntoPositiveConditioning": "Fold Negative Into Positive (Conditioning)",
}
