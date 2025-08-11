from collections.abc import Iterable
from jaxtyping import Float, Int
import torch
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    maxes, _ = in_features.max(dim=dim, keepdim=True)
    out_features = (in_features - maxes).exp()
    return out_features / out_features.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    maxes, _ = inputs.max(dim=-1, keepdim=True)
    shift_features = inputs - maxes
    logsumexp = shift_features.exp().sum(dim=-1, keepdim=True).log()
    return (logsumexp - shift_features.gather(-1, targets.unsqueeze(-1))).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    norms = []
    for param in parameters:
        grad = param.grad
        if grad is not None:
            norm = grad.norm()
            norms.append(norm)
    all_norm = torch.stack(norms).norm().item()
    for param in parameters:
        grad = param.grad
        if grad is not None:
            if all_norm > max_l2_norm:
                grad *= max_l2_norm / (all_norm + 1e-6)
