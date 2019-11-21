from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from kornia.utils import one_hot


# based on:
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> args = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(*args)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: nn.Parameter = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {} and {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        # target_one_hot = one_hot(target, num_classes=input.shape[1],
        #                          device=input.device, dtype=input.dtype)

        # compute the actual focal loss
        weight = self.alpha * torch.pow(torch.tensor(1.) - input_soft,
                           self.gamma.to(input.dtype))
        focal = weight * torch.log(input_soft)

        return F.nll_loss(focal, target, reduction=self.reduction)

        # loss_tmp = torch.sum(target_one_hot * -focal, dim=1)

        # if self.reduction == 'none':
        #     loss = loss_tmp
        # elif self.reduction == 'mean':
        #     loss = torch.mean(loss_tmp)
        # elif self.reduction == 'sum':
        #     loss = torch.sum(loss_tmp)
        # else:
        #     raise NotImplementedError("Invalid reduction mode: {}"
        #                               .format(self.reduction))
        # return loss


######################
# functional interface
######################


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: Optional[float] = 2.0,
        reduction: Optional[str] = 'mean') -> torch.Tensor:
    r"""Function that computes Focal loss.
    See :class:`~kornia.losses.FocalLoss` for details.
    """
    return FocalLoss(alpha, gamma, reduction)(input, target)