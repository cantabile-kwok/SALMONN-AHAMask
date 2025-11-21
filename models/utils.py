# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import StoppingCriteria
from torch import Tensor

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Original code from: https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid/blob/main/gumbel_sigmoid.py  
    We modified the gumbel as the left-tail gumbel distribution.
    
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1), left-tail
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
