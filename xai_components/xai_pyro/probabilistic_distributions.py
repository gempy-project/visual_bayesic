from torch.distributions import constraints

from xai_components.base import InArg, OutArg, Component, xai_component
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch

import matplotlib.image as mpimg
from io import BytesIO


@xai_component
class Normal(Component):
    mean: InArg[float]
    std: InArg[float]
    fn: OutArg[any]

    def __init__(self):
        self.mean = InArg.empty()
        self.std = InArg.empty()
        self.fn = OutArg.empty()

    def execute(self, ctx) -> None:
        self.fn.value = dist.Normal(self.mean.value, self.std.value)
