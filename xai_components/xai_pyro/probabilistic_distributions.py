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
        
        def normal_wrapper():
            mean_value = self.mean.value
            # analyze if it is a scalar or a function. If it is a function, then we need to execute it
            if callable(mean_value):
                mean_value = mean_value()
            
            std_value = self.std.value
            return dist.Normal(mean_value, std_value)
        
        self.fn.value = normal_wrapper


@xai_component
class Uniform(Component):
    low: InArg[float]
    high: InArg[float]
    fn: OutArg[any]

    def __init__(self):
        self.low = InArg.empty()
        self.high = InArg.empty()
        self.fn = OutArg.empty()

    def execute(self, ctx) -> None:
        self.fn.value = dist.Uniform(self.low.value, self.high.value)