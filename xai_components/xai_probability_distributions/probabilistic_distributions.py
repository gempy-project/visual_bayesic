from torch.distributions import constraints

from xai_components.base import InArg, OutArg, Component, xai_component
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch

import matplotlib.image as mpimg
from io import BytesIO


@xai_component(color="#70A3B3")
class Normal(Component):
    mean: InArg[float]
    std: InArg[float]
    fn: OutArg[any]

    def __init__(self):
        super().__init__()
        self.mean = InArg.empty()
        self.std = InArg.empty()
        self.fn = OutArg.empty()

    def execute(self, ctx) -> None:
        def normal_wrapper():
            mean_value = self.mean.value
            std_value = self.std.value
            # analyze if it is a scalar or a function. If it is a function, then we need to execute it
            if callable(mean_value):
                mean_value = mean_value()
            
            if callable(std_value):
                std_value = std_value()

            return dist.Normal(mean_value, std_value)

        self.fn.value = normal_wrapper


@xai_component(color="#70A3B3")
class Gamma(Component):
    concentration: InArg[float]
    rate: InArg[float]
    fn: OutArg[any]

    def __init__(self):
        super().__init__()
        self.concentration = InArg.empty()
        self.rate = InArg.empty()
        self.fn = OutArg.empty()

    def execute(self, ctx) -> None:
        def gamma_wrapper():
            concentration_value = self.concentration.value
        
            # analyze if it is a scalar or a function. If it is a function, then we need to execute it
            if callable(concentration_value):
                concentration_value = concentration_value()
            
            # TODO: Add callable check for rate_value

            rate_value = self.rate.value
            return dist.Gamma(concentration_value, rate_value)

        self.fn.value = gamma_wrapper


@xai_component(color="#70A3B3")
class Uniform(Component):
    low: InArg[float]
    high: InArg[float]
    fn: OutArg[any]

    def __init__(self):
        super().__init__()
        self.low = InArg.empty()
        self.high = InArg.empty()
        self.fn = OutArg.empty()

    def execute(self, ctx) -> None:
        def uniform_wrapper():
            low_value = self.low.value
            # analyze if it is a scalar or a function. If it is a function, then we need to execute it
            if callable(low_value):
                low_value = low_value()

            high_value = self.high.value
            return dist.Uniform(low_value, high_value)

        self.fn.value = uniform_wrapper
