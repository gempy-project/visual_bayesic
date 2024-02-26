from typing import Union

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
class NormalSampler(Component):
    name: InArg[str]
    mean: InArg[Union[float, callable]]
    std: InArg[float]
    obs: InArg[list]
    sample: OutArg[any]

    def __init__(self):
        super().__init__()
        self.name = InArg.empty()
        self.mean = InArg.empty()
        self.std = InArg.empty()
        self.obs = InArg.empty()
        self.sample = OutArg.empty()

    def execute(self, ctx) -> None:
        obs_value = self.obs.value
        if obs_value is not None:
            obs_value = torch.tensor(obs_value, dtype=torch.float32)
        if self.name.value is None:
            self.name.value = "Random Variable"  # TODO: Each random variable should have a unique name 

        def sample_wrapper():
            if callable(self.mean.value):
                mean_value = self.mean.value()
            else:
                mean_value = self.mean.value    
            
            if callable(self.std.value):
                std_value = self.std.value()
            else:
                std_value = self.std.value
            
            distribution_definition = dist.Normal(mean_value, std_value)
            random_variable_name = self.name.value
            return pyro.sample(random_variable_name, distribution_definition, obs=obs_value)

        self.sample.value = sample_wrapper


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



@xai_component(color="#70A3B3")
class GammaSampler(Component):
    name: InArg[str]
    concentration: InArg[float]
    rate: InArg[float]
    obs: InArg[list]
    sample: OutArg[any]

    def __init__(self):
        super().__init__()
        self.name = InArg.empty()
        self.concentration = InArg.empty()
        self.rate = InArg.empty()
        self.obs = InArg.empty()
        self.sample = OutArg.empty()

    def execute(self, ctx) -> None:
        obs_value = self.obs.value
        if obs_value is not None:
            obs_value = torch.tensor(obs_value, dtype=torch.float32)
        if self.name.value is None:
            self.name.value = "Random Variable"

        def sample_wrapper():
            distribution_definition = dist.Gamma(self.concentration.value, self.rate.value)
            random_variable_name = self.name.value
            observed_values = self.obs.value
            return pyro.sample(random_variable_name, distribution_definition, obs=observed_values)

        self.sample.value = sample_wrapper


@xai_component(color="#70A3B3")
class UniformSampler(Component):
    name: InArg[str]
    low: InArg[float]
    high: InArg[float]
    obs: InArg[list]
    sample: OutArg[any]

    def __init__(self):
        super().__init__()
        self.name = InArg.empty()
        self.low = InArg.empty()
        self.high = InArg.empty()
        self.obs = InArg.empty()
        self.sample = OutArg.empty()

    def execute(self, ctx) -> None:
        obs_value = self.obs.value
        if obs_value is not None:
            obs_value = torch.tensor(obs_value, dtype=torch.float32)
        if self.name.value is None:
            self.name.value = "Random Variable"

        def sample_wrapper():
            distribution_definition = dist.Uniform(self.low.value, self.high.value)
            random_variable_name = self.name.value
            observed_values = self.obs.value
            return pyro.sample(random_variable_name, distribution_definition, obs=observed_values)

        self.sample.value = sample_wrapper
