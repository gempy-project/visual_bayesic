import torch
import pyro
from torch.distributions import Distribution
from xai_components.base import InArg, OutArg, Component, xai_component

@xai_component
class Sample(Component):
    name: InArg[str]
    fn: InArg[Distribution]
    obs: InArg[torch.tensor]
    sample: OutArg[any]

    def __init__(self):
        self.name = InArg.empty()
        self.fn = InArg.empty()
        self.obs = InArg.empty()
        self.sample = OutArg.empty()

    def execute(self, ctx) -> None:
        self.sample.value = pyro.sample(self.name.value, self.fn.value, obs=self.obs.value)
