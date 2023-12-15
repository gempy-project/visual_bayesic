import torch
import pyro
from torch.distributions import Distribution
from xai_components.base import InArg, OutArg, Component, xai_component

@xai_component
class Sample(Component):
    name: InArg[str]
    fn: InArg[Distribution]
    obs: InArg[list]
    sample: OutArg[any]

    def execute(self, ctx) -> None:
        obs_value = self.obs.value
        if obs_value is not None:
            obs_value = torch.tensor(obs_value, dtype=torch.float32)
        if self.name.value is None:
            self.name.value = "Random Variable"
        
        self.sample.value = pyro.sample(self.name.value, self.fn.value, obs=obs_value)

@xai_component()
class PyroModel(Component):
    fns: InArg[list[callable]]
    model: OutArg[callable]
    
    def __init__(self):
        self.fns = InArg.empty()
        self.model = OutArg.empty()
        
    def execute(self, ctx) -> None:
        def model():
            for fn in self.fns.value:
                fn()
        self.model.value = model


@xai_component
class PriorPredictive(Component):
    num_samples: InArg[int]
    fn: InArg[callable]
    prior: OutArg[any]

    def __init__(self):
        self.num_samples = InArg.empty()
        self.fn = InArg.empty()
        self.prior = OutArg.empty()

    def execute(self, ctx) -> None:
        self.prior.value = pyro.infer.Predictive(self.fn.value, num_samples=self.num_samples.value)

