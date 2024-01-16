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

        def sample_wrapper():
            distribution_difinition = self.fn.value()
            random_variable_name = self.name.value
            observed_values = obs_value
            return pyro.sample(random_variable_name, distribution_difinition, obs=observed_values)
        
        self.sample.value = sample_wrapper


@xai_component()
class PyroModel(Component):
    fns: InArg[list[callable]]
    model: OutArg[callable]

    def __init__(self):
        self.fns = InArg.empty()
        self.model = OutArg.empty()

    def execute(self, ctx) -> None:
        def model(fn):
            fn()

        self.model.value = model


@xai_component
class PriorPredictive(Component):  # ? Is this only for priors?
    num_samples: InArg[int]
    fn: InArg[callable]
    args: InArg[list]
    prior: OutArg[any]

    def __init__(self):
        self.num_samples = InArg.empty()
        self.fn = InArg.empty()
        self.prior = OutArg.empty()

    def execute(self, ctx) -> None:
        def model(samples):
            fo = samples()
            
        predictive = pyro.infer.Predictive(model, num_samples=self.num_samples.value)
        prior_predictive = predictive(
            self.args.value
        )
        func_of_the_model = self.fn.value
        self.prior.value = prior_predictive
