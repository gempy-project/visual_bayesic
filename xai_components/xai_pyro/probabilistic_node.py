import torch
import pyro
from xai_components.base import InArg, OutArg, Component, xai_component, dynalist


@xai_component
class Sample(Component):
    name: InArg[str]
    fn: InArg[callable]  #: Function that generates a distribution
    obs: InArg[list]
    sample: OutArg[any]

    def execute(self, ctx) -> None:
        obs_value = self.obs.value
        if obs_value is not None:
            obs_value = torch.tensor(obs_value, dtype=torch.float32)
        if self.name.value is None:
            self.name.value = "Random Variable"  # TODO: Each random variable should have a unique name 

        def sample_wrapper():
            distribution_definition = self.fn.value()
            random_variable_name = self.name.value
            observed_values = obs_value
            return pyro.sample(random_variable_name, distribution_definition, obs=observed_values)

        self.sample.value = sample_wrapper


@xai_component
class NUTS(Component):
    model: InArg[callable]
    NUTS: OutArg[NUTS]

    def execute(self, ctx) -> None:
        model = self.model.value
        nuts_kernel = pyro.infer.NUTS(model)
        self.NUTS.value = nuts_kernel
        
        
@xai_component
class PriorPredictive(Component):  # ? Is this only for priors?
    model: InArg[callable]  #: Function that generates a Python callable containing Pyro primitives.
    num_samples: InArg[int]
    args: InArg[dynalist]
    prior: OutArg[any]

    def __init__(self):
        super().__init__()
        self.num_samples = InArg.empty()
        self.model = InArg.empty()
        self.prior = OutArg.empty()

    def execute(self, ctx) -> None:
        model = self.model.value
        predictive = pyro.infer.Predictive(model, num_samples=self.num_samples.value)
        args_list = self.args.value
        prior_predictive = predictive(args_list)
        self.prior.value = prior_predictive


@xai_component
class MCMC(Component):
    NUTS: InArg[NUTS]
    num_samples: InArg[int]
    num_chains: InArg[int]
    mcmc: OutArg[pyro.infer.MCMC]
    
    def execute(self, ctx) -> None:
        nuts_kernel = self.NUTS.value
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=self.num_samples.value, num_chains=self.num_chains.value)
        self.mcmc.value = mcmc
        
        
@xai_component
class RunMCMC(Component):
    mcmc: InArg[any]
    args: InArg[dynalist]
    posterior_samples: OutArg[any]
    
    def execute(self, ctx) -> None:
        mcmc = self.mcmc.value
        args_list = self.args.value
        posterior_samples = mcmc.run(args_list)
        self.posterior_samples.value = posterior_samples
        
        
@xai_component
class GetSamples(Component):
    mcmc: InArg[any]
    num_samples: InArg[int]
    samples: OutArg[any]
    
    def execute(self, ctx) -> None:
        mcmc = self.mcmc.value
        num_samples = self.num_samples.value
        samples = mcmc.get_samples(num_samples)
        self.samples.value = samples
        

@xai_component
class PosteriorPredictive(Component):
    model: InArg[callable]
    posterior_samples: InArg[any]
    num_samples: InArg[int]
    args: InArg[dynalist]
    posterior_predictive: OutArg[any]
    
    def execute(self, ctx) -> None:
        model = self.model.value
        posterior_samples = self.posterior_samples.value
        num_samples = self.num_samples.value
        predictive = pyro.infer.Predictive(model, posterior_samples, num_samples=num_samples)
        args_list = self.args.value
        posterior_predictive = predictive(args_list)
        self.posterior_predictive.value = posterior_predictive