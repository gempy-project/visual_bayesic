import torch
import pyro
from pyro.infer import Predictive, NUTS, MCMC
from xai_components.base import InArg, OutArg, Component, xai_component, dynalist, InCompArg


@xai_component(color="#b47194")
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


@xai_component(color="#776f85")
class NUTS(Component):
    model: InArg[callable]
    NUTS: OutArg[NUTS]

    def execute(self, ctx) -> None:
        model = self.model.value
        nuts_kernel = pyro.infer.NUTS(model)
        self.NUTS.value = nuts_kernel
        

@xai_component(color="#776f85")
class MCMC(Component):
    NUTS: InArg[NUTS]
    num_samples: InArg[int]
    num_chains: InArg[int]
    mcmc: OutArg[pyro.infer.MCMC]
    
    def execute(self, ctx) -> None:
        nuts_kernel = self.NUTS.value
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=self.num_samples.value, num_chains=self.num_chains.value)
        self.mcmc.value = mcmc
        
        
@xai_component(color="#776f85")
class RunMCMC(Component):
    mcmc: InArg[any]
    args: InArg[dynalist]
    # posterior_samples: OutArg[any]
    
    def execute(self, ctx) -> None:
        mcmc = self.mcmc.value
        args_list = self.args.value
        posterior_samples = mcmc.run(args_list)
        # self.posterior_samples.value = posterior_samples
        
        
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


@xai_component(color="#70A3B3")
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


@xai_component(color="#DA8886")
class PosteriorPredictive(Component):
    model: InArg[callable]
    MCMC: InArg[pyro.infer.MCMC]
    num_samples: InArg[int]
    args: InArg[dynalist]
    posterior_predictive: OutArg[any]
    
    def execute(self, ctx) -> None:
        model = self.model.value
        posterior_samples = self.MCMC.value.get_samples(num_samples=self.num_samples.value)
        predictive = pyro.infer.Predictive(model, posterior_samples)
        args_list = self.args.value
        posterior_predictive = predictive(args_list)
        self.posterior_predictive.value = posterior_predictive



@xai_component(color="#776f85")
class FullInference(Component):
    model: InCompArg[callable]
    num_samples: InArg[int]
    num_chains: InArg[int]
    args: InArg[dynalist]
    mcmc: OutArg[pyro.infer.MCMC]
    prior_predictive: OutArg[dict]
    posterior_predictive: OutArg[dict]

    def __init__(self):
        super().__init__()
        self.model = InArg.empty()
        self.num_samples.value = 100
        self.num_chains.value = 1
        self.args = InArg.empty()
        self.mcmc = OutArg.empty()
        self.posterior_predictive = OutArg.empty()
        self.prior_predictive = OutArg.empty()

    def execute(self, ctx) -> None:
        model = self.model.value
        num_samples = self.num_samples.value
        num_chains = self.num_chains.value
        args_list = self.args.value
        nuts_kernel = pyro.infer.NUTS(model)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples, num_chains=num_chains)
        posterior_samples = mcmc.run(args_list)
        self.mcmc.value = mcmc
        
        # * Prior predictive
        predictive = pyro.infer.Predictive(
            model=model, 
            posterior_samples=mcmc.get_samples(num_samples)
        )
        self.posterior_predictive.value = predictive(args_list)


        # * Prior predictive
        prior_predictive = pyro.infer.Predictive(model, num_samples=num_samples)
        self.prior_predictive.value = prior_predictive(args_list)
