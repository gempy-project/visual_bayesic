from xai_components.base import InArg, OutArg, Component, xai_component
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch

@xai_component
class DefinePrior(Component):
    mean: InArg[float]
    std: InArg[float]
    prior_function: OutArg[any]

    def __init__(self):
        self.mean = InArg.empty()
        self.std = InArg.empty()
        self.prior_function = OutArg.empty()

    def execute(self, ctx) -> None:
        def prior():
            return pyro.sample("mu", dist.Normal(self.mean.value, self.std.value))

        self.prior_function.value = prior

@xai_component
class DefineLikelihood(Component):
    mean: InArg[any]
    std: InArg[float]
    likelihood_function: OutArg[any]

    def __init__(self):
        self.mean = InArg.empty()
        self.std = InArg.empty()
        self.likelihood_function = OutArg.empty()

    def execute(self, ctx) -> None:
        def likelihood(data):
            mu = self.mean.value()
            return pyro.sample("obs", dist.Normal(mu, self.std.value), obs=data)

        self.likelihood_function.value = likelihood




@xai_component
class NUTSKernel(Component):
    model_function: InArg[any]
    kernel: OutArg[any]

    def __init__(self):
        self.model_function = InArg.empty()
        self.kernel = OutArg.empty()

    def execute(self, ctx) -> None:
        self.kernel.value = NUTS(self.model_function.value)


@xai_component
class RunMCMC(Component):
    kernel: InArg[any]
    num_samples: InArg[int]
    warmup_steps: InArg[int]
    data: InArg[list[float]]
    posterior_samples: OutArg[list[float]]

    def __init__(self):
        self.kernel = InArg.empty()
        self.num_samples = InArg.empty()
        self.warmup_steps = InArg.empty()
        self.data = InArg.empty()
        self.posterior_samples = OutArg.empty()

    def execute(self, ctx) -> None:
        data = torch.tensor(self.data.value)
        mcmc = MCMC(self.kernel.value, num_samples=self.num_samples.value, warmup_steps=self.warmup_steps.value)
        mcmc.run(data)
        self.posterior_samples.value = mcmc.get_samples()["mu"].numpy().tolist()

from xai_components.base import InArg, Component, xai_component
import matplotlib.pyplot as plt

@xai_component
class PlotPosterior(Component):
    posterior_samples: InArg[list[float]]

    def __init__(self):
        self.posterior_samples = InArg.empty()

    def execute(self, ctx) -> None:
        plt.hist(self.posterior_samples.value, bins=30, density=True)
        plt.title("Posterior distribution of mu")
        plt.xlabel("mu")
        plt.ylabel("Density")
        plt.show()
