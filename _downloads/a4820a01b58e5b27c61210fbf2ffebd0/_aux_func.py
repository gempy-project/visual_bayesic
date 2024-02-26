"""
Example of a Pyro model and inference
=====================================

"""
# sphinx_gallery_thumbnail_number = -1

# %%
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az


def model(distributions_family, data):
    if distributions_family == "normal_distribution":
        mu = pyro.sample('$\mu$', dist.Normal(2.07, 0.07))
    elif distributions_family in "uniform_distribution":
        mu = pyro.sample('$\mu$', dist.Uniform(0, 10))
    else:
        raise ValueError("distributions_family must be either 'normal_distribution' or 'uniform_distribution'")
    sigma = pyro.sample('$\sigma$', dist.Gamma(0.3, 3))
    y = pyro.sample('$y$', dist.Normal(mu, sigma), obs=data)
    return y


def infer_model(distributions_family, data):
    # 1. Prior Sampling
    prior = Predictive(model, num_samples=100)(distributions_family, data)
    # 2. MCMC Sampling
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100)  # Assuming 1000 warmup steps
    mcmc.run(distributions_family, data)
    # Get posterior samples
    posterior_samples = mcmc.get_samples(1100)
    # 3. Sample from Posterior Predictive
    posterior_predictive = Predictive(model, posterior_samples)(distributions_family, data)
    # %%
    az_data = az.from_pyro(
        posterior=mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )

    return az_data
