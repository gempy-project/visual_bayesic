"""
Normal Prior, several observations
``````````````````````````````````


"""
# sphinx_gallery_thumbnail_number = -1

import arviz as az
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from matplotlib.ticker import StrMethodFormatter

from gempy_probability.plot_posterior import PlotPosterior
from pyro.infer import Predictive, NUTS, MCMC

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)


# %%
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


# %%
# 1. Prior Sampling
prior = Predictive(model, num_samples=100)("normal_distribution", y_obs_list)
# 2. MCMC Sampling
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100)  # Assuming 1000 warmup steps
mcmc.run("normal_distribution", y_obs_list)
# Get posterior samples
posterior_samples = mcmc.get_samples(1100)
# 3. Sample from Posterior Predictive
posterior_predictive = Predictive(model, posterior_samples)("normal_distribution", y_obs_list)
# %%
data = az.from_pyro(
    posterior=mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
az_data = data
az.plot_trace(az_data)
plt.show()

# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_bell=True)
p.likelihood_axes.set_xlim(1.90, 2.2)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 9), joyplot=True, marginal=False, likelihood=False, n_samples=31)
p.plot_joy(('$\mu$', '$\sigma$'), '$y$', iteration=14)
plt.show()

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(
    var_names=['$\mu$', '$\sigma$'],
    plot_trace=False,
    credible_interval=0.95,
    kind='kde',
    joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
    joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)

plt.show()

# %%
# License
# =======
# The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:
# 
# https://opensource.org/licenses/BSD-3-Clause
# 
# The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:
# 
# https://creativecommons.org/licenses/by-nc/4.0/
# Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
