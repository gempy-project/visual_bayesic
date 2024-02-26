"""
Uniform Prior, several observations
===================================

"""
# sphinx_gallery_thumbnail_number = -1

import arviz as az
import matplotlib.pyplot as plt
import pyro
import torch
from matplotlib.ticker import StrMethodFormatter

from gempy_probability.plot_posterior import PlotPosterior
from _aux_func import infer_model

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)


# %%
az_data = infer_model(
    distributions_family="uniform_distribution",
    data=y_obs_list
)
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
# No matter which probability density function we choose, for real applications we will never find the exact data 
# generated process---neither we will be able to say if we have found it for that matter---due to an oversimplification 
# of reality. For most applications, the usual families of probability density functions and transformations of those 
# are more than enough approximations for the purpose of the model. In Chapter [sec:model_selection], we will delve into 
# this topic.
#
# Once the model is defined we need to infer the set of parameters \( \varTheta \) of the family of density functions 
# over the observational space, \( \pi_S(y;\varTheta) \). In the case of the normal family, we need to infer the value 
# of the mean, \( \mu \) and standard deviation \( \sigma \). Up to this point, all the description of the probabilistic 
# modelling is agnostic in relation to Frequentist or Bayesian views. These two methodologies diverge on how they 
# infer \( \varTheta \).


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
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde',
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


