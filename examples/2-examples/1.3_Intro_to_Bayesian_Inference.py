"""
Uniform Prior, single observation
=================================


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
    data=y_obs
)
az.plot_trace(az_data)
plt.show()


# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde',
                joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
                joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.axjoin.set_xlim(1.96, 2.22)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-6, hide_lines=True)
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


