#!/usr/bin/env python
# coding: utf-8

# # Logistic growth models of SARS-CoV-2 lineage proportions
# 
# This notebook explores logistic growth models, with the goal of inferring the differential growth rates of different SARS-CoV-2 lineages over time. Before this tutorial you may want to familiarize yourself with [Pyro modeling basics](http://pyro.ai/examples/intro_long.html) and [tensor shapes](http://pyro.ai/examples/tensor_shapes.html) in Pyro and PyTorch.
# 
# **WARNING**: The purpose of this tutorial is to demonstrate Pyro's modeling and inference syntax. 
# Making reliable inferences about SARS-CoV-2 **is not** the purpose of this tutorial.
# 
# #### Table of contents
# 
# * [Overview](#Overview)
# * [Loading data](#Loading-data)
# * [A First Model](#A-first-model)
# * [A regional model](#A-regional-model)
# * [An alternative regional model](#An-alternative-regional-model)
# * [Generalizations](#Generalizations)

# ## Overview <a name="Overview" />
# 
# When different strains/lineages/variants of a virus like SARS-CoV-2 circulate in a population, those lineages that have the largest fitness will tend to dominate, and those lineages that are least fit will tend to be outcompeted by the fittest lineages.
# In this tutorial we set out to infer (differential) growth rates for different 
# SARS-CoV-2 lineages using a spatio-temporal dataset of SARS-CoV-2 genetic sequences.
# We'll start with the simplest possible model and then move on to more complex models with hierarchical structure.

# In[1]:


import os
import datetime
from functools import partial
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
import matplotlib as mpl
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("Using GPU")
    torch.set_default_device("cuda")
else:
    print("Using CPU")

smoke_test = ('CI' in os.environ)  # for use in continuous integration testing


# ## Loading data <a name="Loading-data" />
# 
# Our data consist of a few million genetic sequences of SARS-CoV-2 viruses, clustered into [PANGO lineages](https://cov-lineages.org), and aggregated into a few hundred regions globally and into 28-day time bins. Preprocessing was performed by Nextstrain's [ncov](https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html) tool, and aggregation was performed by the Broad Institute's [pyro-cov](https://github.com/broadinstitute/pyro-cov/blob/master/scripts/preprocess_nextstrain.py) tool.

# In[2]:


from pyro.contrib.examples.nextstrain import load_nextstrain_counts
dataset = load_nextstrain_counts()

def summarize(x, name=""):
    if isinstance(x, dict):
        for k, v in sorted(x.items()):
            summarize(v, name + "." + k if name else k)
    elif isinstance(x, torch.Tensor):
        print(f"{name}: {type(x).__name__} of shape {tuple(x.shape)} on {x.device}")
    elif isinstance(x, list):
        print(f"{name}: {type(x).__name__} of length {len(x)}")
    else:
        print(f"{name}: {type(x).__name__}")
summarize(dataset)


# In this tutorial our interest is in the 3-dimensional tensor
# of counts `dataset["counts"]`, which has shape (T, R, L) where `T` is the number of time bins, `R` is the number of regions, and `L` is the number of strains or PANGO lineages, and `dataset["counts"][t,r,l]` is the number of samples in the corresponding time-region-location bin.
# The count data are heavily skewed towards a few large regions and dominant lineages like `B.1.1.7` and `B.1.617.2`.

# In[3]:


fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
for i, name in enumerate(["time bin", "location", "lineage"]):
    counts = dataset["counts"].sum(list({0, 1, 2} - {i}))
    Y = counts.sort(0, True).values
    axes[i].plot(Y)
    axes[i].set_xlim(0, None)
    axes[0].set_ylim(1, None)
    axes[i].set_yscale("log")
    axes[i].set_xlabel(f"rank", fontsize=18)
    axes[i].set_title(f"{len(Y)} {name}s")
axes[0].set_ylabel("# samples", fontsize=18)
plt.subplots_adjust(wspace=0.05);


# ### Helpers for manipulating data

# In[4]:


def get_lineage_id(s):
    """Get lineage id from string name"""
    return np.argmax(np.array([s]) == dataset['lineages'])

def get_location_id(s):
    """Get location id from string name"""
    return np.argmax(np.array([s]) == dataset['locations'])

def get_aggregated_counts_from_locations(locations):
    """Get aggregated counts from a list of locations"""
    return sum([dataset['counts'][:, get_location_id(loc)] for loc in locations])

start = dataset["start_date"]
step = datetime.timedelta(days=dataset["time_step_days"])
date_range = np.array([start + step * t for t in range(len(dataset["counts"]))])


# ## A first model <a name="A-first-model" />
# 
# First let's zoom-in on Massachusetts and a few surrounding states:

# In[5]:


northeast_states = ['USA / Massachusetts', 
                    'USA / New York', 
                    'USA / Connecticut', 
                    'USA / New Hampshire',
                    'USA / Vermont',
                    'USA / New Jersey',
                    'USA / Maine',
                    'USA / Rhode Island', 
                    'USA / Pennsylvania']

northeast_counts = get_aggregated_counts_from_locations(northeast_states)


# Next let's extract sublineages corresponding to two of the [WHO variants of concern](https://www.who.int/en/activities/tracking-SARS-CoV-2-variants/), Alpha and Delta:

# In[6]:


# The Alpha and Delta variants include many PANGO lineages, which we need to aggregate.
Q_lineages = [lin for lin in dataset['lineages'] if lin[:2] == 'Q.']
AY_lineages = [lin for lin in dataset['lineages'] if lin[:3] == 'AY.']

alpha_lineages = ['B.1.1.7'] + Q_lineages
delta_lineages = ['B.1.617.2'] + AY_lineages

alpha_ids = [get_lineage_id(lin) for lin in alpha_lineages]
delta_ids = [get_lineage_id(lin) for lin in delta_lineages]

alpha_counts = northeast_counts[:, alpha_ids].sum(-1)
delta_counts = northeast_counts[:, delta_ids].sum(-1)


# In[7]:


# Let's combine the counts into a single tensor
alpha_delta_counts = torch.stack([alpha_counts, delta_counts]).T
print(alpha_delta_counts.shape)


# Next let's plot the time series of count proportions (Alpha vs. Delta):

# In[8]:


# We skip the first year or so of the pandemic when Alpha and Delta are not present
start_time = 13
total_counts = (alpha_counts + delta_counts)[start_time:]
dates = date_range[start_time:]
plt.figure(figsize=(7, 4))
plt.plot(dates, alpha_counts[start_time:] / total_counts, 
         label='Alpha')
plt.plot(dates, delta_counts[start_time:] / total_counts, 
         label='Delta')
plt.xlim(min(dates), max(dates))
plt.ylabel("Proportion", fontsize=18)
plt.xticks(rotation=90)
plt.gca().xaxis.set_major_locator(mpl.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %Y"))
plt.legend(fontsize=18)
plt.tight_layout()


# We see that at first Alpha was dominant, but then Delta started outcompeting it until Delta became dominant.
# 
# ### Model definition
# 
# Instead of attempting to model how the total number of observed sequences varies as a function of time (which depends on complex human behavior), we instead model the *proportion* of sequences at each time step that are Alpha versus Delta. In other words if we observed 8 Alpha lineages and 2 Delta lineages at a given time step, we model the proportions 80% and 20% instead of the raw counts 8 and 2. To do this we use a [Logistic Growth](https://en.wikipedia.org/wiki/Logistic_function) model with a [Multinomial](https://pytorch.org/docs/stable/distributions.html#multinomial) distribution as the likelihood.

# In[9]:


def basic_model(counts):
    T, L = counts.shape

    # Define plates over lineage and time
    lineage_plate = pyro.plate("lineages", L, dim=-1)
    time_plate = pyro.plate("time", T, dim=-2)

    # Define a growth rate (i.e. slope) and an init (i.e. intercept) for each lineage
    with lineage_plate:
        rate = pyro.sample("rate", dist.Normal(0, 1))
        init = pyro.sample("init", dist.Normal(0, 1))

    # We measure time in units of the SARS-CoV-2 generation time of 5.5 days
    time = torch.arange(float(T)) * dataset["time_step_days"] / 5.5
    
    # Assume lineages grow linearly in logit space
    logits = init + rate * time[:, None]
    
    # We use the softmax function (the multivariate generalization of the 
    # sigmoid function) to define normalized probabilities from the logits
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (T, L)
    
    # Observe counts via a multinomial likelihood.
    with time_plate:
        pyro.sample(
            "obs",
            dist.Multinomial(probs=probs.unsqueeze(-2), validate_args=False),
            obs=counts.unsqueeze(-2),
        )


# Let's look at the graphical structure of our model

# In[10]:


pyro.render_model(partial(basic_model, alpha_delta_counts))


# ### Define a helper for fitting models

# In[11]:


def fit_svi(model, lr=0.1, num_steps=1001, log_every=250):
    pyro.clear_param_store()  # clear parameters from previous runs
    pyro.set_rng_seed(20211214)
    if smoke_test:
        num_steps = 2
        
    # Define a mean field guide (i.e. variational distribution)
    guide = AutoNormal(model, init_scale=0.01)
    optim = ClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    
    # Train (i.e. do ELBO optimization) for num_steps iterations
    losses = []
    for step in range(num_steps):
        loss = svi.step()
        losses.append(loss)
        if step % log_every == 0:
            print(f"step {step: >4d} loss = {loss:0.6g}")
            
    # Plot to assess convergence.
    plt.figure(figsize=(6, 3))
    plt.plot(losses)
    plt.xlabel("SVI step", fontsize=18)
    plt.ylabel("ELBO loss", fontsize=18)
    plt.tight_layout()

    return guide


# ### Let's fit `basic_model` and inspect the results

# In[12]:


get_ipython().run_cell_magic('time', '', '# We truncate the data to the period with non-zero counts\nguide = fit_svi(partial(basic_model, alpha_delta_counts[13:]), num_steps=1501)\n')


# #### Let's inspect the posterior means of our latent parameters:

# In[13]:


for k, v in guide.median().items():
    print(k, v.data.cpu().numpy())


# As expected the Delta lineage (corresponding to index `1`) has a differential growth rate advantage with respect to the Alpha lineage (corresponding to index `0`) :

# In[14]:


print("Multiplicative advantage: {:.2f}".format(
      np.exp(guide.median()['rate'][1] - guide.median()['rate'][0])))


# This seems like it might be an overestimate. Can we get better estimates by modeling each spatial region individually?

# ## A regional model <a name="A-regional-model"/>
# 
# Instead of focusing on northeastern US states we now consider the entire global dataset and do not aggregate across regions.

# In[15]:


# First extract the data we want to use
alpha_counts = dataset['counts'][:, :, alpha_ids].sum(-1)
delta_counts = dataset['counts'][:, :, delta_ids].sum(-1)
counts = torch.stack([alpha_counts, delta_counts], dim=-1)
print("counts.shape: ", counts.shape)
print(f"number of regions: {counts.size(1)}")


# In[16]:


# We skip the first year or so of the pandemic when Alpha and Delta are not present
start_time = 13
total_counts = (alpha_counts + delta_counts)[start_time:]
dates = date_range[start_time:]
plt.figure(figsize=(7, 4))
plt.plot(dates, delta_counts[start_time:] / total_counts, color="C1", lw=1, alpha=0.5)
plt.xlim(min(dates), max(dates))
plt.ylabel("Proportion", fontsize=18)
plt.xticks(rotation=90)
plt.gca().xaxis.set_major_locator(mpl.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %Y"))
plt.title(f"Delta/(Alpha+Delta) in {counts.size(1)} regions", fontsize=18)
plt.tight_layout()


# In[17]:


# Model lineage proportions in each region as multivariate logistic growth
def regional_model(counts):
    T, R, L = counts.shape
        
    # Now we also define a region plate in addition to the time/lineage plates
    lineage_plate = pyro.plate("lineages", L, dim=-1)
    region_plate = pyro.plate("region", R, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)

    # We use the same growth rate (i.e. slope) for each region
    with lineage_plate:
        rate = pyro.sample("rate", dist.Normal(0, 1))
        
    # We allow the init to vary from region to region
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
    with region_plate, lineage_plate:
        init = pyro.sample("init", dist.Normal(0, init_scale))

    # We measure time in units of the SARS-CoV-2 generation time of 5.5 days
    time = torch.arange(float(T)) * dataset["time_step_days"] / 5.5

    # Instead of using the softmax function we directly use the 
    # logits parameterization of the Multinomial distribution
    logits = init + rate * time[:, None, None]
    
    # Observe sequences via a multinomial likelihood.
    with time_plate, region_plate:
        pyro.sample(
            "obs",
            dist.Multinomial(logits=logits.unsqueeze(-2), validate_args=False),
            obs=counts.unsqueeze(-2),
        )


# In[18]:


pyro.render_model(partial(regional_model, counts))


# In[19]:


get_ipython().run_cell_magic('time', '', 'guide = fit_svi(partial(regional_model, counts), num_steps=3001)\n')


# In[20]:


print("Multiplicative advantage: {:.2f}".format(
      np.exp(guide.median()['rate'][1] - guide.median()['rate'][0])))


# Notice this is a lower estimate than the previous global estimate.

# ## An alternative regional model <a name="An-alternative-regional-model" />
# 
# The regional model we defined above assumed that the `rate` for each lineage did not vary between regions. Here we add additional hierarchical structure and allow the rate to vary from region to region.

# In[21]:


def regional_model2(counts):
    T, R, L = counts.shape
    
    lineage_plate = pyro.plate("lineages", L, dim=-1)
    region_plate = pyro.plate("region", R, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)

    # We assume the init can vary a lot from region to region but
    # that the rate varies considerably less.
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
    
    # As before each lineage has a latent growth rate
    with lineage_plate:
        rate_loc = pyro.sample("rate_loc", dist.Normal(0, 1))
        
    # We allow the rate and init to vary from region to region
    with region_plate, lineage_plate:
        # The per-region per-lineage rate is governed by a hierarchical prior
        rate = pyro.sample("rate",  dist.Normal(rate_loc, rate_scale))
        init = pyro.sample("init", dist.Normal(0, init_scale))

    # We measure time in units of the SARS-CoV-2 generation time of 5.5 days
    time = torch.arange(float(T)) * dataset["time_step_days"] / 5.5
    logits = init + rate * time[:, None, None]
    
    # Observe sequences via a multinomial likelihood.
    with time_plate, region_plate:
        pyro.sample(
            "obs",
            dist.Multinomial(logits=logits.unsqueeze(-2), validate_args=False),
            obs=counts.unsqueeze(-2),
        )


# In[22]:


pyro.render_model(partial(regional_model2, counts))


# In[23]:


get_ipython().run_cell_magic('time', '', 'guide = fit_svi(partial(regional_model2, counts), num_steps=3001)\n')


# In[24]:


print("Multiplicative advantage: {:.2f}".format(
      (guide.median()['rate_loc'][1] - guide.median()['rate_loc'][0]).exp()))


# ## Generalizations <a name="Generalizations" />
# 
# So far we've seen how to model two variants at a time either globally or split across multiple regions, and how to use [pyro.plate](https://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.plate) to model multiple variants or regions or times.
# 
# What other models can you think of that might make epidemiological sense? Here are some ideas:
# 
# - Can you create a model over more than two variants, or even over all PANGO lineages?
# - What variables should be shared across lineages, across regions, or over time?
# - How might you deal with changes of behavior over time, e.g. pandemic waves or vaccination?
# 
# For an example of a larger Pyro model using SARS-CoV-2 lineage data like this, see our paper "Analysis of 2.1 million SARS-CoV-2 genomes identifies mutations associated with transmissibility" ([preprint](https://www.medrxiv.org/content/10.1101/2021.09.07.21263228v1) | [code](https://github.com/broadinstitute/pyro-cov)), and also the [Bayesian workflow tutuorial](workflow.html) using a slightly smaller dataset.

# In[ ]:




