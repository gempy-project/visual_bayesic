"""
Getting started with Visual Bayesic
===================================

Check out the tutorial.xircuit file to see the visual representation of the code below.
from argparse import ArgumentParser
"""

from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotDensity, PlotNormalLikelihoodJoy, PlotPrior, ArvizObject, PlotTrace, VisualizeModelGraph
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModel
from xai_components.xai_probability_distributions.probabilistic_distributions import GammaSampler, NormalSampler
from xai_components.xai_pyro.probabilistic_node import FullInference

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = VisualizeModelGraph()
    c_1 = NormalSampler()
    c_2 = NormalSampler()
    c_3 = FullInference()
    c_4 = ArvizObject()
    c_5 = PlotNormalLikelihoodJoy()
    c_6 = GammaSampler()
    c_7 = PlotDensity()
    c_8 = PlotTrace()
    c_9 = PlotPrior()
    c_10 = PyroModel()
    c_0.model_function = c_10.model
    c_0.model_function = c_10.model
    c_0.model_function = c_10.model
    c_1.name.value = 'Likelihood\n'
    c_1.mean = c_2.sample
    c_1.std = c_6.sample
    c_1.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_2.name.value = 'Likelihood mean\n'
    c_2.mean.value = 2.07
    c_2.std.value = 0.08
    c_3.model = c_10.model
    c_3.num_samples.value = 1000
    c_4.mcmc = c_3.mcmc
    c_4.prior_predictive_values = c_3.prior_predictive
    c_4.posterior_predictive_values = c_3.posterior_predictive
    c_5.az_data = c_4.az_data
    c_5.mean_sample_name.value = 'Likelihood mean\n'
    c_5.std_sample_name.value = 'Likelihood std'
    c_5.y_sample_name.value = 'Likelihood\n'
    c_5.n_samples.value = 19
    c_6.name.value = 'Likelihood std'
    c_6.concentration.value = 3.3
    c_6.rate.value = 1.2
    c_7.az_data = c_4.az_data
    c_8.az_data = c_4.az_data
    c_9.az_data = c_4.az_data
    c_10.arg1 = c_1.sample
    c_0.next = c_3
    c_1.next = c_10
    c_2.next = c_6
    c_3.next = c_4
    c_4.next = c_9
    c_5.next = None
    c_6.next = c_1
    c_7.next = c_5
    c_8.next = c_7
    c_9.next = c_8
    c_10.next = c_0
    next_component = c_2
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')