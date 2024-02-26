from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotDensity, VisualizeModelGraph, PlotNormalLikelihoodJoy, PlotPrior, PlotTrace, ArvizObject
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModel
from xai_components.xai_probability_distributions.probabilistic_distributions import NormalSampler
from xai_components.xai_pyro.probabilistic_node import FullInference

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = PyroModel()
    c_1 = NormalSampler()
    c_2 = FullInference()
    c_3 = ArvizObject()
    c_4 = PlotNormalLikelihoodJoy()
    c_5 = PlotDensity()
    c_6 = PlotTrace()
    c_7 = PlotPrior()
    c_8 = NormalSampler()
    c_9 = VisualizeModelGraph()
    c_0.arg1 = c_8.sample
    c_1.name.value = 'Likelihood mean\n'
    c_1.mean.value = 2.07
    c_1.std.value = 0.08
    c_2.model = c_0.model
    c_2.num_samples.value = 1000
    c_3.mcmc = c_2.mcmc
    c_3.prior_predictive_values = c_2.prior_predictive
    c_3.posterior_predictive_values = c_2.posterior_predictive
    c_4.az_data = c_3.az_data
    c_5.az_data = c_3.az_data
    c_6.az_data = c_3.az_data
    c_7.az_data = c_3.az_data
    c_8.name.value = 'Likelihood\n'
    c_8.mean = c_1.sample
    c_8.std.value = 0.9
    c_8.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_9.model_function = c_0.model
    c_9.model_function = c_0.model
    c_9.model_function = c_0.model
    c_0.next = c_9
    c_1.next = c_8
    c_2.next = c_3
    c_3.next = c_7
    c_4.next = None
    c_5.next = c_4
    c_6.next = c_5
    c_7.next = c_6
    c_8.next = c_0
    c_9.next = c_2
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')