from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotTrace, PlotDensity, VisualizeModelGraph, PlotPrior, ArvizObject
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModel
from xai_components.xai_probability_distributions.probabilistic_distributions import NormalSampler
from xai_components.xai_pyro.probabilistic_node import FullInference

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = PyroModel()
    c_1 = NormalSampler()
    c_2 = NormalSampler()
    c_3 = VisualizeModelGraph()
    c_4 = FullInference()
    c_5 = ArvizObject()
    c_6 = PlotDensity()
    c_7 = PlotTrace()
    c_8 = PlotPrior()
    c_0.arg1 = c_1.sample
    c_1.name.value = 'Likelihood\n'
    c_1.mean = c_2.sample
    c_1.std.value = 0.9
    c_1.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_2.name.value = 'Likelihood mean\n'
    c_2.mean.value = 2.07
    c_2.std.value = 0.08
    c_3.model_function = c_0.model
    c_4.model = c_0.model
    c_4.num_samples.value = 1000
    c_5.mcmc = c_4.mcmc
    c_5.prior_predictive_values = c_4.prior_predictive
    c_5.posterior_predictive_values = c_4.posterior_predictive
    c_6.az_data = c_5.az_data
    c_7.az_data = c_5.az_data
    c_8.az_data = c_5.az_data
    c_0.next = c_3
    c_1.next = c_0
    c_2.next = c_1
    c_3.next = c_4
    c_4.next = c_5
    c_5.next = c_8
    c_6.next = None
    c_7.next = c_6
    c_8.next = c_7
    next_component = c_2
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')