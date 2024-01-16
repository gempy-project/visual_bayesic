from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import PyroModel, Sample, PriorPredictive
from xai_components.xai_pyro.probabilistic_plot import PlotPrior, ArvizObject

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Sample()
    c_1 = Sample()
    c_2 = Normal()
    c_3 = Normal()
    c_4 = PyroModel()
    c_5 = PriorPredictive()
    c_6 = ArvizObject()
    c_7 = PlotPrior()
    c_0.fn = c_2.fn
    c_0.fn = c_2.fn
    c_1.name.value = 'Likelihood\n'
    c_1.fn = c_3.fn
    c_1.fn = c_3.fn
    c_1.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_2.mean.value = 2.07
    c_2.std.value = 0.07
    c_3.mean = c_0.sample
    c_3.std.value = 2.0
    c_4.arg1 = c_1.sample
    c_5.num_samples.value = 20
    c_5.fn = c_4.model
    c_6.prior_predictive_values = c_5.prior
    c_6.prior_predictive_values = c_5.prior
    c_6.prior_predictive_values = c_5.prior
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_7.az_data = c_6.az_data
    c_0.next = c_3
    c_1.next = c_4
    c_2.next = c_0
    c_3.next = c_1
    c_4.next = c_5
    c_5.next = c_6
    c_6.next = c_7
    c_7.next = None
    next_component = c_2
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')