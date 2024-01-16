from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import PriorPredictive, Sample, PyroModel
from xai_components.xai_pyro.probabilistic_plot import PlotTrace, ArvizObject

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Normal()
    c_1 = Sample()
    c_2 = PriorPredictive()
    c_3 = Normal()
    c_4 = PyroModel()
    c_5 = ArvizObject()
    c_6 = PlotTrace()
    c_7 = Sample()
    c_0.mean = c_1.sample
    c_0.std.value = 2.0
    c_1.fn = c_3.fn
    c_1.fn = c_3.fn
    c_2.num_samples.value = 20
    c_2.fn = c_4.model
    c_2.args = c_7.sample
    c_3.mean.value = 2.07
    c_3.std.value = 0.07
    c_4.fns = c_1.sample
    c_5.prior_predictive_values = c_2.prior
    c_5.prior_predictive_values = c_2.prior
    c_6.az_data = c_5.az_data
    c_6.az_data = c_5.az_data
    c_7.name.value = 'Likelihood\n'
    c_7.fn = c_0.fn
    c_7.fn = c_0.fn
    c_7.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_0.next = c_7
    c_1.next = c_0
    c_2.next = c_5
    c_3.next = c_1
    c_4.next = c_2
    c_5.next = c_6
    c_6.next = None
    c_7.next = c_4
    next_component = c_3
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')