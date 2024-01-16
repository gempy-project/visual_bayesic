from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import Sample, PriorPredictive, PyroModel
from xai_components.xai_pyro.probabilistic_plot import PlotTrace, ArvizObject

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = PyroModel()
    c_1 = Normal()
    c_2 = Sample()
    c_3 = PriorPredictive()
    c_4 = ArvizObject()
    c_5 = PlotTrace()
    c_6 = Sample()
    c_7 = Normal()
    c_0.fns = c_2.sample
    c_1.mean.value = 2.07
    c_1.std.value = 0.07
    c_2.fn = c_1.fn
    c_2.fn = c_1.fn
    c_3.num_samples.value = 20
    c_3.fn = c_0.model
    c_3.args = c_6.sample
    c_4.prior_predictive_values = c_3.prior
    c_4.prior_predictive_values = c_3.prior
    c_5.az_data = c_4.az_data
    c_5.az_data = c_4.az_data
    c_6.fn = c_7.fn
    c_6.fn = c_7.fn
    c_6.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_7.mean = c_2.sample
    c_7.std.value = 2.0
    c_0.next = c_3
    c_1.next = c_2
    c_2.next = c_7
    c_3.next = c_4
    c_4.next = c_5
    c_5.next = None
    c_6.next = c_0
    c_7.next = c_6
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')