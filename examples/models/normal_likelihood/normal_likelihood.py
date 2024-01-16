from argparse import ArgumentParser
from xai_components.xai_obsolete.prob_models import DefinePrior
from xai_components.xai_probability_distributions.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import PriorPredictive, Sample

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = DefinePrior()
    c_1 = Sample()
    c_2 = Normal()
    c_3 = Sample()
    c_4 = Normal()
    c_5 = PriorPredictive()
    c_0.mean.value = 2
    c_0.std.value = 8
    c_1.fn = c_2.fn
    c_1.fn = c_2.fn
    c_1.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_2.mean = c_3.sample
    c_2.std.value = 3.0
    c_3.fn = c_4.fn
    c_3.fn = c_4.fn
    c_4.mean.value = 2
    c_4.std.value = 8
    c_5.num_samples.value = 40
    c_5.fn = c_1.sample
    c_0.next = None
    c_1.next = c_5
    c_2.next = c_1
    c_3.next = c_2
    c_4.next = c_3
    c_5.next = None
    next_component = c_4
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')