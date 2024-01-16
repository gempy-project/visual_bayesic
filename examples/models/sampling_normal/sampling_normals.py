from argparse import ArgumentParser
from xai_components.xai_plotting.dist_plot import VisualizeNormalDistribution
from xai_components.xai_obsolete.normal_distribution_pyro import NormalDistributionPyro

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = NormalDistributionPyro()
    c_1 = VisualizeNormalDistribution()
    c_0.loc.value = 20.0
    c_0.scale.value = 1
    c_0.population.value = 100
    c_1.samples = c_0.samples
    c_1.samples = c_0.samples
    c_0.next = c_1
    c_1.next = None
    next_component = c_0
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')