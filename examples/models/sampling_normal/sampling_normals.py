from argparse import ArgumentParser
from xai_components.xai_plotting import VisualizeNormalDistribution
from xai_components import NormalDistributionPyro
from xai_components import NormalDistributionTF

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = VisualizeNormalDistribution()
    c_1 = NormalDistributionTF()
    c_2 = NormalDistributionPyro()
    c_0.samples = c_1.samples
    c_1.loc.value = 10
    c_1.scale.value = 1
    c_1.population.value = 100
    c_2.loc.value = 0.1
    c_2.scale.value = 1
    c_2.population.value = 100
    c_0.next = None
    c_1.next = c_0
    c_2.next = None
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')