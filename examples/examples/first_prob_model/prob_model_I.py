from argparse import ArgumentParser
from visual_bayesic.xai_components import DefinePrior, DefineLikelihood, NUTSKernel, RunMCMC, PlotPosterior


def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = DefineLikelihood()
    c_1 = NUTSKernel()
    c_2 = RunMCMC()
    c_3 = DefinePrior()
    c_4 = PlotPosterior()
    c_0.mean = c_3.prior_function
    c_0.std.value = 5
    c_1.model_function = c_0.likelihood_function
    c_1.model_function = c_0.likelihood_function
    c_2.kernel = c_1.kernel
    c_2.kernel = c_1.kernel
    c_2.num_samples.value = 500
    c_2.warmup_steps.value = 100
    c_2.data.value = [0.5, -0.2, 0.3]
    c_3.mean.value = 2
    c_3.std.value = 8
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_0.next = c_1
    c_1.next = c_2
    c_2.next = c_4
    c_3.next = c_0
    c_4.next = None
    next_component = c_3
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')