from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.prob_models import DefinePrior, NUTSKernel, RunMCMC, PlotPosterior, DefineLikelihood

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = DefineLikelihood()
    c_1 = DefinePrior()
    c_2 = RunMCMC()
    c_3 = NUTSKernel()
    c_4 = PlotPosterior()
    c_0.mean = c_1.prior_function
    c_0.std.value = 5
    c_1.mean.value = 2
    c_1.std.value = 8
    c_2.kernel = c_3.kernel
    c_2.kernel = c_3.kernel
    c_2.num_samples.value = 500
    c_2.warmup_steps.value = 100
    c_2.data.value = [0.5, -0.2, 0.3]
    c_3.model_function = c_0.likelihood_function
    c_3.model_function = c_0.likelihood_function
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_4.posterior_samples = c_2.posterior_samples
    c_0.next = c_3
    c_1.next = c_0
    c_2.next = c_4
    c_3.next = c_2
    c_4.next = None
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')