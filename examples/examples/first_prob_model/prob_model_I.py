from argparse import ArgumentParser
from xai_components.xai_pyro.prob_models import RunMCMC, DefineLikelihood, NUTSKernel, PlotPosterior, DefinePrior

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = RunMCMC()
    c_1 = DefineLikelihood()
    c_2 = DefinePrior()
    c_3 = NUTSKernel()
    c_4 = PlotPosterior()
    c_0.kernel = c_3.kernel
    c_0.kernel = c_3.kernel
    c_0.num_samples.value = 500
    c_0.warmup_steps.value = 100
    c_0.data.value = [0.5, -0.2, 0.3]
    c_1.mean = c_2.prior_function
    c_1.std.value = 5
    c_2.mean.value = 2
    c_2.std.value = 8
    c_3.model_function = c_1.likelihood_function
    c_3.model_function = c_1.likelihood_function
    c_4.posterior_samples = c_0.posterior_samples
    c_4.posterior_samples = c_0.posterior_samples
    c_4.posterior_samples = c_0.posterior_samples
    c_4.posterior_samples = c_0.posterior_samples
    c_0.next = c_4
    c_1.next = c_3
    c_2.next = c_1
    c_3.next = c_0
    c_4.next = None
    next_component = c_2
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')