from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.prob_models import PlotPosterior, NUTSKernel, DefinePrior, DefineLikelihood, RunMCMC
from xai_components.xai_utils.utils import CaptureScreenshot

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = DefineLikelihood()
    c_1 = DefinePrior()
    c_2 = NUTSKernel()
    c_3 = RunMCMC()
    c_4 = PlotPosterior()
    c_5 = CaptureScreenshot()
    c_0.mean = c_1.prior_function
    c_0.std.value = 5
    c_1.mean.value = 2
    c_1.std.value = 8
    c_2.model_function = c_0.likelihood_function
    c_2.model_function = c_0.likelihood_function
    c_3.kernel = c_2.kernel
    c_3.kernel = c_2.kernel
    c_3.num_samples.value = 500
    c_3.warmup_steps.value = 100
    c_3.data.value = [0.5, -0.2, 0.3]
    c_4.posterior_samples = c_3.posterior_samples
    c_4.posterior_samples = c_3.posterior_samples
    c_4.posterior_samples = c_3.posterior_samples
    c_4.posterior_samples = c_3.posterior_samples
    c_5.filename.value = 'graph.png'
    c_0.next = c_2
    c_1.next = c_0
    c_2.next = c_3
    c_3.next = c_5
    c_4.next = None
    c_5.next = c_4
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')