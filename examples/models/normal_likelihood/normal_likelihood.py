from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.prob_models import NUTSKernel, VisualizeModelGraph, PlotPosterior, DefineLikelihood, RunMCMC, DefinePrior
from xai_components.xai_utils.utils import CaptureScreenshot

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = VisualizeModelGraph()
    c_1 = DefineLikelihood()
    c_2 = NUTSKernel()
    c_3 = DefinePrior()
    c_4 = RunMCMC()
    c_5 = PlotPosterior()
    c_6 = CaptureScreenshot()
    c_0.model_function = c_1.likelihood_function
    c_0.model_function = c_1.likelihood_function
    c_0.model_function = c_1.likelihood_function
    c_0.model_function = c_1.likelihood_function
    c_1.mean = c_3.prior_function
    c_1.std.value = 5
    c_2.model_function = c_1.likelihood_function
    c_2.model_function = c_1.likelihood_function
    c_3.mean.value = 2
    c_3.std.value = 8
    c_4.kernel = c_2.kernel
    c_4.kernel = c_2.kernel
    c_4.num_samples.value = 500
    c_4.warmup_steps.value = 100
    c_4.data.value = [0.5, -0.2, 0.3]
    c_5.posterior_samples = c_4.posterior_samples
    c_5.posterior_samples = c_4.posterior_samples
    c_5.posterior_samples = c_4.posterior_samples
    c_5.posterior_samples = c_4.posterior_samples
    c_6.filename.value = 'screenshot.png'
    c_0.next = c_2
    c_1.next = c_0
    c_2.next = c_4
    c_3.next = c_1
    c_4.next = c_6
    c_5.next = None
    c_6.next = c_5
    next_component = c_3
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')