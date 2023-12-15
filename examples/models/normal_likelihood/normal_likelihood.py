from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_pyro.prob_models import PlotPosterior, DefinePrior, NUTSKernel, DefineLikelihood, RunMCMC, VisualizeModelGraph
from xai_components.xai_pyro.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import Sample
from xai_components.xai_utils.utils import CaptureScreenshot

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = DefinePrior()
    c_1 = Sample()
    c_2 = DefineLikelihood()
    c_3 = VisualizeModelGraph()
    c_4 = Normal()
    c_5 = NUTSKernel()
    c_6 = RunMCMC()
    c_7 = PlotPosterior()
    c_8 = CaptureScreenshot()
    c_0.mean.value = 2
    c_0.std.value = 8
    c_1.fn = c_4.fn
    c_1.fn = c_4.fn
    c_2.mean = c_1.sample
    c_2.std.value = 5
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_3.model_function = c_2.likelihood_function
    c_4.mean.value = 2
    c_4.std.value = 8
    c_5.model_function = c_2.likelihood_function
    c_5.model_function = c_2.likelihood_function
    c_6.kernel = c_5.kernel
    c_6.kernel = c_5.kernel
    c_6.num_samples.value = 500
    c_6.warmup_steps.value = 100
    c_6.data.value = [0.5, -0.2, 0.3]
    c_7.posterior_samples = c_6.posterior_samples
    c_7.posterior_samples = c_6.posterior_samples
    c_7.posterior_samples = c_6.posterior_samples
    c_7.posterior_samples = c_6.posterior_samples
    c_8.filename.value = 'screenshot.png'
    c_0.next = None
    c_1.next = c_2
    c_2.next = c_3
    c_3.next = c_5
    c_4.next = c_1
    c_5.next = c_6
    c_6.next = c_8
    c_7.next = None
    c_8.next = c_7
    next_component = c_4
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')