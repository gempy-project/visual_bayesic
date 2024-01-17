from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import ArvizObject, PlotNormalLikelihood, PlotMarginals, PlotNormalLikelihoodJoy, PlotPrior
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModelSampleOneRandomVariable
from xai_components.xai_probability_distributions.probabilistic_distributions import Gamma, Normal
from xai_components.xai_pyro.probabilistic_node import MCMC, NUTS, PosteriorPredictive, RunMCMC, Sample, PriorPredictive

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Sample()
    c_1 = Normal()
    c_2 = PyroModelSampleOneRandomVariable()
    c_3 = PosteriorPredictive()
    c_4 = ArvizObject()
    c_5 = PlotMarginals()
    c_6 = RunMCMC()
    c_7 = Sample()
    c_8 = Gamma()
    c_9 = PlotNormalLikelihood()
    c_10 = PlotPrior()
    c_11 = PriorPredictive()
    c_12 = NUTS()
    c_13 = MCMC()
    c_14 = PlotNormalLikelihoodJoy()
    c_15 = Normal()
    c_16 = Sample()
    c_0.name.value = '$y$'
    c_0.fn = c_15.fn
    c_0.fn = c_15.fn
    c_0.fn = c_15.fn
    c_0.fn = c_15.fn
    c_0.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_1.mean.value = 2.07
    c_1.std.value = 0.07
    c_2.arg1 = c_0.sample
    c_3.model = c_2.model
    c_3.MCMC = c_13.mcmc
    c_3.num_samples.value = 20
    c_4.mcmc = c_13.mcmc
    c_4.prior_predictive_values = c_11.prior
    c_4.prior_predictive_values = c_11.prior
    c_4.prior_predictive_values = c_11.prior
    c_4.posterior_predictive_values = c_3.posterior_predictive
    c_5.az_data = c_4.az_data
    c_5.sample_1_name.value = '\\$mu$'
    c_5.sample_2_name.value = '\\$sigma$'
    c_6.mcmc = c_13.mcmc
    c_6.mcmc = c_13.mcmc
    c_6.mcmc = c_13.mcmc
    c_6.mcmc = c_13.mcmc
    c_6.mcmc = c_13.mcmc
    c_6.mcmc = c_13.mcmc
    c_7.name.value = '\\$sigma$'
    c_7.fn = c_8.fn
    c_8.concentration.value = 0.3
    c_8.rate.value = 3.01
    c_9.az_data = c_4.az_data
    c_9.mean_sample_name.value = '\\$mu$'
    c_9.std_sample_name.value = '\\$sigma$'
    c_9.y_sample_name.value = '$y$'
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_10.az_data = c_4.az_data
    c_11.model = c_2.model
    c_11.num_samples.value = 20
    c_12.model = c_2.model
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.NUTS = c_12.NUTS
    c_13.num_samples.value = 50
    c_13.num_chains.value = 1
    c_14.az_data = c_4.az_data
    c_14.mean_sample_name.value = '\\$mu$'
    c_14.std_sample_name.value = '\\$sigma$'
    c_14.y_sample_name.value = '$y$'
    c_15.mean = c_16.sample
    c_15.std = c_7.sample
    c_16.name.value = '\\$mu$'
    c_16.fn = c_1.fn
    c_16.fn = c_1.fn
    c_16.fn = c_1.fn
    c_0.next = c_2
    c_1.next = c_16
    c_2.next = c_11
    c_3.next = c_4
    c_4.next = c_10
    c_5.next = None
    c_6.next = c_3
    c_7.next = c_15
    c_8.next = c_7
    c_9.next = c_14
    c_10.next = c_9
    c_11.next = c_12
    c_12.next = c_13
    c_13.next = c_6
    c_14.next = c_5
    c_15.next = c_0
    c_16.next = c_8
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')