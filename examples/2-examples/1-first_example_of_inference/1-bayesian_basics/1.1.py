from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotMarginals, ArvizObject, PlotNormalLikelihood, PlotPrior, PlotNormalLikelihoodJoy
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModelSampleOneRandomVariable
from xai_components.xai_probability_distributions.probabilistic_distributions import Normal, Gamma
from xai_components.xai_pyro.probabilistic_node import MCMC, Sample, PosteriorPredictive, PriorPredictive, RunMCMC, NUTS

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Sample()
    c_1 = Normal()
    c_2 = PyroModelSampleOneRandomVariable()
    c_3 = PosteriorPredictive()
    c_4 = ArvizObject()
    c_5 = PlotMarginals()
    c_6 = NUTS()
    c_7 = MCMC()
    c_8 = RunMCMC()
    c_9 = Gamma()
    c_10 = PlotNormalLikelihoodJoy()
    c_11 = PlotNormalLikelihood()
    c_12 = PlotPrior()
    c_13 = PriorPredictive()
    c_14 = Sample()
    c_15 = Normal()
    c_16 = Sample()
    c_0.name.value = '$\\sigma$'
    c_0.fn = c_9.fn
    c_1.mean.value = 2.07
    c_1.std.value = 0.07
    c_2.arg1 = c_16.sample
    c_3.model = c_2.model
    c_3.MCMC = c_7.mcmc
    c_3.num_samples.value = 20
    c_4.mcmc = c_7.mcmc
    c_4.prior_predictive_values = c_13.prior
    c_4.prior_predictive_values = c_13.prior
    c_4.prior_predictive_values = c_13.prior
    c_4.posterior_predictive_values = c_3.posterior_predictive
    c_5.az_data = c_4.az_data
    c_5.sample_1_name.value = '$\\mu$'
    c_5.sample_2_name.value = '$\\sigma$'
    c_6.model = c_2.model
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.NUTS = c_6.NUTS
    c_7.num_samples.value = 50
    c_7.num_chains.value = 1
    c_8.mcmc = c_7.mcmc
    c_8.mcmc = c_7.mcmc
    c_8.mcmc = c_7.mcmc
    c_8.mcmc = c_7.mcmc
    c_8.mcmc = c_7.mcmc
    c_8.mcmc = c_7.mcmc
    c_9.concentration.value = 0.3
    c_9.rate.value = 3.01
    c_10.az_data = c_4.az_data
    c_10.mean_sample_name.value = '$\\mu$'
    c_10.std_sample_name.value = '$\\sigma$'
    c_10.y_sample_name.value = '$y$'
    c_11.az_data = c_4.az_data
    c_11.mean_sample_name.value = '$\\mu$'
    c_11.std_sample_name.value = '$\\sigma$'
    c_11.y_sample_name.value = '$y$'
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_12.az_data = c_4.az_data
    c_13.model = c_2.model
    c_13.num_samples.value = 20
    c_14.name.value = '$\\mu$'
    c_14.fn = c_1.fn
    c_14.fn = c_1.fn
    c_14.fn = c_1.fn
    c_15.mean = c_14.sample
    c_15.std = c_0.sample
    c_16.name.value = '$y$'
    c_16.fn = c_15.fn
    c_16.fn = c_15.fn
    c_16.fn = c_15.fn
    c_16.fn = c_15.fn
    c_16.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_0.next = c_15
    c_1.next = c_14
    c_2.next = c_13
    c_3.next = c_4
    c_4.next = c_12
    c_5.next = None
    c_6.next = c_7
    c_7.next = c_8
    c_8.next = c_3
    c_9.next = c_0
    c_10.next = c_5
    c_11.next = c_10
    c_12.next = c_11
    c_13.next = c_6
    c_14.next = c_9
    c_15.next = c_16
    c_16.next = c_2
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')