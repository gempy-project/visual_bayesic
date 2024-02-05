from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotNormalLikelihoodJoy, PlotNormalLikelihood, PlotPrior, PlotMarginals, ArvizObject
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModel
from xai_components.xai_probability_distributions.probabilistic_distributions import Normal, Gamma
from xai_components.xai_pyro.probabilistic_node import Sample, PriorPredictive, NUTS, RunMCMC, MCMC, PosteriorPredictive

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Sample()
    c_1 = Gamma()
    c_2 = Normal()
    c_3 = PyroModel()
    c_4 = NUTS()
    c_5 = MCMC()
    c_6 = RunMCMC()
    c_7 = PosteriorPredictive()
    c_8 = ArvizObject()
    c_9 = PlotPrior()
    c_10 = PriorPredictive()
    c_11 = PlotNormalLikelihood()
    c_12 = PlotNormalLikelihoodJoy()
    c_13 = PlotMarginals()
    c_14 = Sample()
    c_15 = Sample()
    c_16 = Normal()
    c_0.name.value = '$\\sigma$'
    c_0.fn = c_1.fn
    c_0.fn = c_1.fn
    c_1.concentration.value = 0.3
    c_1.rate.value = 3.01
    c_2.mean.value = 2.07
    c_2.std.value = 0.07
    c_3.arg1 = c_15.sample
    c_4.model = c_3.model
    c_5.NUTS = c_4.NUTS
    c_5.NUTS = c_4.NUTS
    c_5.num_samples.value = 50
    c_5.num_chains.value = 1
    c_6.mcmc = c_5.mcmc
    c_7.model = c_3.model
    c_7.MCMC = c_5.mcmc
    c_7.num_samples.value = 20
    c_8.mcmc = c_5.mcmc
    c_8.prior_predictive_values = c_10.prior
    c_8.posterior_predictive_values = c_7.posterior_predictive
    c_9.az_data = c_8.az_data
    c_9.az_data = c_8.az_data
    c_10.model = c_3.model
    c_10.num_samples.value = 20
    c_11.az_data = c_8.az_data
    c_11.mean_sample_name.value = '$\\mu$'
    c_11.std_sample_name.value = '$\\sigma$'
    c_11.y_sample_name.value = '$y$'
    c_12.az_data = c_8.az_data
    c_12.mean_sample_name.value = '$\\mu$'
    c_12.std_sample_name.value = '$\\sigma$'
    c_12.y_sample_name.value = '$y$'
    c_13.az_data = c_8.az_data
    c_13.sample_1_name.value = '$\\mu$'
    c_13.sample_2_name.value = '$\\sigma$'
    c_14.name.value = '$\\mu$'
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_14.fn = c_2.fn
    c_15.name.value = '$y$'
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.fn = c_16.fn
    c_15.obs.value = [2.12, 2.06, 2.08, 2.05, 2.08, 2.09, 2.19, 2.07, 2.16, 2.11, 2.13, 1.92]
    c_16.mean = c_14.sample
    c_16.std = c_0.sample
    c_0.next = c_16
    c_1.next = c_0
    c_2.next = c_14
    c_3.next = c_10
    c_4.next = c_5
    c_5.next = c_6
    c_6.next = c_7
    c_7.next = c_8
    c_8.next = c_9
    c_9.next = c_11
    c_10.next = c_4
    c_11.next = c_12
    c_12.next = c_13
    c_13.next = None
    c_14.next = c_1
    c_15.next = c_3
    c_16.next = c_15
    next_component = c_2
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')