from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_plotting.probabilistic_plot import PlotPrior, ArvizObject
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModelSampleOneRandomVariable
from xai_components.xai_probability_distributions.probabilistic_distributions import Normal
from xai_components.xai_pyro.probabilistic_node import Sample, RunMCMC, MCMC, PriorPredictive, PosteriorPredictive, NUTS

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = Sample()
    c_1 = PyroModelSampleOneRandomVariable()
    c_2 = PriorPredictive()
    c_3 = ArvizObject()
    c_4 = PlotPrior()
    c_5 = PosteriorPredictive()
    c_6 = MCMC()
    c_7 = RunMCMC()
    c_8 = NUTS()
    c_9 = Sample()
    c_10 = Normal()
    c_11 = Normal()
    c_0.name.value = 'Prior Normal Distribution '
    c_0.fn = c_10.fn
    c_0.fn = c_10.fn
    c_0.fn = c_10.fn
    c_1.arg1 = c_9.sample
    c_2.model = c_1.model
    c_2.model = c_1.model
    c_2.num_samples.value = 20
    c_3.mcmc = c_6.mcmc
    c_3.prior_predictive_values = c_2.prior
    c_3.prior_predictive_values = c_2.prior
    c_3.prior_predictive_values = c_2.prior
    c_3.posterior_predictive_values = c_5.posterior_predictive
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_4.az_data = c_3.az_data
    c_5.model = c_1.model
    c_5.MCMC = c_6.mcmc
    c_5.num_samples.value = 20
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.NUTS = c_8.NUTS
    c_6.num_samples.value = 50
    c_6.num_chains.value = 1
    c_7.mcmc = c_6.mcmc
    c_7.mcmc = c_6.mcmc
    c_7.mcmc = c_6.mcmc
    c_7.mcmc = c_6.mcmc
    c_7.mcmc = c_6.mcmc
    c_7.mcmc = c_6.mcmc
    c_8.model = c_1.model
    c_8.model = c_1.model
    c_8.model = c_1.model
    c_8.model = c_1.model
    c_8.model = c_1.model
    c_9.name.value = 'Likelihood\n'
    c_9.fn = c_11.fn
    c_9.fn = c_11.fn
    c_9.fn = c_11.fn
    c_9.fn = c_11.fn
    c_9.obs.value = [2.12, 2.06, 2.08, 2.05]
    c_10.mean.value = 2.07
    c_10.std.value = 0.07
    c_11.mean = c_0.sample
    c_11.std.value = 2.0
    c_0.next = c_11
    c_1.next = c_2
    c_2.next = c_8
    c_3.next = c_4
    c_4.next = None
    c_5.next = c_3
    c_6.next = c_7
    c_7.next = c_5
    c_8.next = c_6
    c_9.next = c_1
    c_10.next = c_0
    c_11.next = c_9
    next_component = c_10
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')