from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_controlflow.branches import BranchComponent
from xai_components.xai_obsolete.prob_models import VisualizeModelGraph
from xai_components.xai_plotting.probabilistic_plot import PlotNormalLikelihoodJoy, PlotNormalLikelihood, PlotMarginals, ArvizObject, PlotPrior
from xai_components.xai_probabilistic_models.probabilistic_models_I import PyroModel
from xai_components.xai_probability_distributions.probabilistic_distributions import Normal, Uniform, Gamma
from xai_components.xai_pyro.probabilistic_node import Sample, PriorPredictive, PosteriorPredictive, MCMC, NUTS, RunMCMC

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = RunMCMC()
    c_1 = NUTS()
    c_2 = Sample()
    c_3 = Gamma()
    c_4 = Normal()
    c_5 = Sample()
    c_6 = BranchComponent()
    c_7 = Sample()
    c_8 = PyroModel()
    c_9 = VisualizeModelGraph()
    c_10 = PriorPredictive()
    c_11 = ArvizObject()
    c_12 = PlotPrior()
    c_13 = PlotNormalLikelihood()
    c_14 = PlotNormalLikelihoodJoy()
    c_15 = PlotMarginals()
    c_16 = PosteriorPredictive()
    c_17 = Uniform()
    c_18 = Normal()
    c_19 = Sample()
    c_20 = MCMC()
    c_0.mcmc = c_20.mcmc
    c_1.model = c_8.model
    c_2.name.value = '$\\sigma$'
    c_2.fn = c_3.fn
    c_2.fn = c_3.fn
    c_3.concentration.value = 0.3
    c_3.rate.value = 3.01
    c_4.mean.value = 2.07
    c_4.std.value = 0.07
    c_5.name.value = '$\\mu$'
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_5.fn = c_4.fn
    c_6.condition.value = False
    c_7.name.value = '$y$'
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.fn = c_18.fn
    c_7.obs.value = [2.12, 2.06, 2.08, 2.05, 2.08, 2.09, 2.19, 2.07, 2.16, 2.11, 2.13, 1.92]
    c_8.arg1 = c_7.sample
    c_9.model_function = c_8.model
    c_10.model = c_8.model
    c_10.num_samples.value = 20
    c_11.mcmc = c_20.mcmc
    c_11.prior_predictive_values = c_10.prior
    c_11.posterior_predictive_values = c_16.posterior_predictive
    c_12.az_data = c_11.az_data
    c_12.az_data = c_11.az_data
    c_13.az_data = c_11.az_data
    c_13.mean_sample_name.value = '$\\mu$'
    c_13.std_sample_name.value = '$\\sigma$'
    c_13.y_sample_name.value = '$y$'
    c_14.az_data = c_11.az_data
    c_14.mean_sample_name.value = '$\\mu$'
    c_14.std_sample_name.value = '$\\sigma$'
    c_14.y_sample_name.value = '$y$'
    c_15.az_data = c_11.az_data
    c_15.sample_1_name.value = '$\\mu$'
    c_15.sample_2_name.value = '$\\sigma$'
    c_16.model = c_8.model
    c_16.MCMC = c_20.mcmc
    c_16.num_samples.value = 20
    c_17.low.value = 0.0
    c_17.high.value = 10.0
    c_18.mean = c_19.sample
    c_18.std = c_2.sample
    c_19.name.value = '$\\mu$'
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_19.fn = c_17.fn
    c_20.NUTS = c_1.NUTS
    c_20.NUTS = c_1.NUTS
    c_20.num_samples.value = 50
    c_20.num_chains.value = 1
    c_0.next = c_16
    c_1.next = c_20
    c_2.next = c_18
    c_3.next = c_2
    c_4.next = c_5
    c_5.next = None
    c_6.next = c_3
    c_6.when_true = SubGraphExecutor(c_4)
    c_6.when_false = SubGraphExecutor(c_17)
    c_7.next = c_8
    c_8.next = c_9
    c_9.next = c_10
    c_10.next = c_1
    c_11.next = c_12
    c_12.next = c_13
    c_13.next = c_14
    c_14.next = c_15
    c_15.next = None
    c_16.next = c_11
    c_17.next = c_19
    c_18.next = c_7
    c_19.next = None
    c_20.next = c_0
    next_component = c_6
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')