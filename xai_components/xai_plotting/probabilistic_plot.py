import arviz as az
import pyro
from matplotlib import pyplot as plt
from xai_components.base import InArg, Component, xai_component, OutArg


@xai_component
class ArvizObject(Component):
    mcmc: InArg[pyro.infer.MCMC]
    prior_predictive_values: InArg[dict]
    posterior_predictive_values: InArg[dict]
    az_data: OutArg[az.InferenceData]

    def __init__(self):
        super().__init__()
        self.mcmc = InArg.empty()
        self.prior_predictive_values = InArg.empty()
        self.posterior_predictive_values = InArg.empty()
        self.az_data = OutArg.empty()


    def execute(self, ctx) -> None:
        az_data = az.from_pyro(
            posterior=self.mcmc.value,
            prior=self.prior_predictive_values.value,
            posterior_predictive=self.posterior_predictive_values.value
        )

        self.az_data.value = az_data


@xai_component
class PlotTrace(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        self.plot.value = az.plot_trace(self.az_data.value)
        plt.show()


@xai_component
class PlotPrior(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        arviz_object = self.az_data.value
        self.plot.value = az.plot_trace(arviz_object.prior)
        plt.show()


@xai_component
class PlotLikelihood(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        # TODO
        plt.show()


@xai_component
class PlotJoy(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        # TODO
        plt.show()


@xai_component
class PlotMarginals(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        # TODO
        plt.show()
