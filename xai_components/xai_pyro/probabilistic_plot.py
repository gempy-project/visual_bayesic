import arviz as az
from matplotlib import pyplot as plt
from xai_components.base import InArg, Component, xai_component, OutArg


@xai_component
class ArvizObject(Component):
    prior_predictive_values: InArg[dict]
    az_data: OutArg[az.InferenceData]
  
    def execute(self, ctx) -> None:
        az_data = az.from_pyro(
            prior=self.prior_predictive_values.value,
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
        self.plot.value = az.plot_trace(self.az_data.value.prior)
        plt.show()
