import arviz
import arviz as az
import pyro
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from xai_components.base import InArg, Component, xai_component, OutArg, InCompArg

cyanish_ = ["arviz-white", "arviz-cyanish"]
az.style.context(cyanish_, True)


@xai_component(color="#4d4141")
class ArvizObject(Component):
    mcmc: InCompArg[pyro.infer.MCMC]
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


@xai_component(color="#776f85")
class PlotTrace(Component):
    az_data: InCompArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        with az.style.context(cyanish_, after_reset=True):
            data_value: arviz.InferenceData = self.az_data.value
            self.plot.value = az.plot_trace(data_value)
            plt.show()


@xai_component(color="#776f85")
class PlotDensity(Component):
    az_data: InArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        data_value: arviz.InferenceData = self.az_data.value
        self.plot.value = az.plot_trace(data_value)

        with az.style.context(cyanish_, after_reset=True):
            axes = az.plot_density(
                [data_value, data_value.prior],
                data_labels=["Posterior", "Prior"],
                shade=0.2,
            )
        plt.show()


@xai_component(color="#70A3B3")
class PlotPrior(Component):
    az_data: InCompArg[az.InferenceData]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        arviz_object = self.az_data.value
        self.plot.value = az.plot_trace(arviz_object.prior)
        plt.show()


@xai_component(color="#DA8886")
class PlotNormalLikelihood(Component):
    az_data: InCompArg[az.InferenceData]
    mean_sample_name: InArg[str]
    std_sample_name: InArg[str]
    y_sample_name: InArg[str]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        try:
            from gempy_probability.plot_posterior import PlotPosterior
        except ImportError:
            print("You need to install gempy_probability to use this component.")
            return

        p = PlotPosterior(self.az_data.value)
        p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
        p.plot_normal_likelihood(
            mean=self.mean_sample_name.value,
            std=self.std_sample_name.value,
            obs=self.y_sample_name.value,
            iteration=-1,
        )

        # p.likelihood_axes.set_xlim(1.70, 2.40)
        p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        for tick in p.likelihood_axes.get_xticklabels():
            tick.set_rotation(45)
        plt.show()


@xai_component(color="#DA8886")
class PlotNormalLikelihoodJoy(Component):
    az_data: InCompArg[az.InferenceData]
    mean_sample_name: InArg[str]
    std_sample_name: InArg[str]
    y_sample_name: InArg[str]
    n_samples: InArg[int]
    plot: OutArg[any]

    def __init__(self):
        super().__init__()
        self.n_samples = InArg(31)

    def execute(self, ctx) -> None:
        try:
            from gempy_probability.plot_posterior import PlotPosterior
        except ImportError:
            print("You need to install gempy_probability to use this component.")
            return

        p = PlotPosterior(self.az_data.value)

        p.create_figure(figsize=(9, 9), joyplot=True, marginal=False, likelihood=False, n_samples=self.n_samples.value)
        p.plot_joy(
            var_names=(self.mean_sample_name.value, self.std_sample_name.value),
            obs=self.y_sample_name.value,
            iteration=self.n_samples.value // 2
        )
        plt.show()


@xai_component(color="#776f85")
class PlotMarginals(Component):
    az_data: InArg[az.InferenceData]
    sample_1_name: InArg[str]
    sample_2_name: InArg[str]
    plot: OutArg[any]

    def execute(self, ctx) -> None:
        try:
            from gempy_probability.plot_posterior import PlotPosterior
        except ImportError:
            print("You need to install gempy_probability to use this component.")
            return

        p = PlotPosterior(self.az_data.value)

        p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=False)
        p.plot_marginal(
            var_names=[self.sample_1_name.value, self.sample_2_name.value],
            plot_trace=False,
            credible_interval=0.95,
            kind='kde',
            joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
            joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

        plt.show()


@xai_component
@xai_component(color="#776f85")
class VisualizeModelGraph(Component):
    model_function: InArg[any]

    def __init__(self):
        self.model_function = InArg.empty()

    def execute(self, ctx) -> None:
        import torch
        import matplotlib.image as mpimg
        from io import BytesIO

        data = torch.ones(10)
        png_path = "network.png"
        dot = pyro.render_model(
            model=self.model_function.value,
            model_args=(data,),
            render_params=True,
            render_distributions=True,
        )
        # Render the Digraph object into memory as PNG
        png_data = dot.pipe(format='png')

        # Convert byte data to image
        img = mpimg.imread(BytesIO(png_data), format='PNG')
        plt.imshow(img)
        plt.axis('off')  # Turn off axis
        plt.show()
