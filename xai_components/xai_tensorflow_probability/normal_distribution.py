from xai_components.base import InArg, OutArg, Component, xai_component
import tensorflow_probability as tfp

@xai_component
class NormalDistribution(Component):
    loc: InArg[float]
    scale: InArg[float]
    population: InArg[float]
    
    samples: OutArg[list[float]]

    def __init__(self):
        self.loc = InArg.empty()
        self.scale = InArg.empty()
        self.population = InArg.empty()

        self.samples = OutArg.empty()
        self.done = False

    def execute(self, ctx) -> None:
        normal = tfp.distributions.Normal(
            loc=self.loc.value,
            scale=self.scale.value
        )

        self.samples.value = normal.sample(self.population.value)

        self.done = False
