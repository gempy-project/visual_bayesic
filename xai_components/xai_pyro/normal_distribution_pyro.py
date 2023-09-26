import pyro
from pyroapi import distributions

from xai_components.base import InArg, OutArg, Component, xai_component


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
        samples = pyro.sample(
            name="my_samples",
            fn=distributions.Normal(
                loc=self.loc.value,
                scale=self.scale.value
            ),
            sample_shape=(self.population.value,)
        )
        self.samples.value = samples

        self.done = False
