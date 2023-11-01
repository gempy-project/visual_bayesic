from xai_components.base import InArg, Component, xai_component
import seaborn as sns
import matplotlib.pyplot as plt

@xai_component
class VisualizeNormalDistribution(Component):
    samples: InArg[list[float]]
    title: InArg[str]

    def __init__(self):
        self.samples = InArg.empty()
        self.title = InArg.empty()

    def execute(self, ctx) -> None:
        sns.distplot(self.samples.value)
        plt.title(self.title.value)
        plt.show()
