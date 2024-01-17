from xai_components.base import InArg, OutArg, Component, xai_component


@xai_component()
class PyroModelSampleOneRandomVariable(Component):
    # TODO: Here we need to have multiple models
    arg1: InArg[callable]
    model: OutArg[callable]

    def __init__(self):
        super().__init__()
        self.arg1 = InArg.empty()
        self.model = OutArg.empty()

    def execute(self, ctx) -> None:
        def pyro_model(_):
            self.arg1.value()

        self.model.value = pyro_model
