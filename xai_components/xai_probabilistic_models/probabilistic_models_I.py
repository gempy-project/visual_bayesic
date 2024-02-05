from xai_components.base import InArg, OutArg, Component, xai_component, dynalist


@xai_component(color="#70856f")
class PyroModel(Component):
    # TODO: Here we need to have multiple models
    arg1: InArg[dynalist]
    model: OutArg[callable]

    def __init__(self):
        super().__init__()
        self.arg1 = InArg.empty()
        self.model = OutArg.empty()

    def execute(self, ctx) -> None:
        def pyro_model(_):
            for arg in self.arg1.value():
                arg()

        self.model.value = pyro_model
