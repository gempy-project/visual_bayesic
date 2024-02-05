from xai_components.base import InArg, OutArg, Component, xai_component, dynalist


@xai_component(color="#70856f")
class PyroModel(Component):
    # TODO: Here we need to have multiple models
    arg1: InArg[callable]
    # arg2: InArg[callable]
    # arg3: InArg[callable]
    # arg4: InArg[callable]
    # arg5: InArg[callable]

    model: OutArg[callable]

    def __init__(self):
        super().__init__()
        self.arg1 = InArg.empty()
        self.model = OutArg.empty()

    def execute(self, ctx) -> None:
        def pyro_model(_):
            # if self.arg1.value:
            self.arg1.value()

        # if self.arg2.value:
        #     self.arg2.value()
        # if self.arg3.value:
        #     self.arg3.value()
        # if self.arg4.value:
        #     self.arg4.value()
        # if self.arg5.value:
        #     self.arg5.value()

        self.model.value = pyro_model
