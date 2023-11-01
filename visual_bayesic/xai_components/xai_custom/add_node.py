from visual_bayesic.xai_components.base import InArg, OutArg, Component, xai_component


@xai_component
class AddTwoFloats(Component):
    input_1: InArg[float]
    input_2: InArg[float]
    output_int: OutArg[float]

    def __init__(self):
        self.input_1 = InArg.empty()
        self.input_2 = InArg.empty()
        
        self.output_int = OutArg.empty()
        self.done = False

    def execute(self, ctx) -> None:
        input_1 = self.input_1.value
        input_2 = self.input_2.value

        self.output_int.value = input_1 + input_2

        self.done = False
