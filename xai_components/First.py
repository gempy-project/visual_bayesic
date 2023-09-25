from argparse import ArgumentParser
from xai_components.base import SubGraphExecutor
from xai_components.xai_custom.add_node import AddTwoFloats
from xai_components.xai_template.example_components import HelloHyperparameter

def main(args):
    ctx = {}
    ctx['args'] = args
    c_0 = HelloHyperparameter()
    c_1 = AddTwoFloats()
    c_0.input_str = c_1.output_int
    c_1.input_1.value = 4
    c_1.input_2.value = 5
    c_0.next = None
    c_1.next = c_0
    next_component = c_1
    while next_component:
        next_component = next_component.do(ctx)
if __name__ == '__main__':
    parser = ArgumentParser()
    main(parser.parse_args())
    print('\nFinished Executing')