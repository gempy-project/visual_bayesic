import os
from argparse import ArgumentParser
from importlib import import_module
from types import ModuleType

from matplotlib import image as mpimg, pyplot as plt


def execute_and_display_graph(module_name):
    # Dynamic import
    mod = import_module(f"models.{module_name}.{module_name}")
    main_function = getattr(mod, "main")

    # Execute Bayesian Inference
    parser = ArgumentParser()
    main_function(parser.parse_args())
    
    return mod
    
    
def import_model(module_name) -> ModuleType:
    # Dynamic import
    mod = import_module(f"models.{module_name}.{module_name}")
    return mod    


def execute_model(mod: ModuleType):
    main_function = getattr(mod, "main")
    parser = ArgumentParser()
    main_function(parser.parse_args())
    
    
def display_graph(mod: ModuleType):
    module_dir = os.path.dirname(os.path.abspath(mod.__file__))
    # Add module name to the path
    img_path = os.path.join(module_dir, "graph.png")
    img = mpimg.imread(img_path)

    plt.figure(figsize=(23, 10))
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better visualization
