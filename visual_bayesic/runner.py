import os
from argparse import ArgumentParser
from importlib import import_module

from matplotlib import image as mpimg, pyplot as plt


def execute_and_display_graph(module_name):
    # Dynamic import
    mod = import_module(f"models.{module_name}.{module_name}")
    main_function = getattr(mod, "main")

    # Execute Bayesian Inference
    parser = ArgumentParser()
    main_function(parser.parse_args())

    # Display the corresponding graph
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory of the executing script

    module_dir = os.path.dirname(os.path.abspath(mod.__file__))
    # Add module name to the path
    img_path = os.path.join(module_dir, "graph.png")
    img = mpimg.imread(img_path)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
