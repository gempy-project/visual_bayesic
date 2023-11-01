"""
Getting Started
==================================

This example demonstrates the usage of the `visual_bayesic` package for Bayesian Inference using visual scripting. 
The "Getting Started" example is a simple example that showcases the ability of `visual_bayesic` to utilize visual
scripting for defining and running Bayesian Inference. The graphical representation, combined with the Python script,
offers a powerful way to work with Bayesian models, making the process more intuitive and user-friendly.

Let's dive into the execution.
"""

import os
import sys

from visual_bayesic.runner import display_graph, execute_model, import_model

# Determine the base path
if '__file__' in globals():
    base_path = os.path.dirname(__file__)
else:
    base_path = os.getcwd()


# Append the parent directory to sys.path
sys.path.append(os.path.join(base_path, '..'))

normal_likelihood = import_model("normal_likelihood")

# %%
# Below is the screenshot of the visual scripting graph that corresponds to the
# Python code executed above. This graph provides an intuitive way to understand
# the flow and connections of the different components used in the inference process.

display_graph(normal_likelihood)

# %%

# Call the function
execute_model(normal_likelihood)


# %%
# Conclusion:
# -----------
# This example showcased the ability of `visual_bayesic`  to utilize visual scripting for defining and running
# Bayesian Inference. The graphical representation, combined with the Python script, offers a powerful way to work with
# Bayesian models, making the process more intuitive and user-friendly.
