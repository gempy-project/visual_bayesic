"""
Getting Started
==================================

This example demonstrates the usage of the `visual_bayesic` package for Bayesian Inference using visual scripting. 
The "Getting Started" example is a simple example that showcases the ability of `visual_bayesic` to utilize visual
scripting for defining and running Bayesian Inference. The graphical representation, combined with the Python script,
offers a powerful way to work with Bayesian models, making the process more intuitive and user-friendly.

The code below was generated using a visual scripting framework named xircuits. The graphical representation provides
 an intuitive way of defining and running Bayesian Inference. We will visualize the graph at the end.

Let's dive into the execution.
"""

import os
import sys

from visual_bayesic.runner import execute_and_display_graph

# Append the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Call the function
execute_and_display_graph("normal_likelihood")

# %%
# Below is the screenshot of the visual scripting graph that corresponds to the
# Python code executed above. This graph provides an intuitive way to understand
# the flow and connections of the different components used in the inference process.


# %%
# Conclusion:
# -----------
# This example showcased the ability of `visual_bayesic` to utilize visual scripting for defining and running
# Bayesian Inference. The graphical representation, combined with the Python script, offers a powerful way to work with
# Bayesian models, making the process more intuitive and user-friendly.
