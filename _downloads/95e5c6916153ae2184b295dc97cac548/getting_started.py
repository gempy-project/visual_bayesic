"""
Getting Started with Visual Scripting in Bayesian Inference
==========================================================

`visual_bayesic` leverages the power of visual scripting to simplify and enhance the Bayesian inference process. 
This "Getting Started" guide demonstrates how the combination of graphical representations and Python scripts can make 
defining and running Bayesian models more intuitive and user-friendly.

Let's dive in!
"""

import os
import sys

from visual_bayesic.runner import display_graph, execute_model, import_model

# Determine the base path
if '__file__' in globals():
    base_path = os.path.dirname(__file__)
else:
    base_path = os.getcwd()

# Adjust the Python module search path to include the parent directory
sys.path.append(os.path.join(base_path, '..'))

normal_likelihood = import_model("normal_likelihood")

# %%
# The visual scripting graph below represents the Bayesian inference process. 
# Each node corresponds to a step or component, and the connections depict the flow of data and dependencies. 
# This graphical view provides an intuitive way to understand the structure and flow of the Bayesian model.

display_graph(normal_likelihood)

# %%

# Execute the model
execute_model(normal_likelihood)

# %%
# Conclusion
# ----------
# Through this example, we experienced the unique approach of `visual_bayesic` that harnesses the power of 
# visual scripting to simplify Bayesian inference. This intuitive blend of graphics and code empowers users 
# to effectively define, understand, and execute Bayesian models.
