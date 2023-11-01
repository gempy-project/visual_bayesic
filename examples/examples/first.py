"""
This is the first example of the documentation.
"""
import os
import sys
sys.path.append(os.path.abspath('../../'))  # adjust path as necessary
sys.path.append(os.path.abspath('../../../'))  # adjust path as necessary
sys.path.append("C:/Users/MigueldelaVarga/PycharmProjects/VisualBayesic/")

print(os.getcwd())
print(sys.path)
# 
# from xai_components.xai_pyro.prob_models import RunMCMC
# print(xai_components)
# from argparse import ArgumentParser
# 
# from first_prob_model.prob_model_I import main as first_prob_model_I
# parser = ArgumentParser()
# first_prob_model_I(parser.parse_args())