"""
This is the first example of the documentation.
"""
from argparse import ArgumentParser

from first_prob_model.prob_model_I import main as first_prob_model_I

parser = ArgumentParser()
first_prob_model_I(parser.parse_args())