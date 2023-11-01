# -*- coding: utf-8 -*-
"""
Python installation file for visual_bayesic.
"""
import sys
from os import path
from setuptools import setup, find_packages

if not sys.version_info[:2] >= (3, 8):
    sys.exit(f"visual_bayesic is only meant for Python 3.8 and up.\n"
             f"Current version: {sys.version_info[0]}.{sys.version_info[1]}.")

this_directory = path.abspath(path.dirname(__file__))

readme = path.join(this_directory, "README.md")
with open(readme, "r", encoding="utf-8") as f:
    long_description = f.read()
long_description = long_description.split('inclusion-marker')[-1]

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
]

setup(
    name="visual_bayesic",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    description="Description for visual_bayesic",  # You can update this description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="YOUR_PROJECT_URL_HERE",  # Update this URL to your project's repository or website
    author="Miguel de la Varga and Maximilian Hallenberger",  # Update with your name or organization's name
    author_email="miguel@terranigma-solutions.com",
    license="EUPL-1.2",
    install_requires=[
        # List your dependencies here, e.g.
        # "numpy",
        # "pandas",
    ],
    classifiers=CLASSIFIERS,
    zip_safe=False,
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": path.join("visual_bayesic", "_version.py"),
    },
    setup_requires=["setuptools_scm"],
)
