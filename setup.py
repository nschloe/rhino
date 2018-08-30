# -*- coding: utf-8 -*-
#
import os
import codecs

from setuptools import setup, find_packages

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "pynosh", "__about__.py"), "rb") as f:
    exec(f.read(), about)


def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()


setup(
    name="pynosh",
    packages=find_packages(),
    version=about["__version__"],
    description="Nonlinear Schrödinger equations",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__author_email__"],
    install_requires=["numpy", "scipy", "krypy", "meshplex"],
    url="https://github.com/nschloe/pynosh/",
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
