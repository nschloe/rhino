# -*- coding: utf-8 -*-
#
import os
from distutils.core import setup
import codecs

from pynosh import __version__, __author__, __author_email__


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
        ).read()
    except Exception:
        content = ""
    return content


setup(
    name="pynosh",
    packages=["pynosh"],
    version=__version__,
    description="Nonlinear Schrödinger equations",
    long_description=read("README.rst"),
    author=__author__,
    author_email=__author_email__,
    url="https://github.com/nschloe/pynosh/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
