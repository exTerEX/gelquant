#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import setuptools

cwd = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(cwd, "requirements.txt"), encoding="utf-8") as f:
    required = "\n" + f.read()

with io.open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

setuptools.setup(
    name="gelquant",
    version="0.1.1",
    description="Gel analysis pipeline",
    long_description=long_description,
    author="Joseph Harman",
    author_email="josephharman25@gmail.com",
    maintainer="Andreas Sagen",
    maintainer_email="developer@sagen.io",
    url="https://github.com/exterex/gelquant",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=required,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status:: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic:: Scientific / Engineering:: Bio - Informatics"
    ]
)
