import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="consistency_models",
    py_modules=["consistency_models"],
    version="0.1.0",
    description="Unofficial Implementation of Consistency Models",
    author="Simo Ryu",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
