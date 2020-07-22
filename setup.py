from setuptools import setup, find_packages
from benchmark_tools import __version__


def read_requirements(name):
    with open("requirements/" + name + ".in") as f:
        requirements = f.read().strip()
    requirements = requirements.replace("==", ">=").splitlines()  # Loosen strict pins
    return [pp for pp in requirements if pp[0].isalnum()]


requirements = read_requirements("base")

with open("README.md") as f:
    long_description = f.read()

setup(
    name="benchmark_tools",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/rdturnermtl/benchmark_tools/",
    author="Ryan Turner",
    author_email=("rdturnermtl@github.com"),
    license="Apache v2",
    description="Easy benchmarking of machine learning models with sklearn interface with statistical tests built-in.",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["any"],
)
