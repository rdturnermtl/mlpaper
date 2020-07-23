from setuptools import find_packages, setup

from mlpaper import __version__


def read_requirements(name):
    with open("requirements/" + name + ".in") as f:
        requirements = f.read().strip()
    requirements = requirements.replace("==", ">=").splitlines()  # Loosen strict pins
    return [pp for pp in requirements if pp[0].isalnum()]


requirements = read_requirements("base")
demo_requirements = read_requirements("demo")
test_requirements = read_requirements("test")

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mlpaper",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/rdturnermtl/mlpaper/",
    author="Ryan Turner",
    author_email=("rdturnermtl@github.com"),
    license="Apache v2",
    description="Easy benchmarking of machine learning models with sklearn interface with statistical tests built-in.",
    python_requires=">=3.5",
    install_requires=requirements,
    extras_require={"demo": demo_requirements, "test": test_requirements},
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["any"],
)
