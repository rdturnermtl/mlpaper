from setuptools import setup, find_packages
from benchmark_tools import __version__
import io

setup(
    name='benchmark_tools',
    version=__version__,
    description='description here',
    long_description=io.open('README.md', encoding='utf-8').read(),
    url='https://github.com/rdturnermtl/benchmark_tools',
    author='Ryan Turner',
    author_email='turnerry@iro.umontreal.ca',
    license='...',
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
    # python_requires='>=3.4.3',
)