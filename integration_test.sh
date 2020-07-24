#!/bin/bash

set -ex
set -o pipefail

# For now, just run the demos, that's it
PY=python3.7

# Test using pinned
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -r requirements/demo.txt
pip install -e .[demo]
python demos/plot_classifier_comparison.py
python demos/plot_regression_comparison.py
deactivate

# Test using latest
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -e .[demo]
python demos/plot_classifier_comparison.py
python demos/plot_regression_comparison.py
deactivate
