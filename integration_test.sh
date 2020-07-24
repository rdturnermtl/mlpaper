#!/bin/bash
#
# For now, just run the demos, that's it

set -ex
set -o pipefail

PY=python3.7

# Test using pinned
! test -d env_int
virtualenv env_int --python=$PY
source ./env_int/bin/activate
python --version
pip install -r requirements/demo.txt
pip install -e .[demo]
python demos/plot_classifier_comparison.py
python demos/plot_regression_comparison.py
deactivate

# Test using latest
! test -d env_int_latest
virtualenv env_int_latest --python=$PY
source ./env_int_latest/bin/activate
python --version
pip install -e .[demo]
python demos/plot_classifier_comparison.py
python demos/plot_regression_comparison.py
deactivate
