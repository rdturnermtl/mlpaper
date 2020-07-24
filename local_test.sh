#!/bin/bash
#
# For now, just run the regular tests, that's it

set -ex
set -o pipefail

PY=python3.7

# Test using pinned
! test -d env
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -r requirements/tools.txt
pip install -r requirements/test.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings --cov=mlpaper --cov-report html
deactivate

# Test using latest
! test -d env_latest
virtualenv env_latest --python=$PY
source ./env_latest/bin/activate
python --version
pip install -r requirements/tools.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings
deactivate
