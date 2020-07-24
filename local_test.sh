#!/bin/bash
#
# For now, just run the regular tests, that's it

PY=python3.7

# Test using pinned
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -r requirements/tools.txt
pip install -r requirements/test.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings --cov=mlpaper --cov-report html
deactivate

# Test using latest
virtualenv env --python=$PY
source ./env/bin/activate
python --version
pip install -r requirements/tools.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings
deactivate
