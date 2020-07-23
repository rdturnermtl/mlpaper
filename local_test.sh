# For now, just run the regular tests, that's it
python --version

PY=python3.7

# Test using pinned
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r requirements/tools.txt
pip install -r requirements/test.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings
deactivate

# Test using latest
virtualenv env --python=$PY
source ./env/bin/activate
pip install -r requirements/tools.txt
pip install -e .[test]
pytest tests/ -s -v --disable-pytest-warnings
deactivate
