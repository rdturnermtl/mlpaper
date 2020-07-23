# For now, just run the regular tests, that's it
python --version
pytest test/ -s -v --disable-pytest-warnings
