# Running the tests

Make sure that pyimpspec is installed with the `--editable` tag from the root of the repository:

```
pip install --editable .
```

Execute the `run_tests.sh` or `run_tests.bat` script from within the `tests` folder.

There is a GitHub Actions workflow (`test-package.yml`) that runs the tests on different operating systems and different versions of Python.
