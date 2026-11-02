Contributing
============

We love your input! We want to make contributing to SynCCFD as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
-------------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from ``main``.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

Development Environment
-----------------------

1. Clone your fork and install development dependencies::

    git clone https://github.com/your-username/synccfd.git
    cd synccfd
    pip install -e .[dev]

2. Install pre-commit hooks::

    pre-commit install

3. Run tests to verify your setup::

    pytest

Code Style
----------

We use several tools to ensure consistent code style:

- ``black`` for code formatting
- ``flake8`` for code style and quality checks
- ``mypy`` for static type checking
- ``isort`` for import sorting

You can run all style checks with::

    black .
    flake8 .
    mypy .
    isort .

Testing
-------

We use ``pytest`` for testing. To run tests::

    pytest

For coverage report::

    pytest --cov=synccfd --cov-report=html

Documentation
-------------

To build documentation locally::

    cd docs
    make html

The built documentation will be in ``docs/_build/html``.