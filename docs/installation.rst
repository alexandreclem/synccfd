Installation
============

Prerequisites
------------

SynCCFD requires Python 3.9 or later. The package has the following core dependencies:

* numpy>=1.19.0
* pandas>=1.2.0
* geopandas>=0.9.0
* geopy>=2.2.0
* shapely>=1.7.0
* haversine>=2.3.0

Basic Installation
----------------

You can install SynCCFD using pip:

.. code-block:: bash

    pip install synccfd

Development Installation
----------------------

For development, you can install the package with additional dependencies:

.. code-block:: bash

    git clone https://github.com/alexandreclem/synccfd.git
    cd synccfd
    pip install -e ".[dev]"

This will install all development dependencies including:

* Testing tools (pytest, pytest-cov)
* Code quality tools (ruff, mypy, pre-commit)

Documentation and example extras are available via ``pip install -e ".[docs]"``
and ``pip install -e ".[examples]"``.