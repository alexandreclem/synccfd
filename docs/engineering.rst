Software engineering
====================

How the package is tested, checked and released.

Tests
-----

``pytest`` runs the fast suite; ``pytest -m slow`` runs the long end-to-end tests,
including every example notebook (with reduced sizes, through the ``SYNCCFD_*``
variables of :doc:`presets`).

- **Golden test** (``tests/golden``): the legacy engine regenerates a small v0.1.0
  dataset and compares it exactly with the stored one.
- **v2 against legacy** (``tests/engine/test_v2_vs_legacy.py``): the v2 engine
  reproduces the v0.1 model statistically: relative differences of means and rates,
  Kolmogorov–Smirnov distances and, for amounts and profiles, KS tests.
- **Components**: one module per component (``test_drift.py``,
  ``test_injection.py``, ``test_merchants.py``), each with the acceptance criteria
  of its phase of the plan, and tests that switching a component off leaves the
  data unchanged.
- **API**: ``tests/simulator`` (configurations, generator, presets,
  ``config.json``).

Coverage of the fast suite (``pytest --cov=synccfd``), version 0.2.0: **97%** of
2 220 statements; the lowest are ``utils/data.py`` (77%, reading of formats other
than the ones tested) and ``config.py`` (95%, some validation messages).

Static checks
-------------

- ``ruff check .`` and ``ruff format --check .`` (pycodestyle, pyflakes, isort,
  pyupgrade, bugbear, comprehensions; line length 88).
- ``mypy src``, with ``py.typed`` shipped in the package. The code supports Python
  3.9 (``requires-python >= 3.9``): no ``match`` and no ``X | Y`` in annotations
  evaluated at run time.

Continuous integration
----------------------

``.github/workflows/ci.yml`` runs on every push to ``master`` and ``feat/**`` and on
every pull request:

- ``test``: ruff, mypy and the fast suite with coverage, on Python 3.9, 3.10, 3.11
  and 3.12 (Ubuntu 24.04, the last image with Python 3.9);
- ``slow-tests``: the slow suite, including the notebooks, on Python 3.11.

Each phase of the development was one branch and one pull request, merged only with
both jobs green.

Figures and statistics
----------------------

``python scripts/make_figures.py`` runs the four example notebooks on the same
dataset (``presets.realistic()``, the configuration of the dissertation), regenerates
every figure under ``docs/figures/``, draws the architecture diagrams and writes the
dataset statistics to ``docs/figures/dataset_stats.json``. The data are
deterministic, and the figure files have a fixed creation date
(``SOURCE_DATE_EPOCH``), so a second run rewrites unchanged files identically.

Release process
---------------

1. Update the version in ``pyproject.toml``, ``src/synccfd/__init__.py`` and
   ``CITATION.cff`` (with ``date-released``), and move the ``Unreleased`` entries of
   ``CHANGELOG.md`` to the new version, with the date.
2. Merge to ``master`` through a pull request with the CI green.
3. Create the tag ``vX.Y.Z`` and a GitHub release from it. The release triggers
   ``.github/workflows/publish.yml``, which builds the sdist and the wheel and
   publishes them to PyPI with trusted publishing (environment ``pypi``).
4. Read the Docs builds the documentation from ``master`` (``.readthedocs.yaml``).

Versions follow semantic versioning. v0.1.0 stays available on PyPI
(``pip install synccfd==0.1.0``).
