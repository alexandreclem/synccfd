Presets
=======

``synccfd.simulator.presets`` returns ready keyword arguments for
:class:`~synccfd.simulator.generator.DatasetGenerator`. Any argument can be
overridden.

.. code-block:: python

    from synccfd import DatasetGenerator
    from synccfd.simulator import presets

    # The v0.1 engine and model, exactly
    generator = DatasetGenerator(**presets.legacy())

    # Every v0.2 component: merchants, customer drift, seasonality, fraud injection
    generator = DatasetGenerator(**presets.realistic(n_customers=1_000))

    # Everything in one object, with the effective configuration
    result = generator.generate_all()
    result.customers, result.terminals, result.merchants, result.transactions
    result.config  # also written to config.json by generator.save()

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Preset
     - What it sets
   * - ``legacy()``
     - ``engine="legacy"`` and ``FraudConfig()``: with no other argument, exactly
       ``DatasetGenerator()`` of v0.1.
   * - ``realistic()``
     - the v2 engine with ``MerchantConfig()``, ``DriftConfig()``,
       ``SeasonalityConfig()`` and ``FraudConfig.realistic()``, and
       ``DEFAULT_SIZE``: 10 000 customers, 20 000 terminals, 180 days from
       2023-01-01, ``random_state=42``. With no argument, it is the dataset of the
       dissertation.
   * - ``ablation(step)``
     - the components switched on cumulatively, in the order they were added,
       always with the v2 engine and ``DEFAULT_SIZE``.

Ablation
--------

``ablation(step)`` isolates the effect of each component on a model: every step
uses the same engine, size and seed, and adds one component to the previous one.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``step``
     - Configuration
   * - ``"base"``
     - the v0.1 model on the v2 engine: ``FraudConfig(rate_mode="proportional")``
       (frauds alter legitimate transactions, as in v0.1)
   * - ``"drift"``
     - plus ``DriftConfig()`` and ``SeasonalityConfig()``
   * - ``"frauds"``
     - with ``FraudConfig.realistic()`` instead: fraud injection, five scenarios and
       fraudster drift
   * - ``"merchants"``
     - plus ``MerchantConfig()``; the same as ``realistic()``

The fraud rate changes between the steps (about 0.58%, 0.89%, 0.38% and 0.36% with
``DEFAULT_SIZE``): with drift, the fixed amount threshold of scenario 1 labels more
of the legitimate purchases of customers whose regime B spends more, and injection
changes the scenarios. Compare models by metrics that do not depend on the fraud
rate, or per scenario.

Presets by name
---------------

``presets.get("realistic")``, ``presets.get("legacy")`` and
``presets.get("ablation:drift")`` return the same as the functions.
``presets.from_env()`` reads the preset from ``SYNCCFD_PRESET`` and the size from
``SYNCCFD_N_CUSTOMERS``, ``SYNCCFD_N_TERMINALS``, ``SYNCCFD_NB_DAYS`` and
``SYNCCFD_SEED``. The example notebooks use it, so that ``scripts/make_figures.py``
can run all of them on the same dataset.
