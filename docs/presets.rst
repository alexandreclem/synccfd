Presets
=======

``synccfd.simulator.presets`` returns ready keyword arguments for
:class:`~synccfd.simulator.generator.DatasetGenerator`. Any argument can be
overridden.

.. code-block:: python

    from synccfd import DatasetGenerator
    from synccfd.simulator import presets

    # Every component: merchants, customer drift, seasonality, fraud injection
    # (the same as the defaults of DatasetGenerator)
    generator = DatasetGenerator(**presets.realistic(n_customers=1_000))

    # A step of the ablation
    generator = DatasetGenerator(**presets.ablation("drift", n_customers=1_000))

    # Everything in one object, with the effective configuration
    result = generator.generate_all()
    result.customers, result.terminals, result.merchants, result.transactions
    result.config  # also written to config.json by generator.save()

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Preset
     - What it sets
   * - ``realistic()``
     - ``MerchantConfig()``, ``DriftConfig()``,
       ``SeasonalityConfig()`` and ``FraudConfig.realistic()``, and
       ``DEFAULT_SIZE``: 10 000 customers, 20 000 terminals, 180 days from
       2023-01-01, ``random_state=42``. With no argument, it is the dataset of the
       dissertation.
   * - ``ablation(step)``
     - the components switched on cumulatively, in the order they were added, with
       ``DEFAULT_SIZE`` and fraud rates calibrated to be about the same.

Ablation
--------

``ablation(step)`` isolates the effect of each component on a model: every step
uses the same size and seed, and adds one component to the previous one.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``step``
     - Configuration
   * - ``"base"``
     - customers with a stationary behaviour and injected frauds of scenarios 1 to
       4 at constant rates: ``FraudConfig(scenarios=(1, 2, 3, 4))``, no other
       component
   * - ``"drift"``
     - plus ``DriftConfig()`` and ``SeasonalityConfig()``
   * - ``"frauds"``
     - with ``FraudConfig.realistic()`` instead: fraud injection, five scenarios and
       fraudster drift
   * - ``"merchants"``
     - plus ``MerchantConfig()``; the same as ``realistic()``

Every step injects frauds, and the daily fractions of all the scenarios are
multiplied by one factor per step (``ABLATION_FRAUD_SCALE``: 1.08, 1.11, 0.95 and
1), so that the fraud rate is about the same at every step: 0.347%, 0.343%, 0.352%
and 0.350%, on average over three seeds, with ``DEFAULT_SIZE``. The mix of
scenarios still changes between the steps (scenario 5 only exists from
``"frauds"`` on).

Presets by name
---------------

``presets.get("realistic")`` and
``presets.get("ablation:drift")`` return the same as the functions.
``presets.from_env()`` reads the preset from ``SYNCCFD_PRESET`` and the size from
``SYNCCFD_N_CUSTOMERS``, ``SYNCCFD_N_TERMINALS``, ``SYNCCFD_NB_DAYS`` and
``SYNCCFD_SEED``. The example notebooks use it, so that ``scripts/make_figures.py``
can run all of them on the same dataset.
