Output
======

``generate()`` returns three DataFrames (customers, terminals, transactions);
``generate_all()`` returns a ``SimulationResult`` with the merchant profiles and the
configuration as well. ``save()`` writes the profiles, the transactions (one file per
day, or one file) and ``config.json``.

.. warning::

   **Ground truth is not a feature.** Columns starting with ``GT_`` (and
   ``TX_FRAUD_SCENARIO``) describe how the data was generated: fraud scenarios,
   fraudsters, campaigns, drift regimes. A model could not observe them in practice,
   so never use them as model inputs. They are there to evaluate models, for example
   per scenario or per regime. ``TX_FRAUD`` is the label.

Transactions
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Description
   * - ``TRANSACTION_ID``
     - chronological position
   * - ``TX_DATETIME``, ``TX_TIME_SECONDS``, ``TX_TIME_DAYS``
     - time stamp, and time since ``start_date``
   * - ``CUSTOMER_ID``, ``TERMINAL_ID``
     - customer and terminal
   * - ``MERCHANT_ID``, ``MERCHANT_CATEGORY``
     - merchant and its category (with merchants; observable, like an MCC)
   * - ``TX_AMOUNT``, ``TX_TYPE``
     - amount (R$) and type (``CP`` or ``CNP``)
   * - ``TX_BILL_*``, ``TX_TERM_*``, ``TX_SHIPP_*``
     - billing, terminal and shipping locations (latitude, longitude)
   * - ``TX_FRAUD``
     - label: 0 legitimate, 1 fraud
   * - ``TX_FRAUD_SCENARIO``
     - ground truth: 0 legitimate, 1 to 5 fraud scenario
   * - ``GT_REGIME``, ``GT_REGIME_WEIGHT``, ``GT_SHOCK``
     - ground truth of drift: regime (``A`` or ``B``) of a legitimate transaction,
       weight of regime B on its day, global shock
   * - ``GT_FRAUD_SCENARIO``, ``GT_FRAUDSTER_ID``, ``GT_CAMPAIGN_ID``,
       ``GT_COMPROMISE_ID``
     - ground truth of fraud injection (:doc:`components/frauds`)

Customer profiles
-----------------

``CUSTOMER_ID``, ``billing_lat``, ``billing_long``, ``mean_amount``, ``std_amount``,
``mean_nb_tx_per_day``, ``cnp_prob`` and ``available_terminals``. With merchants,
``spending_factor`` and ``pref_<category>``. With drift, ``GT_DRIFT_*``, ``GT_B_*`` and
``GT_RELOCATED`` (:doc:`components/drift`).

Terminal and merchant profiles
------------------------------

``terminal_profiles``: ``TERMINAL_ID``, ``terminal_lat``, ``terminal_long`` and, with
merchants, ``MERCHANT_ID``. ``merchant_profiles`` (with merchants): ``MERCHANT_ID``,
``MERCHANT_CATEGORY``, ``channel``, ``n_terminals``, ``lat``, ``long``.

``config.json``
---------------

The effective configuration, written by ``save()`` next to the data and available as
``DatasetGenerator.config`` and ``SimulationResult.config``: the version of
``synccfd``, the size, the seed and every configuration object, each as a
JSON object with its ``type`` and fields. It records everything needed to
regenerate the dataset with the same version of ``synccfd``.
