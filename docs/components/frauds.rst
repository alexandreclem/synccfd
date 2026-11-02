Fraud scenarios
===============

``FraudConfig`` sets the fraud scenarios. See ``examples/frauds.ipynb``.

Two mechanisms
--------------

- ``mechanism="tamper"`` (default, v0.1): legitimate transactions are altered and
  labelled as fraud. Scenarios 1 to 4. The defaults reproduce v0.1.
- ``mechanism="inject"`` (v2 engine): fraudsters make **new** transactions, added to
  the customers' own ones. Legitimate transactions are never altered nor labelled,
  so a victim's history stays intact. Scenarios 1 to 5.
  ``FraudConfig.realistic()`` switches on injection, the five scenarios, rates
  proportional to the population and fraudster drift.

Scenarios (injection)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 5 20 75

   * - #
     - Scenario
     - What the fraudster does
   * - 1
     - anomalous amount
     - one purchase at :math:`2\,q\,(1 + 0.5\,U)`, :math:`U \sim \mathcal{U}(0, 1]`,
       where :math:`q` is the 95th percentile of the victim's legitimate purchases of
       the last 30 days (with fewer than 5, of all legitimate purchases of the period)
   * - 2
     - compromised terminal
     - a terminal (with merchants, a whole merchant) reads the card of every customer
       who buys there in 28 days; those purchases stay legitimate and get
       ``GT_COMPROMISE_ID``. 1 to 14 days later, a read card is cloned (probability
       0.2: a fast sequence of CP purchases near the fraudster, as in scenario 4) or
       leaks to scenario 3 (0.05)
   * - 3
     - stolen card number
     - :math:`1 + \mathrm{Poisson}(2)` bursts in 14 days, each of 2 to 5 CNP purchases
       within 3 hours, shipped to the fraudster, half of them starting at night;
       amounts :math:`2\,r\,e^{Z}`, with :math:`r` the victim's recent median and
       :math:`Z \sim \mathcal{N}(0, 0.3)`
   * - 4
     - stolen physical card
     - purchases near the victim's city every :math:`\mathrm{Exp}(20\ \text{min})`,
       amounts :math:`3\,r\,e^{Z}`, until the card is blocked (probability 0.15 after
       each purchase), R$ 2 000 are spent, 10 purchases are made or 24 h have passed;
       20% are CNP attempts
   * - 5
     - card testing
     - 3 to 10 CNP purchases of R$ 1 to 5, one every :math:`\mathrm{Exp}(60\ \text{s})`;
       with probability 0.5, scenario 3 follows 1 to 72 hours later, with the same
       ``GT_CAMPAIGN_ID``

Rates and fraudster drift
-------------------------

With ``rate_mode="proportional"``, the number of new victims (customers, or
terminals in scenario 2) of scenario :math:`k` on day :math:`d` is
:math:`\mathrm{Poisson}(f_k \cdot N \cdot m_k(d))`, so the expected fraud rate does not
depend on the size of the dataset (``rate_mode="absolute"``, the v0.1 default, uses
fixed counts). The multiplier :math:`m_k` is the ``intensity_schedule`` of the
scenario, as a function of the fraction :math:`x` of the period:
``IntensitySchedule("step" | "ramp" | "pulse" | "constant", start, end, base,
level)``. In ``FraudConfig.realistic()``, scenario 2 halves from 60% of the period
on (chip cards) and scenario 5 starts at 30%.

With merchants
--------------

Each scenario buys in its own categories (``s1_categories``,
``s3_category_schedule``, ``s4_categories``, ``s5_categories``): scenario 1 in
electronics and travel, scenario 3 in resellable or digital categories (in
``realistic()``, it moves from electronics to digital services at 50% of the
period), scenario 4 and the cloned cards in resellable categories at open merchants,
scenario 5 in digital services. Scenario 2 compromises whole merchants, chosen with
probability proportional to the category's skimming risk.

Ground truth
------------

``TX_FRAUD`` and ``TX_FRAUD_SCENARIO`` (kept for compatibility) and, with injection,
``GT_FRAUD_SCENARIO``, ``GT_FRAUDSTER_ID``, ``GT_CAMPAIGN_ID`` (the transactions of
one attack) and ``GT_COMPROMISE_ID`` (the legitimate purchase where a card was read
and the frauds later made with it).
``synccfd.simulator.ground_truth.compromise_table(transactions)`` lists the
compromises.
