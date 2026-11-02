Customer drift and seasonality
==============================

``DriftConfig`` (customer and population drift) and ``SeasonalityConfig`` (calendar
patterns). See ``examples/drift.ipynb``.

.. code-block:: python

    from synccfd import DatasetGenerator
    from synccfd.simulator.config import DriftConfig, SeasonalityConfig

    generator = DatasetGenerator(
        n_customers=10_000, nb_days=365,
        drift=DriftConfig(customer_fraction=0.3), seasonality=SeasonalityConfig(),
    )

Regimes and drift types
-----------------------

Each customer has a **regime A**, its profile. A drifting customer
(``customer_fraction``, 30% by default) also gets a **regime B** and a drift type
(``type_weights``: sudden 0.3, gradual 0.3, incremental 0.2, recurrent 0.2). The
weight of regime B on day :math:`d` is evaluated at :math:`t = d + 0.5`:

.. math::

   w(t) = \begin{cases}
   \mathbb{1}[t \geq t_0] & \text{sudden} \\
   \min\!\left(1, \max\!\left(0, \frac{t - t_0}{L}\right)\right) & \text{gradual, incremental} \\
   \mathbb{1}[t \geq t_0] \cdot \mathbb{1}[(t - t_0) \bmod P < D] & \text{recurrent}
   \end{cases}

with :math:`t_0` uniform in ``start_day_range`` (20% to 80% of the period),
:math:`L` in ``transition_days_range`` (7 to 30 days), :math:`P` in
``recurrent_period_days_range`` (30 to 90) and :math:`D` in
``recurrent_active_days_range`` (5 to 15).

Regime B is drawn from regime A:

.. math::

   \mu_B = \mu_A e^{Z_1},\quad \sigma_B = \mu_B / 2,\quad
   \lambda_B = \lambda_A e^{Z_2},\quad
   p_B = \operatorname{sigmoid}(\operatorname{logit}(p_A) + Z_3)

with :math:`Z_1 \sim \mathcal{N}(0, 0.5)`, :math:`Z_2 \sim \mathcal{N}(0, 0.4)`,
:math:`Z_3 \sim \mathcal{N}(0, 1)` (``amount_log_mu``, ``amount_log_sigma``,
``frequency_log_sigma``, ``cnp_logit_sigma``) and, with probability 0.05
(``relocation_prob``), a new city. On day :math:`d`, the number of transactions is
:math:`\mathrm{Poisson}\big(s_f(d)\,[(1 - w)\lambda_A + w\lambda_B]\big)`; in the
gradual type each transaction comes from B with probability :math:`w`, in the
sudden and recurrent types the active regime is used, and in the incremental type
the parameters are interpolated, :math:`\theta = (1 - w)\theta_A + w\theta_B`.

With merchants, regime B also changes the category preferences,
:math:`\text{pref}_B = (1 - s)\,\text{pref}_A + s\,\mathbf{q}`, with
:math:`s \sim \mathcal{U}` (``category_shift_range``, 0.3 to 0.8) and
:math:`\mathbf{q}` drawn like a new customer's preferences; the spending factor
gets the amount factor :math:`e^{Z_1}`.

Population drift and seasonality
--------------------------------

``DriftConfig(global_drift=GlobalDrift(...))`` adds a yearly amount trend and shocks
(``Shock``: a period with frequency and amount factors, such as Black Friday).
``SeasonalityConfig`` adds weekday factors, paydays (days 5 and 20 and the last
business day, frequency × 1.3) and December (frequency × 1.2, amount × 1.15). The
daily factors are

.. math::

   s_f(d) = f_{\text{weekday}}(d)\, f_{\text{payday}}(d)\, f_{\text{December}}(d)\,
   f_{\text{shock}}(d), \qquad
   s_v(d) = v_{\text{December}}(d)\,(1 + r)^{d / 365.25}\, v_{\text{shock}}(d).

These patterns repeat, so they are not labelled as drift.

Ground truth
------------

- ``customer_profiles``: ``GT_DRIFT_TYPE`` (``none``, ``sudden``, ``gradual``,
  ``incremental``, ``recurrent``), ``GT_DRIFT_T0``, ``GT_DRIFT_L``,
  ``GT_DRIFT_PERIOD``, ``GT_DRIFT_ACTIVE_DAYS``, the parameters of regime B
  (``GT_B_*``) and ``GT_RELOCATED``.
- ``transactions``: ``GT_REGIME_WEIGHT`` (:math:`w` on the transaction's day),
  ``GT_REGIME`` (``A`` or ``B``; in the incremental type, ``B`` once
  :math:`w \geq 0.5`) and, with global drift, ``GT_SHOCK``.
