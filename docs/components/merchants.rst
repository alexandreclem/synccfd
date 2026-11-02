Merchants and categories
========================

``MerchantConfig``: the amount and the time of a purchase depend on the
merchant's category, not only on the customer. See ``examples/merchants.ipynb``.

Merchants
---------

- **Physical merchants** own the terminals: in each city, the terminals are grouped
  into merchants of :math:`1 + \mathrm{Poisson}(0.5)` terminals, each with a category
  drawn by weight among the categories with CP purchases.
- **Online merchants**: 20 per category with CNP purchases, each with one virtual
  terminal at its head office; their popularity follows a Zipf law
  (:math:`P(\text{rank } r) \propto r^{-1}`).

Default categories
------------------

``DEFAULT_CATEGORIES``; amounts are :math:`\text{median} \times g_c \times
e^{\sigma Z}`, :math:`Z \sim \mathcal{N}(0, 1)`. The values are reasonable choices,
not calibrated on real data, and can be replaced
(``MerchantConfig(categories=...)``).

.. list-table::
   :header-rows: 1

   * - Category
     - Channel
     - Median (R$)
     - :math:`\sigma`
     - Weight
     - CP hours
     - Weekend
     - Skimming risk
     - Online fraction
   * - supermercado
     - CP+CNP
     - 120
     - 0.7
     - 0.20
     - 8–21 h
     - 1.2
     - 1
     - 0.10
   * - restaurante_bar
     - CP
     - 60
     - 0.6
     - 0.15
     - 11–24 h
     - 1.4
     - 2
     - —
   * - posto_combustivel
     - CP
     - 150
     - 0.4
     - 0.10
     - 6–23 h
     - 1.0
     - 3
     - —
   * - farmacia
     - CP+CNP
     - 70
     - 0.6
     - 0.08
     - 8–22 h
     - 1.0
     - 1
     - 0.15
   * - vestuario
     - CP+CNP
     - 180
     - 0.7
     - 0.08
     - 10–22 h
     - 1.3
     - 1
     - 0.30
   * - eletronicos
     - CP+CNP
     - 900
     - 0.8
     - 0.04
     - 10–22 h
     - 1.1
     - 1
     - 0.50
   * - viagem_turismo
     - CNP
     - 1200
     - 0.7
     - 0.03
     - —
     - 1.0
     - —
     - —
   * - ecommerce_marketplace
     - CNP
     - 150
     - 0.9
     - 0.15
     - —
     - 1.1
     - —
     - —
   * - servicos_digitais
     - CNP
     - 35
     - 0.5
     - 0.10
     - —
     - 1.0
     - —
     - —
   * - transporte_app
     - CNP
     - 25
     - 0.5
     - 0.07
     - —
     - 0.7
     - —
     - —

Each category also has a 24-hour profile (peaks: supermarket at 18 h, restaurant at
12:30 and 20:30, transport at 8 h and 18:30, digital services at night, ...).

A purchase
----------

Each customer :math:`c` has preferences :math:`\text{pref}_c \sim
\mathrm{Dirichlet}(10\,\mathbf{p})` (:math:`\mathbf{p}`: the category weights), a
spending factor :math:`g_c \sim \mathrm{LogNormal}(0, 0.5)` and two favourite merchants
per category. A purchase on day :math:`d` draws, in this order:

1. the **category**, :math:`P(k) \propto \text{pref}_{c,k} \cdot
   f_k^{[\text{weekend}(d)]} \cdot \mathbb{1}[c \text{ can buy in } k]` (a CP-only
   category needs a merchant of it within 10 km);
2. the **type**: fixed by CP-only and CNP-only categories; in a mixed category, CNP
   with probability :math:`\sigma(c_k + \operatorname{logit}(p_c))`, where
   :math:`p_c` is the customer's ``cnp_prob`` and :math:`c_k` is set so that the mean
   over :math:`p_c \sim \mathcal{U}(0, 1)` is the category's online fraction; always
   CNP without a merchant of the category in reach;
3. the **merchant**: a favourite with probability 0.6, otherwise a reachable
   terminal of the category (CP) or an online merchant by popularity (CNP);
4. the **hour**, from the category profile (CP: opening hours only);
5. the **amount**, :math:`\text{median}_k \cdot g_c \cdot e^{\sigma_k Z}`, times the
   daily factors.

Outputs
-------

- ``transactions``: ``MERCHANT_ID`` and ``MERCHANT_CATEGORY``. They are observable
  attributes, like the MCC of a real transaction, so they are not ``GT_*``.
- ``terminal_profiles``: ``MERCHANT_ID``; the virtual terminals of the online
  merchants come after the physical ones.
- ``customer_profiles``: ``spending_factor`` and ``pref_<category>``; ``mean_amount``
  and ``std_amount`` are recomputed from the generated transactions.
- ``merchant_profiles``: ``MERCHANT_ID``, ``MERCHANT_CATEGORY``, ``channel``,
  ``n_terminals``, ``lat``, ``long``.
