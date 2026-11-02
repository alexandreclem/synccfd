# Plano de implementação — SynCCFD v0.2

> Documento para ser executado com o Claude Code no VS Code, dentro do repositório
> `alexandreclem/synccfd`. Salve este arquivo como `docs/plan-v0.2.md` e peça uma fase
> por vez (veja "Como usar com o Claude Code", no fim).
>
> O documento tem duas metades:
> 1. **Plano** (seções "Contexto" até "Riscos"): o que fazer e como verificar.
> 2. **Para o texto** (seção final): o que foi feito de fato, preenchido ao fim de cada
>    fase. É a matéria-prima do cap. 4 da dissertação e de partes do cap. 2.

## Contexto

O SynCCFD v0.1.0 é o simulador de Santos et al. (SBSeg 2024) empacotado como biblioteca.
A versão 0.2 (frente [SIM] do `MAPA_DO_MESTRADO.md`) acrescenta:

1. **Customers** — evolução comportamental com *concept drift* configurável (Fase 2);
2. **Frauds** — cenários ajustados e fraudadores mais realistas (Fase 3);
3. **Merchants** — categorias de comerciantes com distribuições de gasto e horários (Fase 4).

Todas as fases ficam prontas **antes da qualificação**.

Nada disso existe na v0.1.0. O simulador gera **apenas dados brutos** (perfis e transações
rotuladas). Engenharia de atributos, modelos e avaliação ficam no pipeline (frente [DDM]).

Este plano define o que implementar, como e como verificar.

## Princípios (valem para todas as fases)

1. **Compatibilidade total.** Com a configuração padrão, `DatasetGenerator(...).generate()`
   produz **exatamente** os mesmos DataFrames da v0.1.0 (mesmas colunas, mesmos valores).
   Isso é garantido por um teste de regressão "golden" criado na Fase 0, **antes** de
   qualquer refatoração.
2. **Tudo novo é opt-in.** Cada componente (drift, fraudes novas, merchants) é ligado por
   um objeto de configuração próprio. `None` = desligado. Isso permite ablação na
   dissertação: base → +drift → +fraudes realistas → +merchants.
3. **Dois motores.**
   - `legacy`: o código atual, intocado na lógica e na ordem dos sorteios aleatórios. É usado
     quando nenhum componente novo está ligado.
   - `v2`: motor novo, usado quando qualquer componente novo está ligado. Usa **apenas**
     `numpy.random.Generator` local (`np.random.default_rng(seed)`), nunca o estado global
     (`np.random.seed`, `random.seed`). Cada subsistema recebe seu próprio fluxo, derivado com
     `np.random.SeedSequence(seed).spawn(k)`, para que ligar um componente não altere os
     sorteios de outro.
4. **Gabarito separado de atributos.** Toda informação "verdade" que um modelo não poderia
   observar (cenário de fraude, regime de drift, etc.) vai em colunas com prefixo `GT_`.
   `TX_FRAUD_SCENARIO` é mantida por compatibilidade. A documentação deve dizer que colunas
   `GT_*` e `TX_FRAUD_SCENARIO` nunca podem ser usadas como atributos.
5. **Parâmetros empíricos e documentados.** Como no artigo, os valores padrão são escolhas
   razoáveis, não calibradas em dados reais. Todo valor padrão fica num único lugar
   (`synccfd/simulator/config.py` e `synccfd/simulator/presets.py`), com um comentário curto
   explicando a escolha.
6. **Python 3.9.** O `pyproject.toml` declara `requires-python >= 3.9`: não usar `match`,
   nem `X | Y` em anotações avaliadas em tempo de execução (usar `Optional[...]`/`Union[...]`).
7. **Uma fase = um PR.** Cada fase termina com `ruff check .`, `mypy src` e `pytest` passando.

## Visão geral da API final

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import (
    FraudConfig, MerchantConfig, DriftConfig, SeasonalityConfig,
)
from synccfd.simulator import presets

# 1) Igual à v0.1 (motor legacy)
gen = DatasetGenerator(n_customers=10_000, n_terminals=20_000, nb_days=180,
                       start_date="2023-01-01", random_state=42)

# 2) Componentes ligados à parte (motor v2)
gen = DatasetGenerator(
    n_customers=10_000, n_terminals=20_000, nb_days=180,
    start_date="2023-01-01", random_state=42,
    drift=DriftConfig(customer_fraction=0.3),                               # Fase 2
    seasonality=SeasonalityConfig(),                                        # Fase 2
    fraud=FraudConfig(rate_mode="proportional", scenarios=(1, 2, 3, 4, 5)), # Fase 3
    merchants=MerchantConfig(),                                             # Fase 4
)

# 3) Preset pronto para os experimentos do mestrado
gen = DatasetGenerator(**presets.realistic(n_customers=10_000, nb_days=180))

customers, terminals, transactions = gen.generate()   # assinatura mantida
result = gen.generate_all()                           # novo: dataclass com tudo
result.merchants, result.customers, result.terminals, result.transactions
```

`generate()` continua retornando 3 DataFrames, para não quebrar a API. O novo
`generate_all()` retorna um `SimulationResult` com `customers`, `terminals`, `merchants`
(ou `None`), `transactions` e `config`, a configuração efetiva serializável. `save()`
passa a gravar também `merchant_profiles` e `config.json` quando existirem.

---

## Fase 0 — Rede de segurança e configuração da fraude

**Objetivo:** congelar o comportamento da v0.1 e expor os parâmetros que hoje estão fixos
no código.

### 0.1 Teste golden (fazer primeiro, antes de tocar em qualquer código)
- Criar `tests/golden/test_legacy_reproducibility.py`.
- Gerar com a v0.1 **atual** um dataset pequeno (`n_customers=100, n_terminals=200,
  nb_days=15, start_date="2023-01-01", random_state=42`) e salvar os três DataFrames em
  `tests/golden/data/v0_1_0_small.parquet` (ou um `.pkl` por DataFrame).
- O teste regenera com a configuração padrão e compara com
  `pd.testing.assert_frame_equal(..., check_exact=True)`.
- Deixar um script `scripts/make_golden.py` que recria o arquivo, para que fique claro como
  ele foi produzido. Só pode ser rodado na v0.1.0.

### 0.2 `FraudConfig`
Novo módulo `src/synccfd/simulator/config.py` com dataclasses `frozen=True`:

```python
@dataclass(frozen=True)
class FraudConfig:
    scenarios: tuple[int, ...] = (1, 2, 3, 4)       # quais cenários aplicar
    rate_mode: str = "absolute"                      # "absolute" (v0.1) | "proportional"
    # Cenário 1 — valor alto
    s1_amount_threshold: float = 220.0
    # Cenário 2 — terminais comprometidos
    s2_terminals_per_day: int = 2
    s2_terminal_fraction_per_day: float = 1e-4       # usado em rate_mode="proportional"
    s2_duration_days: int = 28
    # Cenários 3 e 4 — clientes comprometidos
    s3_customers_per_day: int = 3
    s4_customers_per_day: int = 3
    s3_customer_fraction_per_day: float = 3e-4
    s4_customer_fraction_per_day: float = 3e-4
    s3_duration_days: int = 14
    s4_duration_days: int = 5
    s3_tamper_fraction: float = 1 / 3
    amount_multiplier: float = 5.0
    s4_cnp_prob: float = 0.5
```

- Os valores padrão reproduzem exatamente a v0.1. O golden test tem que continuar passando.
  Atenção: a v0.1 sorteia `s3 + s4` clientes **numa única chamada**
  (`sample(n=6, random_state=day)`) e depois divide (os 3 primeiros vão para o cenário 3).
  O caminho legacy precisa manter isso, além da checagem `len(disponíveis) >= s3 + s4`.
- `rate_mode="proportional"`: número por dia = `max(1, round(fração × população))`. Isso
  resolve o problema atual em que um dataset pequeno sai com cerca de 15% de fraude.
- `FraudSimulator` passa a receber o `FraudConfig` e ler dele todos os valores fixos.
- Validação em `__post_init__`: frações entre 0 e 1, durações > 0, cenários conhecidos.

### 0.3 Critérios de aceite
- [ ] Golden test passa.
- [ ] Teste: `rate_mode="proportional"` mantém a taxa de fraude estável (±30% relativo)
      entre `n_customers=500` e `n_customers=5000` (mesmos dias).
- [ ] Teste: `FraudConfig(scenarios=(1,))` gera só `TX_FRAUD_SCENARIO ∈ {0, 1}`.
- [ ] CHANGELOG e README atualizados (seção "Fraud configuration").

---

## Fase 1 — Motor v2 (base sem extras)

**Objetivo:** um motor novo, rápido e com RNG local, que reproduz **estatisticamente** o
modelo da v0.1. É a fundação sobre a qual entram merchants, drift e fraudes novas.

### 1.1 Estrutura
```
src/synccfd/simulator/
  config.py          # dataclasses (Fase 0 e seguintes)
  presets.py         # legacy(), realistic(), etc.
  legacy/            # mover generator.py e fraud.py atuais para cá, SEM mudar lógica
  engine/
    rng.py           # SeedSequence -> fluxos nomeados (customers, terminals, merchants,
                     #   transactions, drift, fraud)
    customers.py
    terminals.py     # reaproveita LocationFeatures
    transactions.py
    fraud.py         # cenários v2
  generator.py       # DatasetGenerator: escolhe legacy ou v2
```
- `DatasetGenerator` decide o motor: `legacy` se `merchants is None and drift is None and
  seasonality is None and fraud.engine == "legacy"`; caso contrário, `v2`. Expor
  `DatasetGenerator(..., engine="auto" | "legacy" | "v2")` para forçar.
- `LocationFeatures` usa `random` global. No motor v2, injetar um `np.random.Generator`
  (parâmetro opcional `rng`). Sem `rng`, o comportamento atual se mantém (necessário para
  o legacy).

### 1.2 Geração de transações no v2
- Mesmo modelo da v0.1: Poisson diário por cliente, valor normal truncado, tipo CP/CNP por
  `cnp_prob`, horário CP ~ N(12h, 20000 s), CNP bimodal (10h/20h), localizações iguais.
- Implementar **vetorizado por cliente** (sortear todos os dias de uma vez), nada de laço
  por transação. Meta: 10k clientes × 180 dias em menos de 2 min num notebook comum.
- Corrigir de forma explícita, só no v2: transações com horário inválido são reamostradas
  (não descartadas). Assim a frequência diária fica fiel ao parâmetro. Documentar.

### 1.3 Critérios de aceite
- [ ] Golden test continua passando (legacy intocado).
- [ ] Teste estatístico v2 × legacy, com os mesmos parâmetros (`n_customers=1000`,
      `nb_days=30`), comparando:
      transações por cliente-dia, valor médio, proporção CNP, histograma de horário (teste
      KS, p > 0,01) e taxa de fraude por cenário (diferença relativa < 20%).
- [ ] Reprodutibilidade: mesma seed → mesmos dados; seeds diferentes → dados diferentes.
- [ ] Benchmark simples em `scripts/benchmark.py` registrado no PR.

---

## Fase 2 — Customers com *concept drift* configurável

**Objetivo:** o comportamento dos clientes muda ao longo do tempo, de forma controlada e
rotulada. Isso cobre a taxonomia clássica de Gama et al. (2014), já citada no artigo:
súbito, gradual, incremental e recorrente.

### 2.1 Modelo de regimes
Cada cliente tem um **regime A** (perfil original). Um cliente com drift ganha também um
**regime B**, gerado a partir de A por perturbações configuráveis:

| atributo | perturbação em B |
|---|---|
| gasto (`fator_cliente` ou `mean_amount`) | × `LogNormal(μ_amount, σ_amount)`, padrão μ=0, σ=0,5 |
| frequência (`mean_nb_tx_per_day`) | × `LogNormal(0, 0,4)` |
| `cnp_prob` | logit(p) + `N(0, 1)` |
| `category_pref` (só quando a Fase 4, comerciantes, estiver pronta) | mistura `(1−ρ)·A + ρ·Dirichlet(...)`, ρ ~ U(0,3; 0,8) |
| cidade (opcional, `relocation_prob=0.05`) | mudança de cidade (endereço de cobrança e terminais disponíveis) |

**Peso do regime B no tempo**, `w(t) ∈ [0, 1]`, conforme o tipo de drift do cliente:

| tipo | `w(t)` | como aplicar |
|---|---|---|
| súbito | 0 antes de `t0`, 1 depois | parâmetros de B a partir de `t0` |
| gradual | sobe linearmente de 0 a 1 em `[t0, t0+L]` | **cada transação** vem de B com probabilidade `w(t)` (mistura de conceitos) |
| incremental | idem | **parâmetros interpolados**: `θ(t) = (1−w)·θA + w·θB` |
| recorrente | alterna entre A e B a cada `P` dias (com duração de B configurável) | ex.: períodos de férias, viagens |

Gradual e incremental usam o mesmo `w(t)` mas se aplicam de forma diferente, como na
literatura: mistura de dois conceitos × deslocamento contínuo de um conceito.

### 2.2 `DriftConfig`
```python
@dataclass(frozen=True)
class DriftConfig:
    customer_fraction: float = 0.3                      # fração de clientes com drift
    type_weights: Mapping[str, float] = field(default_factory=lambda: {
        "sudden": 0.3, "gradual": 0.3, "incremental": 0.2, "recurrent": 0.2})
    start_day_range: tuple[float, float] = (0.2, 0.8)   # t0 como fração de nb_days
    transition_days_range: tuple[int, int] = (7, 30)    # L (gradual/incremental)
    recurrent_period_days_range: tuple[int, int] = (30, 90)
    recurrent_active_days_range: tuple[int, int] = (5, 15)
    amount_log_sigma: float = 0.5
    frequency_log_sigma: float = 0.4
    cnp_logit_sigma: float = 1.0
    category_shift_range: tuple[float, float] = (0.3, 0.8)   # usado só com comerciantes
    relocation_prob: float = 0.05
    global_drift: Optional["GlobalDrift"] = None        # ex.: inflação de valores
```
- Validar em `__post_init__` que os pesos somam 1 e que os intervalos são válidos.
- `GlobalDrift` (opcional): drift de população, que afeta todos os clientes. Por exemplo,
  `amount_trend_per_year=0.05` (inflação) ou um choque numa data, como uma "Black Friday"
  (multiplicador de frequência e valor por alguns dias).

### 2.3 Sazonalidade (`SeasonalityConfig`, separada do drift)
Padrões de calendário que são estacionários, logo **não** são drift. Ficam separados para
não confundir na avaliação:
- dia da semana (multiplicador de frequência por dia; com comerciantes, o fim de semana também
  varia por categoria);
- dia de pagamento (pico de frequência nos dias 5 e 20 e no último dia útil);
- dezembro (multiplicador de frequência e de valor).

### 2.4 Gabarito
- `customer_profiles`: `GT_DRIFT_TYPE` (`none|sudden|gradual|incremental|recurrent`),
  `GT_DRIFT_T0`, `GT_DRIFT_L`, `GT_DRIFT_PERIOD` e os parâmetros de B (colunas `GT_B_*`).
- `transactions`: `GT_REGIME_WEIGHT` (o `w(t)` no momento da transação) e `GT_REGIME`
  (`A`/`B`, no caso gradual o regime sorteado).

### 2.5 Critérios de aceite
- [ ] Golden test passa.
- [ ] Teste: clientes `none` têm a média móvel de valor estável (sem tendência
      significativa); clientes `sudden` mostram mudança de nível em `t0`.
- [ ] Teste: em `gradual`, a fração de transações com `GT_REGIME == "B"` acompanha `w(t)`.
- [ ] Teste: `customer_fraction=0` + sazonalidade desligada gera a mesma distribuição da
      Fase 1.
- [ ] Notebook `examples/drift.ipynb`: um painel por tipo de drift (valor e frequência de
      um cliente ao longo do tempo, com `t0` e `L` marcados) e a fração de clientes em B
      ao longo do tempo. **Esses gráficos são a figura central do cap. 4.**

---

## Fase 3 — Frauds: cenários adaptados e fraudadores realistas

**Objetivo:** fraudes injetadas como comportamento de um agente (o fraudador), coerentes
com o *drift* dos clientes, e com a estratégia dos fraudadores mudando no tempo.

**Esta fase não depende dos comerciantes** (Fase 4, que vem logo depois). Cada cenário tem
uma versão **sem comerciantes**, implementada aqui, e uma versão **enriquecida com
comerciantes**, ligada automaticamente quando a Fase 4 existir. Como todo componente é
opcional, isso também permite a ablação (fraudes realistas com e sem comerciantes).

> ⚠️ **Validar com o orientador antes de implementar:** a mudança de adulteração para
> injeção (3.1, abaixo). Ela muda a semântica dos cenários em relação ao artigo.

### 3.1 Mudança de mecanismo (só no motor v2): injeção em vez de adulteração
Na v0.1, os cenários 3 e 4 **alteram** transações legítimas do cliente (mudam
local, tipo e multiplicam o valor por 5). No v2, o fraudador **gera transações novas**,
que se somam às do cliente:
- o histórico legítimo do cliente continua íntegro, como acontece na realidade;
- valor, horário e comerciante da fraude vêm do comportamento do fraudador, não do cliente;
- `FraudConfig(engine="v2")` liga esse mecanismo; `engine="legacy"` mantém a adulteração.

### 3.2 Cenários no v2

| # | cenário | comportamento do fraudador (sem comerciantes) | enriquecimento com comerciantes (Fase 4) |
|---|---|---|---|
| 1 | valor anômalo | transação acima do quantil `q` (padrão 0,995) da distribuição **global** de valores. Substitui o limiar fixo de R$ 220, que não se ajusta a outras configurações | quantil **por categoria** (viagem e eletrônicos legitimamente caros não viram fraude) |
| 2 | terminal comprometido (*skimming*) | um terminal é comprometido por `D` dias; as transações CP ali viram fraude, e o cartão de cada vítima entra numa fila de "cartões vazados" que alimenta o cenário 3 dias depois (encadeamento realista) | compromete o **comerciante** inteiro (todos os seus terminais), com seleção ponderada pelo risco da categoria (posto e restaurante mais altos) |
| 3 | número do cartão roubado (CNP) | por `D` dias, o fraudador faz compras **novas** só em CNP, com entrega no local do fraudador, horário com viés noturno, rajadas de 2 a 5 compras em poucas horas e valores acima do padrão do cliente | compras concentradas em categorias revendáveis ou digitais (eletrônicos, marketplace, serviços digitais, viagem), com valores acima da mediana da categoria |
| 4 | cartão físico roubado (CP) | começa num instante `τ` e faz compras **novas** em sequência rápida (minutos), em terminais perto da cidade da vítima, até atingir um limite de gasto ou ser bloqueado (`s4_max_txs`, `s4_block_prob_per_tx`). Pode haver tentativas CNP | compras em categorias revendáveis (eletrônicos, posto, farmácia, supermercado) |
| 5 | **card testing** (novo) | o fraudador testa o cartão com 3 a 10 compras CNP de R$ 1–5, em poucos minutos. Com probabilidade `p_escalate`, segue para o cenário 3 após um intervalo de horas a dias. As transações de teste são rotuladas como cenário 5 | testes em serviços digitais |

### 3.3 Drift dos fraudadores
- `FraudConfig.intensity_schedule`: para cada cenário, uma função do tempo (constante,
  rampa, degrau ou pulso) que multiplica sua taxa diária. Exemplo do preset:
  - cenário 2 cai pela metade a partir de 60% do período (bancos reforçam o chip);
  - cenário 5 surge a partir de 30% do período;
  - com comerciantes: cenário 3 migra de categoria (eletrônicos → serviços digitais) a partir
    de 50% do período.
- Gabarito: `GT_FRAUD_SCENARIO` (igual a `TX_FRAUD_SCENARIO`), `GT_FRAUDSTER_ID` e
  `GT_CAMPAIGN_ID`, este último agrupando as transações de um mesmo ataque. É útil para
  avaliação por cenário e por ataque (por exemplo, "o modelo detectou o ataque até a k-ésima
  transação?").

### 3.4 Interação com drift (importante para a dissertação)
- Um cliente em transição de regime (drift legítimo) **não** deve ser confundido com fraude
  no gabarito. As duas coisas são independentes e estão rotuladas separadamente. Isso
  permite medir falsos positivos causados por drift.
- Opcional: `fraud_target_bias`, em que fraudadores preferem clientes de alto gasto.

### 3.5 Critérios de aceite
- [ ] Golden test passa.
- [ ] Teste: no v2, o número de transações legítimas de cada cliente não muda quando a
      fraude é ligada (injeção, não adulteração).
- [ ] Teste: cenário 5 só tem valores ≤ R$ 5 e `TX_TYPE == "CNP"`; quando escala, o
      `GT_CAMPAIGN_ID` é compartilhado com o cenário 3 seguinte.
- [ ] Teste: as transações do cenário 4 acontecem em até 24 h a partir de `τ` e somam até o
      limite configurado.
- [ ] Teste: `intensity_schedule` em degrau muda a taxa diária do cenário na data prevista.
- [ ] Taxa de fraude total do preset `realistic` entre 0,2% e 1% das transações
      (configurável; documentar o valor obtido).
- [ ] Teste: todos os cenários funcionam com os comerciantes desligados.
- [ ] Notebook `examples/frauds.ipynb`: taxa por cenário ao longo do tempo, exemplo de
      linha do tempo de uma campanha (*card testing* → CNP), valor por cenário.

---

## Fase 4 — Merchants (categorias de comerciantes)

**Objetivo:** o valor e o horário de uma compra passam a depender do *tipo* de comerciante,
e não só do cliente.

Esta é a última fase de modelagem. Além do modelo abaixo, ela liga os enriquecimentos já previstos nas fases
anteriores: a mudança de preferência por categoria no *drift* (Fase 2) e as versões dos
cenários de fraude com comerciantes (Fase 3).

### 4.1 Modelo
- **Comerciante** (`MERCHANT_ID`): tem uma categoria e um ou mais terminais. Os terminais
  atuais passam a pertencer a comerciantes: cada comerciante recebe `k ~ 1 + Poisson(λ)`
  terminais próximos (mesma cidade).
- Comerciantes **CNP-only** (e-commerce, serviços digitais) não têm terminal físico. Para
  manter o esquema, recebem um `TERMINAL_ID` virtual com localização = sede (cidade sorteada).
- **Categoria** (inspirada em MCC), definida em `MerchantConfig.categories`:

| categoria | canal | valor (lognormal, mediana R$) | σ log | perfil horário | peso |
|---|---|---|---|---|---|
| supermercado | CP+CNP | 120 | 0,7 | 8–21h, pico 18h | 0,20 |
| restaurante_bar | CP | 60 | 0,6 | picos 12h e 20h, até 0h | 0,15 |
| posto_combustivel | CP | 150 | 0,4 | 6–23h, plano | 0,10 |
| farmacia | CP+CNP | 70 | 0,6 | 8–22h | 0,08 |
| vestuario | CP+CNP | 180 | 0,7 | 10–22h, pico 16h | 0,08 |
| eletronicos | CP+CNP | 900 | 0,8 | 10–22h | 0,04 |
| viagem_turismo | CNP | 1200 | 0,7 | 8–23h | 0,03 |
| ecommerce_marketplace | CNP | 150 | 0,9 | bimodal 10h/21h | 0,15 |
| servicos_digitais | CNP | 35 | 0,5 | 24h, pico noite | 0,10 |
| transporte_app | CNP | 25 | 0,5 | picos 8h e 18h | 0,07 |

  Os valores são empíricos, mesmo critério do artigo. Ficam em
  `presets.DEFAULT_CATEGORIES` e podem ser substituídos pelo usuário.
- **Perfil horário**: vetor de 24 pesos por categoria (`hourly_profile`). Para CP,
  probabilidade zero fora do horário de funcionamento. Horário = hora sorteada pelo perfil
  + minuto/segundo uniformes.
- **Fim de semana**: multiplicador opcional por categoria (`weekend_factor`; por exemplo,
  restaurante 1,4 e transporte 0,7).
- **Preferências do cliente**: vetor `category_pref ~ Dirichlet(α × pesos_globais)`, com α
  configurável (α pequeno = clientes mais "especializados"). Salvo em `customer_profiles`.
- **Valor da compra**: `valor = mediana_categoria × fator_cliente × LogNormal(0, σ_categoria)`,
  com `fator_cliente ~ LogNormal(0, 0,5)` (gasto relativo do cliente). O `mean_amount` da
  v0.1 deixa de ser usado no motor v2 com merchants e é recalculado a partir dos dados
  gerados, para manter a coluna.
- **Escolha da transação**:
  1. sorteia a categoria pela preferência do cliente, renormalizada pelos canais possíveis;
  2. define o tipo: CP se a categoria aceita CP e há terminal disponível dessa categoria no
     raio do cliente; CNP se a categoria é só CNP; em categorias mistas, usa `cnp_prob` do
     cliente;
  3. escolhe o comerciante: CP, entre os disponíveis no raio, uniforme (ou com fidelidade,
     ver abaixo); CNP, entre os comerciantes CNP da categoria, com popularidade Zipf
     (poucos marketplaces grandes);
  4. sorteia horário e valor pela categoria.
- **Fidelidade** (opcional, `loyalty=0.6`): cada cliente tem um pequeno conjunto de
  comerciantes favoritos por categoria e volta a eles com essa probabilidade. Isso cria
  padrões sequenciais reais, útil para a LSTM.

### 4.2 Saídas
- `merchant_profiles`: `MERCHANT_ID, MERCHANT_CATEGORY, channel, n_terminals, lat, long`.
- `terminal_profiles`: ganha `MERCHANT_ID`.
- `transactions`: ganha `MERCHANT_ID` e `MERCHANT_CATEGORY`. São atributos observáveis
  (o MCC aparece em transações reais), então **não** levam o prefixo `GT_`.

### 4.3 Critérios de aceite
- [ ] Golden test passa.
- [ ] Teste: com comerciantes ligados, o cenário 1 usa o quantil por categoria, e o cenário 3
      concentra compras nas categorias configuradas.
- [ ] Teste: com *drift* e comerciantes ligados, clientes do regime B têm preferência por
      categoria diferente da do regime A.
- [ ] Notebooks das Fases 2 e 3 atualizados para mostrar também a visão por categoria.
- [ ] Teste: em CP, todo horário está dentro do horário de funcionamento da categoria.
- [ ] Teste: categorias só CNP nunca geram `TX_TYPE == "CP"`.
- [ ] Teste: a mediana do valor por categoria fica a menos de 15% do configurado
      (amostra grande).
- [ ] Notebook `examples/merchants.ipynb` com: valor por categoria (boxplot, escala log),
      heatmap hora × categoria, participação de cada categoria. Esses gráficos vão para o
      cap. 4 da dissertação.

---

## Fase 5 — Presets, documentação e release 0.2.0

- `presets.legacy()` gera exatamente a v0.1.
- `presets.realistic(**overrides)` liga merchants, drift, sazonalidade e fraude v2, com
  os padrões deste documento.
- `presets.ablation(step)`, com `step ∈ {"base", "drift", "frauds", "merchants"}`: liga os
  componentes cumulativamente, na mesma ordem das fases, usando o motor v2 em todas as
  etapas, para comparação justa. Esse é o desenho experimental da dissertação.
- `SimulationResult.config` serializado em `config.json` no `save()`, junto com
  `synccfd.__version__`, para reprodutibilidade.
- Documentação (Sphinx): uma página por componente, com as fórmulas do *drift* (Fase 2), os
  cenários de fraude (Fase 3) e a tabela de categorias (Fase 4).
- **Documentação de arquitetura** (`docs/architecture.rst`): diagrama de módulos, fluxo de
  geração (perfis → transações → *drift* → fraudes), papel de cada configuração e como
  estender (por exemplo, como adicionar um cenário de fraude). É a base da seção de
  arquitetura do cap. 4 da dissertação.
- **Engenharia de software**: conferir e documentar a cobertura de testes (`pytest --cov`), o
  CI, a tipagem (`mypy`) e o processo de release. Esses itens entram na seção de engenharia
  de software do cap. 4.
- README: novas colunas no esquema e aviso sobre `GT_*`.
- CHANGELOG `0.2.0`, versão em `pyproject.toml` e `__init__.py`, atualizar `CITATION.cff`.

---

## Fora do escopo desta etapa
- Engenharia de atributos, pipeline de ML, divisão prequencial, LSTM e avaliação: ficam no
  plano do pipeline (frente [DDM]). O simulador só gera dados brutos, mas o esquema de saída
  já foi desenhado para o pipeline (`MERCHANT_*` observáveis, `GT_*` para
  avaliação por cenário e por regime).
- Calibração com dados reais.
- Paralelização multi-processo (só se o benchmark da Fase 1 não atingir a meta).

## Riscos e decisões registradas
| tema | decisão |
|---|---|
| reproduzir o artigo | não é possível com a v0.1 (a ordem dos sorteios mudou na reescrita). A linha de base da dissertação será refeita sobre o SynCCFD |
| fraude por injeção | muda a semântica dos cenários 3 e 4 no v2. Justificativa na dissertação: o histórico legítimo do cliente é preservado, o que é mais realista e evita contaminar os atributos de agregação. **Validar com o orientador antes da Fase 3** |
| ordem das fases | *drift* → fraudes → comerciantes. As fases anteriores funcionam sem os comerciantes, o que mantém cada componente opcional e permite a ablação |
| limiar fixo do cenário 1 | substituído por quantil por categoria no v2 |
| taxa de fraude e escala | `rate_mode="proportional"` é o padrão nos presets novos |
| drift × sazonalidade | separados: sazonalidade é estacionária e não deve ser contada como drift |

---

## Como usar com o Claude Code

1. Copie este arquivo para `docs/plan-v0.2.md` no repositório e crie uma branch:
   `git checkout -b feat/v0.2-simulator`.
2. Execute **uma fase por vez**, de preferência numa sessão nova para cada uma. Prompt
   sugerido:

   > Leia `docs/plan-v0.2.md`. Implemente **somente a Fase N**, seguindo os princípios do
   > início do documento. Antes de escrever código, me mostre um plano curto dos arquivos que
   > vai criar ou alterar. Ao final, rode `ruff check .`, `mypy src` e `pytest` e confira cada
   > item dos critérios de aceite da Fase N, marcando o que passou. Por último, preencha a
   > subseção da Fase N em "Para o texto", no fim do documento, com o que foi realmente
   > implementado (decisões, parâmetros finais, figuras, números e desvios do plano). Escreva
   > em português, em frases completas, e não invente números: use só o que foi medido.

3. Na Fase 0, confira que o teste golden foi criado **antes** de qualquer mudança
   (`git log` deve mostrar o commit do golden primeiro).
4. Ao fim de cada fase, rode o notebook da fase e olhe os gráficos você mesmo: é a melhor
   verificação de que o comportamento faz sentido, e os gráficos vão para a dissertação.
5. Revise a subseção "Para o texto" da fase: ela precisa estar correta, porque o texto da
   dissertação vai ser escrito a partir dela.
6. Faça um PR por fase para `master`.

Ordem e dependências: **0 → 1 → 2 → 3 → 4 → 5**. Cada fase depende só das anteriores, e todas
ficam prontas antes da qualificação.

---

# Para o texto

> Preenchido **ao fim de cada fase**, com o que foi realmente feito. Não é o plano: se a
> implementação divergiu do plano, vale o que foi implementado, e a divergência é registrada
> em "Desvios do plano".
>
> **Figuras:** cada notebook salva as figuras em `docs/figures/<fase>/` em PDF (vetorial),
> com nome descritivo. Registre aqui o caminho de cada figura e uma legenda proposta.
>
> **Números:** todo número citado aqui vem de uma execução registrada (notebook, script ou
> teste). Informe a configuração usada (seed, nº de clientes, nº de dias).

## Onde cada fase entra na dissertação

| fase | cap. 4 (SynCCFD) | outros capítulos |
|---|---|---|
| 0 | 4.3 Modelo de simulação (cenários de fraude configuráveis); 4.7 Engenharia de software (teste de reprodutibilidade) | — |
| 1 | 4.2 Arquitetura; 4.3 Modelo de simulação; 4.7 Engenharia de software (desempenho, RNG) | — |
| 2 | 4.4 Componentes novos — *concept drift*; 4.6 Saídas e gabarito | cap. 2: tipos de *drift* e sazonalidade |
| 3 | 4.4 Componentes novos — fraudadores realistas; 4.6 Saídas e gabarito | cap. 2: *card testing* nos cenários de fraude |
| 4 | 4.4 Componentes novos — comerciantes; 4.3 Modelo de simulação (terminais → comerciantes) | — |
| 5 | 4.1 Visão geral e requisitos; 4.2 Arquitetura; 4.5 Configuração e API; 4.7 Engenharia de software; 4.8 Análise exploratória | — |

(As seções 4.1–4.8 seguem a estrutura do cap. 4 definida no `MAPA_DO_MESTRADO.md`.)

## Modelo de subseção (copiar para cada fase)

```markdown
## Fase N — <título>

**Status:** concluída em <data> · PR <link>

### O que foi implementado
<2 a 5 parágrafos, em prosa: o modelo/comportamento implementado, como funciona, por que
foi feito assim. Escrito para ser adaptado diretamente para a dissertação.>

### Decisões e justificativas
| decisão | alternativa considerada | por que esta |
|---|---|---|

### Parâmetros finais (padrões)
| parâmetro | valor padrão | significado |
|---|---|---|

### Equações / regras
<fórmulas e regras exatamente como implementadas, em LaTeX quando fizer sentido>

### Figuras geradas
| arquivo | legenda proposta | seção da dissertação |
|---|---|---|

### Números
| medida | valor | configuração |
|---|---|---|

### Testes que comprovam o comportamento
<lista dos testes relevantes e o que cada um garante>

### Desvios do plano
<o que mudou em relação ao plano e por quê; "nenhum" se não houve>

### Pendências / ideias
<o que ficou de fora ou pode ser melhorado>
```

## Fase 0 — Rede de segurança e configuração da fraude
*(a preencher)*

### Observações prévias sobre a v0.1 (registradas antes da implementação)

**O parâmetro `random_state` quase não afeta os dados gerados pela v0.1.** O motor legacy
fixa as sementes internamente, independentemente do valor recebido: os perfis de clientes
são gerados após `np.random.seed(0)` e os de terminais após `np.random.seed(1)`; as
transações de cada cliente são geradas após `random.seed(CUSTOMER_ID)` e
`np.random.seed(CUSTOMER_ID)`; e os sorteios da fraude usam `random_state=day`. Por isso, o
teste golden continua válido (ele só exige que a mesma configuração produza os mesmos
dados), mas a propriedade "sementes diferentes geram dados diferentes" só pode ser exigida
do motor v2. O comportamento do legacy é mantido como está, para preservar a compatibilidade.

**No legacy, os sorteios de um cenário de fraude não são independentes dos demais.** Cada
chamada de adulteração dos cenários 3 e 4 reinicializa o estado global (`random.seed(day)` e
`np.random.seed(day)`), e os sorteios seguintes dependem da sequência de chamadas feitas
antes. Assim, desligar um cenário com `FraudConfig(scenarios=...)` no caminho legacy pode
alterar os sorteios dos cenários que continuam ligados. O critério de aceite da Fase 0
(apenas os rótulos dos cenários ligados aparecem) não é afetado, mas as comparações de
ablação devem ser feitas no motor v2, que usa um fluxo aleatório separado por subsistema,
como já previsto na Fase 5.

## Fase 1 — Motor v2
*(a preencher)*

## Fase 2 — Customers com *concept drift*
*(a preencher)*

## Fase 3 — Fraudadores realistas
*(a preencher)*

## Fase 4 — Merchants
*(a preencher)*

## Fase 5 — Presets, documentação e release 0.2.0
*(a preencher)*

## Fechamento do [SIM]
*(preencher ao final de todas as fases)*
- Resumo do SynCCFD v0.2 em 1 parágrafo (base para o resumo e a introdução da dissertação).
- Configuração exata do dataset usado nos experimentos (preset, seed, tamanho) e suas
  estatísticas: nº de transações, taxa de fraude total e por cenário, % de clientes com
  *drift* por tipo, nº de comerciantes por categoria.
- Lista final de figuras do cap. 4, na ordem em que devem aparecer.
