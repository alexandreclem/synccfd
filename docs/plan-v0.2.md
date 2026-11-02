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
8. O notebook `examples/simulator.ipynb` continua funcionando. Ele gera as figuras do
   simulador usadas no TCC. Toda fase precisa manter esse notebook executável; na Fase 5
   ele passa a exportar as figuras da dissertação.

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
  *Corrigido na implementação: contagem diária ~ `Poisson(fração × população)`, sem
  mínimo (ver "Para o texto", Fase 0).*
- `FraudSimulator` passa a receber o `FraudConfig` e ler dele todos os valores fixos.
- Validação em `__post_init__`: frações entre 0 e 1, durações > 0, cenários conhecidos.

### 0.3 Critérios de aceite
- [x] Golden test passa.
- [x] Teste: `rate_mode="proportional"` mantém a taxa de fraude estável (±30% relativo)
      entre `n_customers=500` e `n_customers=5000` (mesmos dias).
      *Verificado com as frações padrão sobre a média de 20 seeds e 60 dias, após a
      troca da regra por Poisson (ver "Para o texto", Fase 0).*
- [x] Teste: `FraudConfig(scenarios=(1,))` gera só `TX_FRAUD_SCENARIO ∈ {0, 1}`.
- [x] CHANGELOG e README atualizados (seção "Fraud configuration").

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
- [x] Golden test continua passando (legacy intocado).
- [x] Teste estatístico v2 × legacy, com os mesmos parâmetros (`n_customers=1000`,
      `nb_days=30`), comparando:
      transações por cliente-dia, valor médio, proporção CNP, histograma de horário (teste
      KS, p > 0,01) e taxa de fraude por cenário (diferença relativa < 20%).
      *Horário: critério trocado do p-valor para o tamanho da diferença (estatística de
      KS $D < 0{,}02$, separadamente para CP e CNP, semente fixa); ver "Para o texto",
      Fase 1.*
- [x] Reprodutibilidade: mesma seed → mesmos dados; seeds diferentes → dados diferentes.
- [x] Benchmark simples em `scripts/benchmark.py` registrado no PR.
      *Resultado registrado no "Para o texto"; vai para a descrição do PR.*
- [x] Teste que executa `examples/simulator.ipynb` de ponta a ponta com parâmetros
      reduzidos (por exemplo, `nbclient` ou `pytest --nbmake`), marcado como `slow` e
      rodando no CI.
      *Passa localmente; o job `slow` do CI já o executa, falta confirmar no primeiro
      push.*
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
- [x] Golden test passa.
- [x] Teste: clientes `none` têm a média móvel de valor estável (sem tendência
      significativa); clientes `sudden` mostram mudança de nível em `t0`.
      *Critério por tamanho do efeito: variação do valor normalizado ao longo do
      período menor que 2%; correlação maior que 0,85 entre a mudança de nível
      observada em `t0` e a configurada (ver "Para o texto", Fase 2).*
- [x] Teste: em `gradual`, a fração de transações com `GT_REGIME == "B"` acompanha `w(t)`.
- [x] Teste: `customer_fraction=0` + sazonalidade desligada gera a mesma distribuição da
      Fase 1.
      *Mais forte que o pedido: gera exatamente os mesmos dados.*
- [x] Notebook `examples/drift.ipynb`: um painel por tipo de drift (valor e frequência de
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

### 3.1 Mudança de mecanismo (só no motor v2): injeção em vez de adulteração
Na v0.1, os cenários 3 e 4 **alteram** transações legítimas do cliente (mudam
local, tipo e multiplicam o valor por 5), e o cenário 1 **rotula** como fraude as
transações legítimas acima de R$ 220. No v2, nos cenários 1, 3, 4 e 5, o fraudador **gera
transações novas**, que se somam às do cliente:
- o histórico legítimo do cliente continua íntegro, como acontece na realidade;
- valor, horário e comerciante da fraude vêm do comportamento do fraudador, não do cliente;
- `FraudConfig(engine="v2")` liga esse mecanismo; `engine="legacy"` mantém a adulteração.
  *Implementado como `FraudConfig(mechanism="inject")` / `mechanism="tamper"`, para não
  confundir com o motor do `DatasetGenerator` (ver "Para o texto", Fase 3).*

*O cenário 2 é a exceção, como na tabela abaixo: ele rotula as compras CP legítimas
feitas no terminal comprometido, sem alterá-las.*

### 3.2 Cenários no v2

| # | cenário | comportamento do fraudador (sem comerciantes) | enriquecimento com comerciantes (Fase 4) |
|---|---|---|---|
| 1 | valor anômalo | **injetado**, como os demais: o fraudador faz uma compra **nova** com valor muito acima do histórico do próprio cliente, isto é, acima de um quantil alto (`q`) do valor das compras recentes dele, multiplicado por um fator configurável. Transações legítimas **nunca** são rotuladas como fraude. Substitui o limiar fixo de R$ 220, que rotulava o *drift* legítimo como fraude (ver "Para o texto", Fase 2); o legacy mantém o limiar fixo | a compra fraudulenta cai em categorias de valor alto (eletrônicos, viagem); o quantil continua sendo o do histórico do cliente |
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
- [x] Golden test passa.
- [x] Teste: no v2, o número de transações legítimas de cada cliente não muda quando a
      fraude é ligada (injeção, não adulteração).
      *Mais forte que o pedido: as transações legítimas são idênticas, coluna a coluna, às
      geradas com a fraude desligada.*
- [x] Teste: no v2, nenhuma transação legítima é rotulada como fraude pelo cenário 1, e
      cada fraude do cenário 1 tem valor acima de fator × quantil `q` das compras recentes
      do cliente. Com *drift* ligado, a fração do cenário 1 vinda do regime B fica próxima
      da fração de transações do regime B (documentar os dois valores).
      *Medido: 13,5% e 13,5% (ver "Para o texto", Fase 3); no teste, diferença menor que
      0,05.*
- [x] Teste: cenário 5 só tem valores ≤ R$ 5 e `TX_TYPE == "CNP"`; quando escala, o
      `GT_CAMPAIGN_ID` é compartilhado com o cenário 3 seguinte.
- [x] Teste: as transações do cenário 4 acontecem em até 24 h a partir de `τ` e somam até o
      limite configurado.
- [x] Teste: `intensity_schedule` em degrau muda a taxa diária do cenário na data prevista.
- [x] Taxa de fraude total do preset `realistic` entre 0,2% e 1% das transações
      (configurável; documentar o valor obtido).
      *Os presets são da Fase 5; aqui, `FraudConfig.realistic()`, que o preset vai usar.
      Medido: 0,44% (ver "Para o texto", Fase 3).*
- [x] Teste: todos os cenários funcionam com os comerciantes desligados.
- [x] Notebook `examples/frauds.ipynb`: taxa por cenário ao longo do tempo, exemplo de
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
- [ ] Teste: com comerciantes ligados, as fraudes do cenário 1 caem nas categorias de valor
      alto, e o cenário 3 concentra compras nas categorias configuradas.
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
- Atualizar `examples/simulator.ipynb` para gerar as figuras da dissertação: um parâmetro
  escolhe o preset (`legacy` ou `realistic`) e o tamanho; cada figura é salva em PDF em
  `docs/figures/exploratoria/` com os nomes da lista abaixo; mantém as mesmas figuras do
  TCC, na mesma ordem, agora com os dados da v0.2, acrescentando o cenário 5 e, com
  comerciantes, a visão por categoria. O diagrama do fluxo do simulador é redesenhado
  junto com o diagrama de arquitetura.

  Nomes das figuras, na ordem do notebook, tirados das exportações comentadas da versão
  local do notebook usada no TCC (branch `backup/v0.2-old-history`):
  1. `transactions_table` (no TCC era PNG, exportada com `dfi.export`)
  2. `simulator_txs_time_per_type`
  3. `simulator_txs_time_and_amounts`
  4. `simulator_sampled_locations`
  5. `simulator_city_radius`
  6. `populous_cities_distribution`
  7. `fraud_scenarios_count`
  8. `transactions_per_day`
  9. `fraud_scenarios_type_and_amounts`
  10. `fraud_scenarios_bill_shipp_dist2`

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
| fraude por injeção | muda a semântica dos cenários 1, 3 e 4 no v2. Justificativa na dissertação: o histórico legítimo do cliente é preservado, o que é mais realista e evita contaminar os atributos de agregação |
| ordem das fases | *drift* → fraudes → comerciantes. As fases anteriores funcionam sem os comerciantes, o que mantém cada componente opcional e permite a ablação |
| limiar fixo do cenário 1 | no v2, substituído por fraude injetada: compra acima de fator × quantil alto das compras recentes do próprio cliente; transações legítimas nunca viram fraude. O legacy mantém o limiar fixo. Motivo: com o limiar fixo, 86% do cenário 1 vinha de clientes no regime B (Fase 2) |
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

**Status:** concluída em 2026-11-02 · integrada ao master pelo PR #1 (`d32326d`)

### O que foi implementado

Antes de qualquer alteração no código, o comportamento da v0.1.0 foi congelado num teste
de regressão "golden". O script `scripts/make_golden.py` gera, com a configuração padrão,
um conjunto pequeno (100 clientes, 200 terminais, 15 dias, início em 2023-01-01,
`random_state=42`) e grava os três DataFrames em `tests/golden/data/` no formato Parquet.
O teste `tests/golden/test_legacy_reproducibility.py` regenera o conjunto e o compara
com os arquivos gravados, coluna por coluna. IDs, tempos em segundos e dias, tipos de
transação, rótulos de fraude e terminais disponíveis são comparados por igualdade exata.
As colunas de ponto flutuante (coordenadas, valores das transações e parâmetros dos
perfis de clientes) são comparadas com tolerância
relativa de $10^{-12}$ e tolerância absoluta zero. As datas são convertidas para a
mesma unidade (`datetime64[ns]`) antes da comparação exata. O commit do teste golden
(`b27cfa7`) antecede todos os commits que alteram o código da biblioteca.

A igualdade exata em todas as colunas, prevista no plano, não se sustenta entre
plataformas. Gerado no Linux (Ubuntu 22.04, Python 3.10, pandas 2.3.3), o conjunto tem
exatamente as mesmas linhas, os mesmos valores, tipos e rótulos (741 transações
fraudulentas nos dois casos), mas as coordenadas diferem no último bit, em até
$1{,}4 \times 10^{-14}$, porque as bibliotecas matemáticas do Windows e do Linux
arredondam de forma diferente. Além disso, a unidade inferida para `TX_DATETIME` depende
da versão do pandas (`[ms]` no arquivo gravado com o pandas 3.0.3, `[ns]` no pandas
2.3.3). A tolerância de $10^{-12}$ é cerca de cem vezes maior que a diferença observada e
ainda detecta mudanças reais: um teste confirma que `amount_multiplier=5.0001` (variação
relativa de $2 \times 10^{-5}$ no valor) é rejeitado.

Em seguida, todos os parâmetros da fraude que estavam fixos no código passaram a ser
lidos de um objeto de configuração imutável, `FraudConfig`, definido em
`synccfd/simulator/config.py`. Ele permite escolher quais cenários aplicar e ajustar o
limiar de valor do cenário 1, o número de terminais e clientes comprometidos por dia, as
durações de cada cenário, a fração de transações adulteradas no cenário 3, o
multiplicador de valor dos cenários 3 e 4 e a probabilidade de CNP no cenário 4. A
configuração é passada por `DatasetGenerator(fraud=...)` ou `FraudSimulator(config=...)`.
Os valores padrão reproduzem exatamente a v0.1.0, o que o teste golden confirma.

Foi acrescentado um segundo modo de cálculo da taxa de comprometimento,
`rate_mode="proportional"`. Nele, o número de terminais (cenário 2) ou de clientes
(cenários 3 e 4) comprometidos em cada dia é sorteado de uma distribuição de Poisson
com média $f \cdot N$, sendo $f$ a fração configurada e $N$ o tamanho da população. O
número esperado de comprometimentos por dia é, portanto, proporcional à população, e um
dia pode não ter nenhum. No modo `"absolute"` (padrão, igual à v0.1), esse número é
fixo, de modo que a taxa de fraude cresce à medida que o conjunto diminui: com 100
clientes e 15 dias, 26,2% das transações são fraudulentas, contra 0,59% com 5 000
clientes e 10 dias. As frações padrão do modo proporcional foram escolhidas para que, em
média, coincidam com as contagens absolutas no tamanho padrão do gerador (10 000 clientes
e 20 000 terminais): $3 \times 10^{-4} \cdot 10\,000 = 3$ clientes e
$10^{-4} \cdot 20\,000 = 2$ terminais por dia.

As contagens diárias do modo proporcional são sorteadas uma única vez, antes da
aplicação dos cenários, com um gerador local (`numpy.random.Generator`). Cada cenário
recebe um fluxo aleatório próprio, derivado de `np.random.SeedSequence(random_state)`,
de modo que desligar um cenário não altera as contagens dos outros e o estado aleatório
global usado pelo motor legacy não é tocado. No motor legacy, a semente passa a
controlar essas contagens; a escolha de *quais* terminais e clientes são comprometidos
continua dependendo só do dia, como na v0.1.

Em conjuntos pequenos, o modo proporcional mantém a taxa *esperada*, mas cada execução
varia mais: com 500 clientes, a média é de 0,15 vítima nova por dia em cada um dos
cenários 3 e 4, e a maioria dos dias não tem nenhuma. Com 60 dias e 20 sementes, a taxa
média de fraude foi de 0,508% com 500 clientes (desvio-padrão de 0,110% entre sementes) e
de 0,543% com 5 000 clientes (desvio-padrão de 0,029%), uma diferença relativa de 6,4%.

Para que `mypy src` passasse, como exige o princípio 7, foram corrigidos os 36 erros de
tipagem que já existiam na v0.1.0. As correções foram apenas de anotação (tipos
`Optional`, anotação de atributos inicializados com `None` e de variáveis locais) e não
alteram o comportamento, o que o teste golden confirma. Também foi corrigida uma falha da
v0.1 nas filas dos cenários 3 e 4 (ver "Decisões e justificativas").

### Decisões e justificativas
| decisão | alternativa considerada | por que esta |
|---|---|---|
| golden em três arquivos Parquet | pickle (`.pkl`) | pickles podem não abrir em outras versões do pandas, e o CI testa Python 3.9 a 3.12 |
| comparar o DataFrame regenerado após passar pelo mesmo ciclo de escrita e leitura em Parquet | comparar direto com o arquivo lido | o Parquet devolve a coluna de listas `available_terminals` como arrays; o ciclo torna a comparação equivalente sem relaxar a igualdade exata |
| floats com `rtol=1e-12` e `atol=0`, datas normalizadas para `[ns]`, demais colunas exatas | `check_exact=True` em tudo, como no plano | a igualdade exata falha no Linux só pelo último bit das coordenadas e pela unidade da data; rótulos, valores, IDs e tipos continuam exatos |
| sortear os clientes dos cenários 3 e 4 numa única chamada, `sample(n=n3+n4, random_state=day)`, e dividir depois | um sorteio por cenário | preserva a sequência de sorteios da v0.1 |
| `int(n * s3_tamper_fraction)` no lugar de `int(n / 3)` | manter a divisão por 3 fixa | verificado que `int(n * (1/3)) == int(n / 3)` para todo `n < 3 000 000`, então o padrão é idêntico |
| contagem diária proporcional sorteada de $\mathrm{Poisson}(f \cdot N)$, sem mínimo | $\max(1, \mathrm{round}(f \cdot N))$, como no plano original | o mínimo de 1 tornava a taxa dependente do tamanho quando $f \cdot N < 1$ (ver "Desvios do plano"); a Poisson mantém a média exata para qualquer $N$ |
| um fluxo aleatório local por cenário, derivado de `SeedSequence(random_state)` | usar o estado global do `numpy` | não altera os sorteios do legacy e torna os cenários independentes entre si (princípio 3) |
| retirar elementos das filas dos cenários 3 e 4 só se não estiverem vazias (registrado como correção de bug no CHANGELOG) | manter o `popleft()` incondicional | na v0.1, um dia sem clientes disponíveis suficientes não acrescentava nada à fila, mas um elemento era retirado depois, o que podia gerar `IndexError`. Com os padrões, quando a v0.1 não falhava, o resultado é idêntico |
| `FraudConfig` imutável (`frozen=True`), com `scenarios` sempre guardado como tupla | dataclass mutável | a configuração efetiva será serializada em `config.json` (Fase 5); imutabilidade evita alterá-la por engano depois de criada |
| regra de contagem testada diretamente (rápido); teste de ponta a ponta marcado como `slow`, excluído da execução padrão (`addopts = -m "not slow"`) e executado num job separado do CI | só o teste de ponta a ponta | o teste de ponta a ponta leva cerca de 4 min; a regra de contagem pode ser verificada em milissegundos |

### Parâmetros finais (padrões)
| parâmetro | valor padrão | significado |
|---|---|---|
| `scenarios` | `(1, 2, 3, 4)` | cenários aplicados |
| `rate_mode` | `"absolute"` | `"absolute"`: contagem fixa por dia; `"proportional"`: contagem diária sorteada de $\mathrm{Poisson}(f \cdot N)$ |
| `s1_amount_threshold` | 220,0 | transações acima deste valor são fraude (cenário 1) |
| `s2_terminals_per_day` | 2 | terminais comprometidos por dia (modo absoluto) |
| `s2_terminal_fraction_per_day` | $10^{-4}$ | fração de terminais comprometidos por dia (modo proporcional) |
| `s2_duration_days` | 28 | dias em que um terminal fica comprometido |
| `s3_customers_per_day`, `s4_customers_per_day` | 3, 3 | clientes comprometidos por dia (modo absoluto) |
| `s3_customer_fraction_per_day`, `s4_customer_fraction_per_day` | $3 \times 10^{-4}$, $3 \times 10^{-4}$ | fração de clientes comprometidos por dia (modo proporcional) |
| `s3_duration_days`, `s4_duration_days` | 14, 5 | dias em que um cliente fica comprometido |
| `s3_tamper_fraction` | 1/3 | fração das transações do cliente do cenário 3 que viram fraude |
| `amount_multiplier` | 5,0 | multiplicador do valor das transações adulteradas (cenários 3 e 4) |
| `s4_cnp_prob` | 0,5 | probabilidade de uma transação adulterada do cenário 4 ser CNP |

### Equações / regras
- Cenário 1: $\mathrm{fraude}(x) = [\,\mathrm{valor}(x) > s_1\,]$, com $s_1$ = `s1_amount_threshold`.
- Comprometimentos no dia $d$, para o cenário $k \in \{2, 3, 4\}$ e população $N$
  (terminais para $k=2$, clientes para $k=3,4$):
  $$
  n_k(d) = \begin{cases}
    c_k & \text{se } \texttt{rate\_mode} = \texttt{"absolute"} \\
    \sim \mathrm{Poisson}(f_k \cdot N) & \text{se } \texttt{rate\_mode} = \texttt{"proportional"}
  \end{cases}
  $$
  com sorteios independentes entre dias e entre cenários, e $n_k(d) = 0$ se o cenário
  $k$ não está em `scenarios`. Assim, $\mathbb{E}[n_k(d)] / N = f_k$ para qualquer $N$.
- No dia $d$, os $n_3(d) + n_4(d)$ clientes são sorteados numa única chamada entre os que
  não estão comprometidos; os $n_3(d)$ primeiros vão para o cenário 3 e os demais para o
  cenário 4. O sorteio só ocorre se houver pelo menos $n_3(d) + n_4(d)$ clientes
  disponíveis.

### Figuras geradas
Nenhuma (a Fase 0 não tem notebook).

### Números
| medida | valor | configuração |
|---|---|---|
| transações do conjunto golden | 2 826 | 100 clientes, 200 terminais, 15 dias, `random_state=42`, padrões |
| taxa de fraude do conjunto golden | 26,2% (cenário 1: 2; 2: 193; 3: 209; 4: 337 transações) | idem |
| taxa de fraude, modo absoluto | 5,06% (500 clientes) × 0,59% (5 000 clientes) | terminais = 2 × clientes, 10 dias, início 2023-01-01, `random_state=42` |
| taxa de fraude com a regra original do plano, $\max(1, \mathrm{round}(f \cdot N))$, frações padrão | 1,78% (500) × 0,23% (5 000) | idem (medido antes da correção, para justificá-la) |
| taxa de fraude média, modo proporcional (Poisson), frações padrão | 0,508% (500 clientes; DP 0,110%) × 0,543% (5 000 clientes; DP 0,029%); diferença relativa de 6,4% | terminais = 2 × clientes, 60 dias, início 2023-01-01, 20 sementes (0 a 19); DP = desvio-padrão entre sementes |
| taxa média por cenário, mesma configuração | 500 clientes: cenário 1: 0,074%; 2: 0,166%; 3: 0,113%; 4: 0,156%. 5 000 clientes: 1: 0,057%; 2: 0,225%; 3: 0,117%; 4: 0,145% | idem |
| maior diferença entre Windows e Linux no conjunto golden | $1{,}4 \times 10^{-14}$ (em `TX_SHIPP_LONG`; demais coordenadas até $7{,}1 \times 10^{-15}$); nenhuma diferença nas colunas não float | Windows 11, Python 3.11.4, pandas 3.0.3 × Ubuntu 22.04 (WSL), Python 3.10.12, pandas 2.3.3 |
| tempo de geração das transações legítimas (motor legacy) | 6,1 s (500 clientes, 9 790 transações) e 47,5 s (5 000 clientes, 96 566 transações) | 10 dias; Windows 11, Python 3.11.4; referência para o benchmark da Fase 1 |

### Testes que comprovam o comportamento
- `tests/golden/test_legacy_reproducibility.py`:
  - `test_matches_v0_1_0`: a configuração padrão reproduz a v0.1.0 (três DataFrames;
    colunas não float exatas, floats com `rtol=1e-12`);
  - `test_detects_small_behaviour_change`: com `amount_multiplier=5.0001`, a comparação
    falha na coluna `TX_AMOUNT`, isto é, a tolerância não esconde mudanças reais.
- `tests/simulator/test_fraud_config.py`:
  - `test_defaults_match_v0_1`: os padrões correspondem às constantes da v0.1, incluindo
    `rate_mode="absolute"`;
  - `test_invalid_values_raise`: validação de cenários desconhecidos ou repetidos, modo
    inválido, frações fora de $[0, 1]$, durações não positivas, contagens negativas,
    limiar negativo e multiplicador não positivo;
  - `test_absolute_counts_ignore_population` e `test_disabled_scenarios_have_zero_count`:
    contagens do modo absoluto e de cenários desligados;
  - `test_proportional_counts_have_poisson_mean`: para $N \in \{500, 5\,000, 50\,000\}$,
    média e variância das contagens diárias iguais a $f \cdot N$ (200 000 dias);
  - `test_proportional_counts_can_be_zero`: com $f \cdot N = 0{,}15$, a fração de dias
    sem vítima é $e^{-0{,}15}$ (sem mínimo de 1);
  - `test_simulator_counts_are_reproducible_and_independent`: mesma semente, mesmas
    contagens; sementes diferentes, contagens diferentes; desligar o cenário 3 não muda
    as contagens do cenário 4;
  - `test_only_selected_scenarios_are_labelled`: para vários subconjuntos de cenários,
    só os rótulos ligados aparecem e `TX_FRAUD` é coerente com `TX_FRAUD_SCENARIO`;
  - `test_scenario_1_only_through_dataset_generator`: `FraudConfig(scenarios=(1,))` via
    `DatasetGenerator` gera só os rótulos 0 e 1;
  - `test_amount_threshold_is_configurable`: o limiar do cenário 1 é respeitado;
  - `test_proportional_rate_is_stable_across_sizes` (`slow`, fora da execução padrão):
    com as frações padrão, a taxa média de fraude em 20 sementes e 60 dias difere menos
    de 30% entre 500 e 5 000 clientes.

### Desvios do plano
- **Correção do plano:** a regra do modo proporcional prevista em 0.2,
  $\max(1, \mathrm{round}(f \cdot N))$, foi substituída por uma contagem diária sorteada
  de $\mathrm{Poisson}(f \cdot N)$, sem mínimo. Com a regra original e as frações
  padrão, a taxa não ficava estável: com 500 clientes, $3 \times 10^{-4} \cdot 500 =
  0{,}15$ virava 1 cliente por dia, cerca de 6,7 vezes a fração desejada, e as taxas
  medidas foram de 1,78% (500 clientes) e 0,23% (5 000 clientes). Com a Poisson, as
  frações padrão foram mantidas e a diferença caiu para 6,4%.
- O critério de estabilidade é verificado sobre a média de 20 sementes e 60 dias, não
  sobre uma única execução, porque em conjuntos pequenos a taxa realizada varia bastante
  entre sementes.
- O golden foi gravado como três arquivos Parquet (`v0_1_0_small_<nome>.parquet`), não
  um único arquivo.
- O golden não usa `check_exact=True` em todas as colunas: floats com `rtol=1e-12` e
  datas normalizadas para a mesma unidade (ver "O que foi implementado").
- Fora do escopo original: correção dos 36 erros de `mypy` preexistentes (commit
  `9fb3c15`, apenas anotações); remoção do `continue-on-error` do `mypy` no CI; job
  separado no CI para os testes `slow`; CI disparado também em push para branches
  `feat/**`; runner do CI fixado em `ubuntu-24.04`, porque o `ubuntu-latest` (26.04) não
  oferece mais Python 3.9; exemplo do README reformatado para o ruff 0.17, que passou a
  formatar blocos de código em Markdown.
- Os commits da Fase 0 foram reaplicados sobre o `origin/master` (o repositório no GitHub
  foi recriado com outro histórico); por isso os hashes citados aqui são os da branch
  reaplicada.

### Pendências / ideias
- O mypy 2.x avisa que `python_version = "3.9"` não é mais suportado na configuração.
  Não impede a checagem, mas pode exigir fixar a versão do mypy no CI do Python 3.9.
- Na Fase 1, o motor v2 deve sortear também *quais* terminais e clientes são
  comprometidos com o gerador local, e não com `random_state=day`.
- Escolher, nos presets da Fase 5, frações proporcionais adequadas aos tamanhos usados
  nos experimentos.

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

**Status:** concluída em 2026-11-02 · integrada ao master pelo PR #2 (`d4bc73c`)

### O que foi implementado

O código da v0.1 foi movido, sem alteração de lógica, para o pacote
`synccfd.simulator.legacy` (`legacy/generator.py` e `legacy/fraud.py`). O
`DatasetGenerator`, em `simulator/generator.py`, passou a escolher o motor pelo
parâmetro `engine`: `"legacy"` executa o código da v0.1; `"v2"` executa o motor novo;
`"auto"` (padrão) escolhe o legacy enquanto nenhum componente exclusivo do v2 estiver
ligado, o que ainda não existe nesta fase. Os caminhos de importação antigos
(`synccfd.simulator.generator.DatasetGenerator`, `synccfd.simulator.fraud.FraudSimulator`
e as importações do pacote raiz) continuam funcionando, e o teste golden confirma que o
legacy segue reproduzindo a v0.1.0.

O motor v2 (`synccfd.simulator.engine`) implementa o mesmo modelo da v0.1, mas usa
apenas geradores locais (`numpy.random.Generator`), nunca o estado aleatório global. Toda
a aleatoriedade vem de um único `SeedSequence(random_state)`, do qual se derivam fluxos
com nome e posição fixos (`customers`, `terminals`, `merchants`, `transactions`, `drift`
e `fraud`). Como a posição de cada fluxo é fixa, acrescentar ou ligar um componente nas
fases seguintes não altera os sorteios dos demais. Os subfluxos (um por cliente, um por
cenário de fraude) são derivados de forma pura, pela chave de derivação do
`SeedSequence`: o fluxo do cliente $i$ não depende de quantos clientes existem.

A geração de transações é vetorizada por cliente: para cada cliente, todos os dias são
sorteados de uma vez, a partir do fluxo próprio daquele cliente. Assim, as transações de
um cliente não dependem dos outros clientes, o que permitirá, na Fase 2, ligar o
*drift* em alguns clientes sem alterar as transações dos demais. Os cenários de fraude 1
a 4 têm a mesma semântica do legacy (adulteração das transações do cliente), e cada
cenário sorteia a partir do seu subfluxo do fluxo `fraud`. As funções geográficas
também foram vetorizadas (`location/sampling.py`): pontos aleatórios ao redor de um
centro, com rejeição dos pontos fora do Brasil por um teste de pertinência vetorizado
do `shapely` 2, e distâncias de haversine calculadas em blocos para encontrar os
terminais ao alcance de cada cliente. O `LocationFeatures` ganhou um parâmetro opcional
`rng` nos métodos de geração; sem ele, o comportamento do legacy é mantido.

O v2 corrige, de forma explícita, um efeito colateral da v0.1: um horário sorteado fora
do dia é sorteado novamente, em vez de a transação ser descartada. Na v0.1, cerca de 3%
das transações CP eram descartadas, o que reduzia a frequência diária abaixo do
parâmetro de Poisson e aumentava um pouco a proporção de CNP. Além disso, no v2 as
transações são reordenadas cronologicamente depois da aplicação das fraudes (o cenário 4
desloca transações no tempo) e o `TRANSACTION_ID` passa a ser a posição cronológica.

Com 10 000 clientes, 20 000 terminais e 180 dias, o v2 gera 3,59 milhões de transações
em 19,1 s, abaixo da meta de 2 min. Com 1 000 clientes e 180 dias, o legacy leva 67,7 s
e o v2 leva 1,1 s, cerca de 60 vezes menos. A comparação estatística com o legacy
(1 000 clientes, 30 dias) não encontrou diferenças além das esperadas pela correção do
descarte de horários.

### Decisões e justificativas
| decisão | alternativa considerada | por que esta |
|---|---|---|
| um fluxo aleatório por cliente para as transações | sortear todas as transações de todos os clientes de uma vez | com um vetor global, mudar os parâmetros de um cliente (por exemplo, pelo *drift* da Fase 2) deslocaria os sorteios de todos os outros; com fluxos por cliente, os clientes sem *drift* ficam idênticos, o que permite comparar cenários com os mesmos números aleatórios |
| fluxos nomeados em posição fixa, derivados pela chave do `SeedSequence` (funções puras) | `SeedSequence.spawn`, que guarda estado | `spawn` devolve filhos diferentes a cada chamada; a derivação pela chave dá sempre o mesmo fluxo para o mesmo nome, independentemente da ordem das chamadas |
| horário inválido é sorteado novamente | descartar a transação, como na v0.1 | a frequência diária fica fiel ao parâmetro de Poisson (previsto no plano) |
| reordenar as transações e reatribuir o `TRANSACTION_ID` depois das fraudes | manter a ordem anterior, como na v0.1 | o cenário 4 desloca transações no tempo; para o pipeline prequencial, a ordem cronológica precisa valer em todo o conjunto |
| destino de pontos pela fórmula esférica (raio médio de 6 371,0088 km) | `geopy` (elipsoide WGS84), ponto a ponto | permite vetorizar; a distância continua uniforme em $[0, r]$ e o ângulo uniforme, como na v0.1, e a diferença entre esfera e elipsoide é desprezível em raios de até 12 km |
| comparação v2 × legacy em três camadas (modelo de transações com os mesmos perfis; fraude sobre as mesmas transações; ponta a ponta) | só a comparação ponta a ponta | separa a variação do modelo de transações da variação de quais clientes e cidades foram sorteados, que é grande com 1 000 clientes |
| KS do horário separado por tipo (CP e CNP) | KS sobre todas as transações | a mistura CP/CNP depende do `cnp_prob` dos clientes sorteados; por tipo, as duas distribuições são, em teoria, idênticas |
| teste do notebook substitui os tamanhos em memória | criar parâmetros no notebook | o arquivo do notebook do TCC fica intocado; o teste falha se algum dos trechos substituídos deixar de existir |
| `shapely>=2.0` e `scipy` (dev) | manter `shapely>=1.7`; implementar o KS à mão | o v2 usa o teste de pertinência vetorizado do `shapely` 2; o `scipy` fornece o KS de duas amostras |

### Parâmetros finais (padrões)
Os parâmetros do modelo são os mesmos da v0.1 (ver `engine/transactions.py` e
`engine/customers.py`); a Fase 1 não cria parâmetros novos de modelo.

| parâmetro | valor padrão | significado |
|---|---|---|
| `engine` | `"auto"` | `"legacy"`, `"v2"` ou `"auto"` (legacy enquanto não houver componente exclusivo do v2) |
| fluxos aleatórios | `customers`, `terminals`, `merchants`, `transactions`, `drift`, `fraud` | ordem fixa; nomes só podem ser acrescentados ao fim |

### Equações / regras
Para cada cliente $c$, com fluxo próprio, e cada dia $d \in \{0, \dots, D-1\}$:
- $N_{c,d} \sim \mathrm{Poisson}(\lambda_c)$, com $\lambda_c$ = `mean_nb_tx_per_day`;
- valor $v \sim \mathcal{N}(\mu_c, \sigma_c)$; se $v < 0$, $v \sim \mathcal{U}(0, 2\mu_c)$;
  arredondado para centavos;
- tipo CNP com probabilidade $p_c$ = `cnp_prob` (sempre CNP se o cliente não tiver
  terminal ao alcance);
- horário do dia $t$ (em segundos, truncado para inteiro): CP,
  $t \sim \mathcal{N}(43\,200;\ 20\,000)$; CNP, com probabilidade 1/2,
  $t \sim \mathcal{N}(36\,000;\ 12\,500)$, ou $t \sim \mathcal{N}(72\,000;\ 12\,500) \bmod 86\,400$;
  se $t \notin [0, 86\,400]$, $t$ é sorteado novamente (no legacy, a transação é descartada);
- terminal: CP, uniforme entre os terminais a até 10 km do endereço de cobrança; CNP,
  uniforme entre todos;
- endereço de entrega: CP, terminal (0,8), cobrança (0,15) ou ponto aleatório a até
  10 km da cobrança (0,05); CNP, cobrança (0,9) ou ponto aleatório (0,1).

Ponto aleatório ao redor de $(\varphi_1, \lambda_1)$: distância $s \sim \mathcal{U}(0, r)$,
rumo $\theta \sim \mathcal{U}(0°, 360°)$, $\delta = s / R$, com $R = 6\,371{,}0088$ km;
$\varphi_2 = \arcsin(\sin\varphi_1 \cos\delta + \cos\varphi_1 \sin\delta \cos\theta)$ e
$\lambda_2 = \lambda_1 + \operatorname{atan2}(\sin\theta \sin\delta \cos\varphi_1,\ \cos\delta - \sin\varphi_1 \sin\varphi_2)$;
pontos fora do polígono do Brasil são sorteados de novo.

### Figuras geradas
Nenhuma. O notebook `examples/simulator.ipynb` continua gerando as figuras do TCC com o
motor legacy (princípio 8).

### Números
| medida | valor | configuração |
|---|---|---|
| tempo de geração, v2 | 19,1 s (3 589 537 transações) | 10 000 clientes, 20 000 terminais, 180 dias, `random_state=42`; Windows 11, Python 3.11.4, Intel 64 Family 6 Model 183; `scripts/benchmark.py` |
| tempo de geração, legacy × v2 | 67,7 s (354 028 transações) × 1,1 s (352 658 transações) | 1 000 clientes, 2 000 terminais, 180 dias; idem |
| parcela do tempo no cálculo dos terminais ao alcance | cerca de 14 s de 30 s (execução com *profiler*) | 10 000 clientes, 20 000 terminais, 180 dias |
| (A) transações por cliente-dia, legacy × v2 com os mesmos perfis | 1,9646 × 1,9937 (+1,5%) | 1 000 clientes, 2 000 terminais, 30 dias; perfis do legacy com `random_state=42`; fluxos do v2 com semente 1 |
| (A) valor médio | 54,79 × 54,56 (−0,4%); KS p = 0,45 | idem |
| (A) proporção de CNP | 0,672 × 0,664 | idem |
| (A) KS do horário do dia | CP: D = 0,0153 (p = 0,019); CNP: D = 0,0062 (p = 0,43) | idem; D é a maior distância entre as duas distribuições empíricas acumuladas |
| (B) taxa de fraude por cenário, legacy × média do v2 em 10 sementes | 1: 0,056% × 0,056%; 2: 1,469% × 1,479% (+0,6%); 3: 1,086% × 1,065% (−2,0%); 4: 1,407% × 1,389% (−1,2%) | mesmas transações legítimas do legacy (1 000 clientes, 30 dias); `FraudConfig()` |
| (C) KS dos perfis de clientes, legacy × v2 | `mean_amount` p = 0,50; `mean_nb_tx_per_day` p = 0,57; `cnp_prob` p = 0,34; `billing_lat` p = 0,055; `billing_long` p = 0,37 | 1 000 clientes, `random_state=42` nos dois motores |
| (C) transações por cliente-dia e taxa de fraude, ponta a ponta | 1,9646 × 1,9591; 4,02% × 4,24% | 1 000 clientes, 2 000 terminais, 30 dias, `random_state=42` |
| (C) média de terminais ao alcance por cliente | legacy: 19,95; v2: 18,02 (sementes 0 a 7: de 17,9 a 22,4) | 1 000 clientes, 2 000 terminais |
| conjunto do tamanho do golden no v2 | 2 907 transações (legacy: 2 826), 27,2% de fraude (legacy: 26,2%), 0,42 s | 100 clientes, 200 terminais, 15 dias, `random_state=42` |
| execução do notebook com tamanhos reduzidos | 40,7 s | 1 000 clientes, 2 000 terminais, 20 dias; amostras de 2 000 transações nos gráficos |

### Testes que comprovam o comportamento
- `tests/golden/test_legacy_reproducibility.py`: o legacy continua reproduzindo a v0.1.0
  depois da reorganização.
- `tests/engine/test_rng.py`: nomes e ordem dos fluxos fixos; geradores puros; mesma
  semente, mesmos sorteios; fluxos distintos entre si; o fluxo do cliente $i$ não depende
  do número de clientes.
- `tests/engine/test_sampling.py`: os pontos de destino estão à distância pedida; o
  haversine coincide com o do pacote `haversine` (`rtol=1e-12`); os pontos ficam no raio e
  dentro do Brasil, com distância média igual a $r/2$; as cidades são sorteadas com peso
  pela população; os terminais ao alcance coincidem exatamente com os do legacy.
- `tests/engine/test_v2_engine.py`: `"auto"` escolhe o legacy; mesmo esquema (colunas e
  tipos) do legacy; ordem cronológica e `TRANSACTION_ID` sequencial; horários válidos;
  transações CP só em terminais ao alcance; mesma semente, mesmos dados; sementes
  diferentes, dados diferentes; o estado aleatório global não é tocado; as transações de
  um cliente não dependem dos outros clientes; só os cenários ligados aparecem; limiar do
  cenário 1; cenário 3 sempre CNP; a fraude não muda o número de transações; o cenário 2
  não depende dos cenários 3 e 4.
- `tests/engine/test_v2_vs_legacy.py` (`slow`): as comparações (A), (B) e (C) da tabela
  acima, com as tolerâncias: razão de transações por cliente-dia entre 1,00 e 1,04; valor
  médio a 2% (e KS do valor com p > 0,01); proporção de CNP a 0,02; horário do dia com
  estatística de KS $D < 0{,}02$, para CP e para CNP; perfis de clientes com KS p > 0,01;
  taxa por cenário e taxa total a 20%; transações por cliente-dia ponta a ponta a 5%.
- `tests/examples/test_simulator_notebook.py` (`slow`): executa
  `examples/simulator.ipynb` de ponta a ponta.

### Desvios do plano
- O critério do horário do dia usa o tamanho da diferença, e não o p-valor previsto no
  plano: a estatística de KS deve ser $D < 0{,}02$, separadamente para CP e CNP, com a
  semente fixa. Com cerca de 20 000 transações por tipo, o p-valor aponta diferenças
  pequenas demais para importar: em CP, $D = 0{,}0153$ deu p = 0,019, perto do limiar de
  0,01. O KS foi feito por tipo porque a mistura CP/CNP depende dos clientes sorteados
  (ver "Decisões").
- A comparação v2 × legacy foi organizada em três camadas (ver "Decisões"), em vez de
  uma única comparação ponta a ponta.
- O teste do notebook reduz também o tamanho das amostras usadas nos gráficos (de 10 000
  para 2 000 transações), porque o notebook sorteia 10 000 transações de cada tipo nos 5
  primeiros dias, o que exigiria cerca de 4 000 clientes.
- `presets.py` ainda não foi criado (fica para a Fase 5, onde os presets são definidos);
  o campo `FraudConfig.engine` (adulteração × injeção) fica para a Fase 3.
- Acrescentados: `location/sampling.py` (funções geográficas vetorizadas) e o módulo
  `simulator/fraud.py`, mantido como caminho de importação compatível.
- O v2 tem duas diferenças em relação ao legacy que o plano não previa (ver a seção
  seguinte).

### Diferenças do v2 em relação ao legacy
Além da correção prevista no plano (horários inválidos sorteados de novo, em vez de
descartados), o v2 difere do legacy em dois pontos:

1. **Renumeração depois das fraudes.** No v2, as transações são reordenadas
   cronologicamente depois da aplicação dos cenários de fraude, e o `TRANSACTION_ID`
   passa a ser a posição na ordem cronológica. Na v0.1, o cenário 4 desloca transações
   para o primeiro dia do período de roubo, mas o conjunto não é reordenado, de modo que
   a ordem do `TRANSACTION_ID` deixa de ser cronológica. Em caso de horários iguais, o v2
   desempata pelo cliente e depois pela ordem de geração, o que torna o resultado
   determinístico.
2. **Geometria esférica, e não geodésica.** No v2, o ponto a uma distância $s$ e com um
   rumo $\theta$ de um centro é calculado pela fórmula esférica, com raio médio
   $R = 6\,371{,}0088$ km, a mesma do pacote `haversine`. No legacy, ele é calculado
   pelo `geopy`, sobre o elipsoide WGS84. As distâncias entre pontos (terminais ao
   alcance, fraudador perto da vítima) usam haversine nos dois motores. A diferença entre
   esfera e elipsoide é desprezível nos raios usados (até 12 km).

**Para o modelo orientado a dados (frente [DDM]):** os atributos de distância calculados
no pipeline (por exemplo, entre o endereço de cobrança e o terminal ou o endereço de
entrega) devem usar a mesma fórmula do simulador: haversine com $R = 6\,371{,}0088$ km
(`synccfd.simulator.location.sampling.haversine_km`). Assim, um limiar de distância usado
pelo simulador (como o raio de 10 km dos terminais ao alcance) tem o mesmo significado
nos atributos.

### Pendências / ideias
- O cálculo dos terminais ao alcance é por força bruta (todos os pares cliente ×
  terminal) e responde por cerca de metade do tempo do v2; pode ser acelerado agrupando
  os terminais por cidade, se necessário.
- No v2, a escolha de *quais* terminais e clientes são comprometidos já usa o gerador
  local (pendência da Fase 0 resolvida).
- O teste do notebook foi confirmado no CI (Linux, Python 3.11): o job `slow`, com a
  comparação v2 × legacy e o notebook, levou cerca de 6 min.
- No CI com Python 3.10 (numpy 2.2), o `mypy` acusou um erro de tipo que não aparece
  com o numpy mais recente: os stubs do numpy 2.2 tratam de forma diferente a forma do
  array resultante de uma indexação booleana. Foi corrigido só com anotação de tipo.
  Conferir o `mypy` contra a versão mínima suportada do numpy, e não só contra a mais
  recente.

## Fase 2 — Customers com *concept drift*

**Status:** concluída em 2026-11-02 · integrada ao master pelo PR #3 (`7ac9566`)

### O que foi implementado

O motor v2 passou a simular *concept drift* de clientes, seguindo a taxonomia de Gama et
al. (2014). Cada cliente tem um regime A, o seu perfil original. Uma fração configurável
dos clientes (30% por padrão, em número exato) ganha também um regime B, sorteado a
partir de A:
- o valor médio é multiplicado por um fator log-normal;
- a frequência média também;
- a probabilidade de CNP é deslocada na escala logit;
- com probabilidade de 5%, o cliente muda de cidade, o que troca o endereço de
  cobrança e os terminais ao alcance.

O tipo de *drift* de cada cliente define o peso do regime B ao longo do tempo, $w(t)$:
súbito (degrau em $t_0$), gradual e incremental (rampa linear em $[t_0, t_0 + L]$) e
recorrente (episódios de $D$ dias a cada $P$ dias, a partir de $t_0$). Gradual e
incremental usam o mesmo $w(t)$, mas se aplicam de formas diferentes, como na
literatura. No gradual, há mistura de dois conceitos: cada transação vem do regime B
com probabilidade $w(t)$. No incremental, um único conceito se desloca: os parâmetros
são interpolados, $\theta(t) = (1 - w)\,\theta_A + w\,\theta_B$. O peso é avaliado no meio
de cada dia e vale para todas as transações daquele dia.

Separados do *drift* do cliente, foram implementados dois componentes que valem para
toda a população. O primeiro é o *drift* global (`GlobalDrift`): uma tendência de valor,
como a inflação, e choques em datas determinadas, como uma *Black Friday*. O segundo é
a sazonalidade (`SeasonalityConfig`), com fatores de frequência por dia da semana, nos
dias de pagamento (5, 20 e último dia útil) e em dezembro, mês em que também há um
fator de valor. A sazonalidade se repete a cada semana, mês ou ano, portanto é
estacionária e não é *drift*. Por isso fica num componente próprio e não aparece nas
colunas de regime, para não ser confundida com *drift* na avaliação.

Toda a informação sobre o *drift* fica em colunas de gabarito (`GT_*`), que nunca devem
ser usadas como atributos. Nos perfis de clientes, elas registram o tipo de *drift*, os
parâmetros de tempo e os parâmetros do regime B. Nas transações, registram o peso
$w(t)$ do dia e o regime de origem da transação (para o incremental, que não tem regime
de origem, `B` a partir de $w \geq 0{,}5$).

Os sorteios do *drift* vêm do fluxo aleatório `drift`, separado dos demais. Os clientes
sem *drift* seguem exatamente a mesma sequência de sorteios do motor da Fase 1. Com
isso, ligar o *drift* não altera as transações desses clientes, e
`customer_fraction=0` sem sazonalidade reproduz exatamente os dados da Fase 1. Essa
propriedade permite comparar, nos experimentos, conjuntos com e sem *drift* que diferem
apenas nos clientes com *drift*.

### Decisões e justificativas
| decisão | alternativa considerada | por que esta |
|---|---|---|
| número exato de clientes com *drift*, $\mathrm{round}(f \cdot N)$ | sortear cada cliente com probabilidade $f$ | a fração com *drift* fica exatamente a configurada, o que facilita comparar experimentos |
| $w(t)$ avaliado no meio de cada dia e aplicado a todas as transações do dia | avaliar $w$ no instante exato de cada transação | a frequência é diária (Poisson por dia), e as transições duram de 7 a 30 dias, de modo que a resolução diária basta; assim os horários continuam sendo sorteados depois do resto, na mesma ordem da Fase 1 |
| frequência no gradual: taxa diária $(1 - w)\,\lambda_A + w\,\lambda_B$, e cada transação vem de B com probabilidade $w$ | sobrepor dois processos de Poisson, um com taxa $(1 - w)\,\lambda_A$ e outro com taxa $w\,\lambda_B$ | a fração de transações de B fica igual a $w(t)$, como pede o plano; na sobreposição, ela seria $w\,\lambda_B / [(1 - w)\,\lambda_A + w\,\lambda_B]$ |
| `GT_REGIME` do incremental é `B` a partir de $w \geq 0{,}5$; a mudança de cidade também acontece em $w = 0{,}5$ | não rotular o incremental | o incremental não tem regime de origem por transação; o rótulo por limiar permite separar as transações em antes e depois, e `GT_REGIME_WEIGHT` guarda o valor contínuo |
| desvio-padrão do valor no regime B igual a metade da média, como na v0.1 | sortear um fator próprio para o desvio | mantém a relação da v0.1 e evita mais um parâmetro |
| clientes sem *drift* seguem o mesmo caminho de sorteios da Fase 1 | gerar todos os clientes pelo caminho com *drift* | garante dados idênticos com *drift* desligado (ablação com os mesmos números aleatórios) |
| sazonalidade e *drift* global multiplicam a taxa diária e o valor de todos os clientes; só os choques ganham uma coluna de gabarito (`GT_SHOCK`) | colunas de gabarito para cada fator | os fatores são funções determinísticas da data e da configuração; os choques são eventos que a avaliação pode querer isolar |
| figuras do notebook geradas sem fraude | gerar com a fraude padrão | com a fraude ligada, o cenário 1 (limiar fixo de R$ 220) marca como fraude muitas compras de clientes cujo regime B gasta mais, e elas somem das transações legítimas (ver "Números") |

### Parâmetros finais (padrões)
| parâmetro | valor padrão | significado |
|---|---|---|
| `customer_fraction` | 0,3 | fração de clientes com *drift* |
| `type_weights` | súbito 0,3; gradual 0,3; incremental 0,2; recorrente 0,2 | probabilidade de cada tipo entre os clientes com *drift* |
| `start_day_range` | (0,2; 0,8) | $t_0$ como fração do número de dias |
| `transition_days_range` | (7, 30) | $L$, em dias (gradual e incremental) |
| `recurrent_period_days_range` | (30, 90) | $P$, em dias (recorrente) |
| `recurrent_active_days_range` | (5, 15) | $D$, em dias (recorrente) |
| `amount_log_mu`, `amount_log_sigma` | 0; 0,5 | fator do valor médio em B ~ LogNormal($\mu$, $\sigma$) |
| `frequency_log_sigma` | 0,4 | fator da frequência em B ~ LogNormal(0, $\sigma$) |
| `cnp_logit_sigma` | 1,0 | deslocamento do logit da probabilidade de CNP em B ~ $\mathcal{N}(0, \sigma)$ |
| `relocation_prob` | 0,05 | probabilidade de o regime B morar em outra cidade |
| `category_shift_range` | (0,3; 0,8) | mudança de preferência por categoria (usada só com comerciantes, Fase 4) |
| `global_drift` | `None` | *drift* global: `amount_trend_per_year` (padrão 0) e choques (`Shock`) |
| `weekday_frequency` | (0,95; 0,95; 1,0; 1,0; 1,1; 1,15; 0,85) | fator de frequência de segunda a domingo (média 1) |
| `payday_days`, `payday_last_business_day` | (5, 20); sim | dias de pagamento |
| `payday_frequency_multiplier` | 1,3 | fator de frequência nos dias de pagamento |
| `december_frequency_multiplier`, `december_amount_multiplier` | 1,2; 1,15 | fatores de frequência e de valor em dezembro |

### Equações / regras
Peso do regime B no dia $d$, avaliado em $t = d + 0{,}5$:
$$
w(t) = \begin{cases}
0 & \text{sem drift} \\
\mathbb{1}[t \geq t_0] & \text{súbito} \\
\min\!\left(1, \max\!\left(0, \frac{t - t_0}{L}\right)\right) & \text{gradual e incremental} \\
\mathbb{1}[t \geq t_0] \cdot \mathbb{1}[(t - t_0) \bmod P < D] & \text{recorrente}
\end{cases}
$$

Regime B, a partir do regime A de cada cliente:
$\mu_B = \mu_A \cdot e^{Z_1}$, com $Z_1 \sim \mathcal{N}(\mu_{\text{amount}}, \sigma_{\text{amount}})$;
$\sigma_B = \mu_B / 2$;
$\lambda_B = \lambda_A \cdot e^{Z_2}$, com $Z_2 \sim \mathcal{N}(0, \sigma_{\text{freq}})$;
$p_B = \operatorname{sigmoid}(\operatorname{logit}(p_A) + Z_3)$, com $Z_3 \sim \mathcal{N}(0, \sigma_{\text{cnp}})$.

Geração, para o dia $d$ com peso $w = w(d + 0{,}5)$ e fatores de calendário $s_f(d)$
(frequência) e $s_v(d)$ (valor):
- número de transações: $N_d \sim \mathrm{Poisson}\big(s_f(d)\,[(1 - w)\,\lambda_A + w\,\lambda_B]\big)$;
- gradual: cada transação vem de B com probabilidade $w$ e usa os parâmetros do seu
  regime; súbito e recorrente: os parâmetros do regime ativo; incremental:
  $\theta = (1 - w)\,\theta_A + w\,\theta_B$ para $\theta \in \{\mu, \sigma, p\}$;
- valor: sorteado como na Fase 1 com os parâmetros acima e multiplicado por $s_v(d)$.

Fatores de calendário: $s_f(d) = f_{\text{semana}}(d) \cdot f_{\text{pagamento}}(d) \cdot f_{\text{dezembro}}(d) \cdot f_{\text{choque}}(d)$
e $s_v(d) = v_{\text{dezembro}}(d) \cdot (1 + r)^{d / 365{,}25} \cdot v_{\text{choque}}(d)$,
em que $r$ é a tendência anual de valor do *drift* global.

### Figuras geradas
| arquivo | legenda proposta | seção da dissertação |
|---|---|---|
| `docs/figures/fase2/drift_exemplos_clientes.pdf` | Um cliente de cada tipo de *drift*. À esquerda, o valor de cada transação, colorido pelo regime de origem, e os valores médios dos regimes A e B; à direita, as transações por dia, a média móvel de 14 dias e a taxa esperada $\lambda(t)$. A linha vertical marca $t_0$; a faixa cinza, a transição $[t_0, t_0 + L]$; as faixas laranja, os episódios do regime B (recorrente). Clientes escolhidos entre os frequentes ($\geq 2{,}5$ transações por dia no regime A) com a maior mudança de valor; no gradual e no incremental, só entre os de transição longa ($L \geq 25$ dias). | 4.4 Componentes novos — *concept drift* (figura central) |
| `docs/figures/fase2/drift_fracao_regime_b.pdf` | Peso médio do regime B, $\bar w(t)$, entre os clientes de cada tipo de *drift* e entre todos os clientes. Para os tipos súbito e recorrente, é a fração de clientes no regime B. | 4.4 Componentes novos — *concept drift* |
| `docs/figures/fase2/drift_parametros_regime_b.pdf` | Distribuição das mudanças sorteadas para os clientes com *drift*: razão dos valores médios e das frequências (escala log) e diferença na probabilidade de CNP. | 4.4 ou 4.6 Saídas e gabarito |

Configuração das figuras: 5 000 clientes, 10 000 terminais, 365 dias a partir de
2023-01-01, `random_state=42`, `DriftConfig()` padrão, fraude desligada.

### Números
| medida | valor | configuração |
|---|---|---|
| clientes por tipo de *drift* | sem *drift*: 3 500; gradual: 457; súbito: 425; incremental: 312; recorrente: 306; mudaram de cidade: 73 | 5 000 clientes, 365 dias, `random_state=42`, `DriftConfig()` |
| média de $\lvert \ln(\mu_B / \mu_A) \rvert$ | entre 0,39 (recorrente) e 0,43 (súbito) | idem |
| `customer_fraction=0` × motor da Fase 1 | dados idênticos (perfis e transações, fora as colunas `GT_*`) | 1 000 clientes, 120 dias, `random_state=42`, sem fraude |
| clientes sem *drift*: variação do valor normalizado ($v / \mu_A$) ao longo do período | −0,16% (critério: menor que 2%) | 1 000 clientes, 2 000 terminais, 120 dias, `random_state=42`, `customer_fraction=0,5`, sem fraude |
| súbito: correlação entre $\ln$(valor médio depois/antes de $t_0$) e $\ln(\mu_B / \mu_A)$ | 0,991 (127 clientes com pelo menos 10 transações antes e depois); erro absoluto médio 0,05 | idem |
| gradual: fração de transações de B × $w$ médio, em 5 faixas de $w$ | diferença máxima de 0,011 (por exemplo, $w = 0{,}502$ e fração 0,508) | idem |
| incremental: valor médio observado ÷ média interpolada | 1,029 (o excesso vem da troca de sorteios negativos, como no regime A da Fase 1) | idem |
| tempo de geração no v2 | sem *drift*: 14,5 s; com *drift*: 16,1 s; com *drift* e sazonalidade: 24,2 s | 10 000 clientes, 20 000 terminais, 180 dias, `random_state=42` |
| cenário 1 (limiar fixo de R$ 220) nas transações de cada regime | regime A: 0,06%; regime B: 2,30%; 11 360 das 13 229 transações do cenário 1 (86%) vêm do regime B, que tem 13,5% das transações | 5 000 clientes, 365 dias, `random_state=42`, `DriftConfig()`, `FraudConfig()` |

### Testes que comprovam o comportamento
- `tests/golden/test_legacy_reproducibility.py`: o legacy continua reproduzindo a v0.1.0.
- `tests/engine/test_drift.py`:
  - validação de `DriftConfig`, `SeasonalityConfig` e `Shock`; `type_weights` somente
    para leitura; escolha do motor (`auto` vira `v2` com *drift* ou sazonalidade; o
    legacy recusa esses componentes);
  - forma de $w(t)$ para cada tipo;
  - `test_no_drift_customers_are_unchanged`: `customer_fraction=0` reproduz exatamente
    a Fase 1; com *drift* ligado, os clientes sem *drift* têm exatamente as mesmas
    transações;
  - `test_no_drift_customers_have_no_trend`: variação do valor normalizado ao longo do
    período menor que 2%;
  - `test_sudden_drift_changes_level_at_t0`: correlação maior que 0,85 entre a mudança
    observada e a configurada, e `GT_REGIME` igual a `A` antes e `B` depois de $t_0$;
  - `test_gradual_regime_share_follows_weight`: fração de B igual a $w$ em 5 faixas, com
    tolerância de 0,05;
  - `test_incremental_parameters_are_interpolated`: valor médio igual à média
    interpolada, com tolerância de 5%;
  - `test_recurrent_episodes`: `GT_REGIME` igual a `B` exatamente nos episódios;
  - `test_relocated_customers_buy_near_their_new_address`: no regime B, o endereço de
    cobrança é o novo e as compras CP ficam a até 10 km dele;
  - gabarito (`GT_*`), ausência de rótulos de fraude criados pelo *drift*,
    reprodutibilidade;
  - fatores diários (dia da semana, último dia útil, dias 5 e 20, dezembro, choque,
    tendência), efeito da sazonalidade no volume diário e rótulo `GT_SHOCK`.
- `tests/examples/test_notebooks.py` (`slow`): executa `examples/simulator.ipynb` e
  `examples/drift.ipynb` de ponta a ponta.

### Desvios do plano
- Os critérios "média móvel estável" e "mudança de nível em $t_0$" foram verificados
  por tamanho do efeito (variação menor que 2%; correlação maior que 0,85), e não por um
  teste de significância, pelo mesmo motivo do KS da Fase 1: com muitas transações, um
  p-valor acusa diferenças pequenas demais para importar.
- O teste de `customer_fraction=0` exige dados idênticos, e não só a mesma distribuição.
- O gabarito ganhou `GT_DRIFT_ACTIVE_DAYS` (duração dos episódios do recorrente),
  `GT_RELOCATED`, `GT_B_billing_lat` e `GT_B_billing_long`, além das colunas previstas, e
  `GT_SHOCK` nas transações quando há *drift* global.
- A frequência no gradual usa a taxa diária interpolada, e não a sobreposição de dois
  processos (ver "Decisões").
- `category_pref` (mudança de preferência por categoria) fica para a Fase 4, como
  previsto; `category_shift_range` já existe no `DriftConfig`, mas ainda não tem efeito.
- O teste do notebook passou a executar também `examples/drift.ipynb`; o arquivo de
  teste foi renomeado para `tests/examples/test_notebooks.py`.

### Pendências / ideias
- **Interação com o cenário 1 (resolvida na Fase 3):** com o limiar fixo de R$ 220,
  86% das transações do cenário 1 (11 360 de 13 229) vêm de clientes no regime B, que
  responde por só 13,5% das transações; no regime B, 2,30% das transações viram cenário
  1, contra 0,06% no regime A (ver "Números"). Ou seja, o cenário 1 rotula como fraude
  mudanças legítimas de comportamento, e um detector treinado nesses dados aprenderia a
  tratar *drift* como fraude. **Decisão (2026-11-02):** no motor v2, o cenário 1 passa a
  ser injetado, como os demais: o fraudador faz uma compra nova com valor acima de um
  quantil alto das compras recentes do próprio cliente, multiplicado por um fator
  configurável, e transações legítimas nunca são rotuladas como fraude. O legacy mantém o
  limiar fixo (golden test). Ver a tabela de cenários da Fase 3.
- Os fatores de sazonalidade são aplicados igualmente a todos os clientes; com
  comerciantes (Fase 4), o fim de semana passa a variar por categoria.
- O *drift* global ainda não tem figura própria no notebook.

## Fase 3 — Fraudadores realistas

**Status:** implementada em 2026-11-02, na branch `feat/v0.2-phase3-frauds`, aguardando
revisão · PR *(a abrir)*

### O que foi implementado

O motor v2 ganhou um segundo mecanismo de fraude, a **injeção**
(`FraudConfig(mechanism="inject")`). Na v0.1, as fraudes são produzidas alterando ou
rotulando transações legítimas: o cenário 1 rotula toda compra acima de R$ 220, e os
cenários 3 e 4 mudam o tipo, o local e o valor de compras do cliente. Na injeção, o
fraudador é um agente que faz transações **novas**, que se somam às do cliente. O
histórico legítimo de cada cliente fica intacto, como na realidade, e os atributos de
agregação calculados sobre esse histórico (médias e contagens por janela) não são
contaminados pela fraude. A única exceção é o cenário 2: o terminal comprometido é o
ponto da fraude, e as compras CP feitas nele são rotuladas, sem alteração. O mecanismo
antigo (`mechanism="tamper"`) continua sendo o padrão, e o motor legacy só aceita ele;
o golden test não muda.

Cada cenário injetado descreve o comportamento de um tipo de fraudador. No cenário 1
(valor anômalo), a vítima recebe uma compra muito acima do seu próprio histórico: o
dobro do quantil de 95% das suas compras dos 30 dias anteriores, vezes $1 + 0{,}5\,U$.
O resto da transação (tipo, terminal das compras CP, endereços, modelo de horário)
copia uma compra recente da vítima, de modo que só o valor é anômalo. No cenário 2,
um terminal fica comprometido por 28 dias; suas compras CP viram fraude, e cada cartão
lido ali vaza, com probabilidade de 5%, para uma campanha do cenário 3 que começa de 1
a 14 dias depois, do mesmo fraudador. No cenário 3 (número do cartão roubado), o
fraudador faz rajadas de 2 a 5 compras CNP em até 3 h, em dias sorteados de um período
de 14 dias, com entrega no seu próprio endereço, metade das rajadas de madrugada e
valores em torno do dobro da mediana recente da vítima. No cenário 4 (cartão físico
roubado), a partir de um instante $\tau$, o fraudador compra perto da cidade da vítima,
em média a cada 20 minutos, até o cartão ser bloqueado (probabilidade de 15% após cada
compra), até a próxima compra ultrapassar o limite de R$ 2 000, até 10 compras ou até
24 h. O cenário 5 (*card testing*) é novo: de 3 a 10 compras CNP de R$ 1 a 5 em poucos
minutos; em metade dos casos, o fraudador volta de 1 a 72 h depois com o mesmo cartão,
numa campanha do cenário 3.

Os fraudadores também mudam com o tempo. `FraudConfig.intensity_schedule` associa a
cada cenário um multiplicador da sua taxa diária (constante, degrau, rampa ou pulso,
com os tempos como fração do período). `FraudConfig.realistic()`, a configuração que o
preset `realistic` da Fase 5 vai usar, liga os cinco cenários com taxas proporcionais à
população e dois exemplos de *drift* dos fraudadores: o cenário 2 cai pela metade a
partir de 60% do período (adoção do chip) e o cenário 5 surge a partir de 30%. Com essa
configuração, 0,44% das transações são fraude.

O gabarito de fraude ganhou três colunas: `GT_FRAUD_SCENARIO` (igual a
`TX_FRAUD_SCENARIO`), `GT_FRAUDSTER_ID` e `GT_CAMPAIGN_ID`. Uma campanha agrupa as
transações de um mesmo ataque: um terminal comprometido, uma sequência de cartão
roubado ou um *card testing* com as compras que o seguem. Isso permite avaliar um
detector por ataque, e não só por transação (por exemplo, em que transação da campanha
o ataque foi detectado). As transações injetadas não têm regime de *drift*
(`GT_REGIME` vazio), porque não vêm do comportamento do cliente.

Com isso, a fraude e o *drift* ficam independentes, como pede a seção 3.4. Com o
limiar fixo, 86% do cenário 1 vinha de clientes no regime B (Fase 2). Com a injeção,
as vítimas do cenário 1 estão no regime B em 13,5% dos casos, a mesma fração do regime
B nas transações legítimas, e nenhuma transação legítima é rotulada pelo cenário 1.

### Decisões e justificativas
| decisão | alternativa considerada | por que esta |
|---|---|---|
| cenário 1 injetado, com referência no histórico recente do próprio cliente (Decisão de 2026-11-02, ver Fase 2) | quantil global ou por categoria dos valores | um limiar comum a todos rotula como fraude os clientes que gastam mais, inclusive por *drift* legítimo (86% do cenário 1 no regime B, na Fase 2); o histórico recente acompanha o regime atual do cliente |
| no cenário 1, a transação copia uma compra recente da vítima, exceto horário e valor | sortear tipo, terminal e endereços do zero | isola o sinal do cenário no valor; os outros cenários cobrem os sinais de tipo, local e horário |
| no cenário 1, vítima com menos de 5 compras nos 30 dias anteriores é pulada | usar o valor médio do perfil | sem histórico não há "acima do histórico"; a perda é pequena (488 fraudes contra cerca de 540 vítimas sorteadas, em 10 000 clientes × 180 dias), concentrada nos primeiros dias e nos clientes que compram pouco |
| valor do cenário 1 arredondado para cima, ao centavo | arredondamento comum | garante que todo valor fique acima de fator × quantil, como no critério de aceite |
| cenário 2 continua rotulando compras legítimas (só as CP) | injetar compras com cartões clonados no próprio terminal | segue a tabela 3.2: o terminal é o ponto da fraude; as CNP não passam pelo terminal físico; o uso posterior dos cartões lidos é modelado pelo vazamento para o cenário 3 |
| nome `mechanism` (`"tamper"`, `"inject"`) | `FraudConfig(engine="v2")`, como no plano | `engine` já é o motor do `DatasetGenerator`; `FraudConfig(engine="legacy")` dentro do motor v2 seria confuso |
| `mechanism="tamper"` continua o padrão | injeção como padrão no v2 | o padrão de `FraudConfig` reproduz a v0.1 (Fase 0) e o v2 da Fase 1; a injeção é ligada explicitamente ou por `FraudConfig.realistic()` |
| cenário 5 e `intensity_schedule` só com injeção | aplicar também à adulteração | a adulteração existe para reproduzir a v0.1; misturar os dois mecanismos complicaria a comparação |
| no modo `absolute` com `intensity_schedule`, a contagem $c \cdot m(d)$ é arredondada para cima ou para baixo ao acaso | arredondar para o inteiro mais próximo | mantém a média $c \cdot m(d)$; com $c = 3$ e $m = 0{,}5$, o arredondamento comum daria 2 por dia em vez de 1,5 |
| vítimas sorteadas ao acaso entre todos os clientes, cenário a cenário | excluir clientes já atacados, como a v0.1 | com as taxas proporcionais, a chance de o mesmo cliente ser vítima de dois ataques simultâneos é muito pequena; `fraud_target_bias` (opcional no plano) ficou para depois |
| fraudes injetadas depois do fim do período são descartadas | estender o período | mantém o número de dias configurado; afeta só campanhas que começam nos últimos dias |
| transações injetadas sem regime de *drift* (`GT_REGIME` vazio) | herdar o regime da vítima | a transação não vem do comportamento do cliente; o regime da vítima pode ser obtido do gabarito do cliente, como no notebook |

### Parâmetros finais (padrões)
Parâmetros novos do `FraudConfig`, usados só com `mechanism="inject"`:

| parâmetro | valor padrão | significado |
|---|---|---|
| `mechanism` | `"tamper"` | `"tamper"` (v0.1) ou `"inject"` |
| `s1_customers_per_day` / `s1_customer_fraction_per_day` | 3 / 3e-4 | vítimas do cenário 1 por dia |
| `s5_customers_per_day` / `s5_customer_fraction_per_day` | 2 / 2e-4 | vítimas do cenário 5 por dia |
| `history_days` | 30 | janela do histórico recente da vítima (cenários 1, 3 e 4) |
| `min_history_tx` | 5 | mínimo de compras recentes para o cenário 1 |
| `s1_quantile`, `s1_amount_factor`, `s1_amount_spread` | 0,95; 2,0; 0,5 | valor do cenário 1: $2 \times q_{0,95} \times (1 + 0{,}5\,U)$ |
| `s2_leak_prob` | 0,05 | probabilidade de um cartão lido no terminal comprometido ir para o cenário 3 |
| `s2_leak_delay_days_range` | (1, 14) | dias até o uso do cartão vazado |
| `s3_mean_bursts` | 3 | rajadas por campanha: $1 + \mathrm{Poisson}(2)$ |
| `s3_burst_size_range` | (2, 5) | compras por rajada |
| `s3_burst_hours` | 3 | duração máxima de uma rajada |
| `s3_amount_factor` | 2,0 | valor do cenário 3 em relação à mediana recente |
| `s4_max_txs` | 10 | compras no máximo |
| `s4_block_prob_per_tx` | 0,15 | probabilidade de bloqueio após cada compra |
| `s4_spending_limit` | 2 000 | limite de gasto da sequência (R$) |
| `s4_mean_gap_minutes` | 20 | intervalo médio entre compras |
| `s4_amount_factor` | 3,0 | valor do cenário 4 em relação à mediana recente |
| `s4_cnp_attempt_prob` | 0,2 | probabilidade de uma compra ser tentativa CNP |
| `s5_n_tests_range` | (3, 10) | compras de teste |
| `s5_amount_range` | (1, 5) | valor de cada teste (R$) |
| `s5_mean_gap_seconds` | 60 | intervalo médio entre testes |
| `s5_escalate_prob` | 0,5 | probabilidade de seguir para o cenário 3 |
| `s5_escalation_delay_hours_range` | (1, 72) | horas até o cenário 3 |
| `night_prob` | 0,5 | probabilidade de uma rajada (cenários 3 e 5) começar de madrugada (0–6 h) |
| `amount_log_sigma` | 0,3 | dispersão log-normal dos valores dos cenários 3 e 4 |
| `intensity_schedule` | vazio | multiplicador da taxa diária de cada cenário ao longo do tempo |

`FraudConfig.realistic()`: cenários 1 a 5, `rate_mode="proportional"`,
`mechanism="inject"`, cenário 2 com degrau para 0,5 em 60% do período e cenário 5 com
degrau de 0 para 1 em 30%; os demais valores são os padrões acima. Os cenários 2, 3 e
4 usam as frações e durações da Fase 0 (`s2_terminal_fraction_per_day = 1e-4`,
`s2_duration_days = 28`, `s3_customer_fraction_per_day = s4_customer_fraction_per_day =
3e-4`, `s3_duration_days = 14`).

### Equações / regras
Taxa diária do cenário $k$ no dia $d$, com multiplicador $m_k(d)$ avaliado em
$x = (d + 0{,}5)/N_{\text{dias}}$:
$n_k(d) \sim \mathrm{Poisson}\big(f_k \cdot N \cdot m_k(d)\big)$ no modo proporcional
($N$ = clientes, ou terminais no cenário 2), e
$n_k(d) = \lfloor c_k\, m_k(d) + U \rfloor$ no modo absoluto com multiplicador.

Multiplicadores: degrau, $m(x) = b$ se $x < s$ e $\ell$ caso contrário; rampa,
$m(x) = b + (\ell - b)\,\min\!\big(1, \max(0, (x - s)/(e - s))\big)$; pulso, $m(x) = \ell$
se $s \leq x < e$ e $b$ fora.

Cenário 1, para a vítima $c$ no dia $d$, com $H_c$ = valores das compras legítimas de
$c$ em $[d - 30, d)$ (só se $|H_c| \geq 5$):
$\text{valor} = \big\lceil 2 \cdot Q_{0{,}95}(H_c) \cdot (1 + 0{,}5\,U) \big\rceil_{\text{centavo}}$,
com $U \sim \mathcal{U}(0, 1]$.

Cenários 3 e 4: $\text{valor} = \big\lceil r_c \cdot a_k \cdot e^{Z} \big\rceil_{\text{centavo}}$,
com $r_c$ = mediana de $H_c$ no início do ataque (ou o valor médio do perfil, com menos
de 5 compras), $a_3 = 2$, $a_4 = 3$ e $Z \sim \mathcal{N}(0; 0{,}3)$.

Cenário 3: $1 + \mathrm{Poisson}(2)$ rajadas, cada uma num dia sorteado de
$[d_0, d_0 + 14)$; início de madrugada ($\mathcal{U}[0, 6\,\text{h})$) com probabilidade
0,5, senão $\mathcal{U}[6, 24\,\text{h})$; de 2 a 5 compras, a primeira no início da
rajada e as demais em instantes uniformes nas 3 h seguintes.

Cenário 4: compras em $\tau, \tau + G_1, \tau + G_1 + G_2, \dots$, com
$G_i \sim \mathrm{Exp}(\text{média } 20\,\text{min})$; para antes da compra que levaria o
total acima de R$ 2 000, depois de uma compra com probabilidade 0,15 (bloqueio), após
10 compras ou quando o próximo instante passa de $\tau + 24\,\text{h}$. As compras CP
são em terminais a até 10 km do fraudador, que está a até 12 km do endereço de cobrança
da vítima.

Cenário 5: de 3 a 10 testes em $t_0, t_0 + G_1, \dots$, com
$G_i \sim \mathrm{Exp}(\text{média } 60\,\text{s})$ e valores $\mathcal{U}(1, 5)$; com
probabilidade 0,5, uma campanha do cenário 3 começa em $t_{\text{último}} + \mathcal{U}(1, 72)\,\text{h}$,
com a primeira rajada nesse instante.

### Figuras geradas
| arquivo | legenda proposta | seção da dissertação |
|---|---|---|
| `docs/figures/fase3/fraudes_taxa_por_cenario.pdf` | Fração semanal das transações que são fraude, por cenário, com `FraudConfig.realistic()`. As linhas verticais marcam o *drift* dos fraudadores: a taxa do cenário 2 cai pela metade a partir de 60% do período, e o cenário 5 (*card testing*) surge a partir de 30%. | 4.4 Componentes novos — fraudes |
| `docs/figures/fase3/fraudes_campanha_card_testing.pdf` | Linha do tempo de uma campanha: o fraudador testa o cartão com compras de R$ 1 a 5 em poucos minutos (cenário 5, à direita em detalhe) e, horas depois, faz rajadas de compras CNP com o mesmo cartão (cenário 3). Em cinza, as compras legítimas da vítima, que continuam normalmente. | 4.4 Componentes novos — fraudes |
| `docs/figures/fase3/fraudes_valor_por_cenario.pdf` | Valor das transações legítimas e das fraudes de cada cenário (escala log; caixas: quartis; hastes: até 1,5 × IQR, em log). | 4.4 Componentes novos — fraudes |
| `docs/figures/fase3/fraudes_cenario1_regime_b.pdf` | Cenário 1 e *drift*, nos mesmos dados: fração das transações legítimas que vêm do regime B; fração das transações que o limiar fixo de R$ 220 (v0.1) rotularia que vêm do regime B; fração das vítimas do cenário 1 injetado que estavam no regime B no dia da fraude. | 4.4 ou 4.6 Saídas e gabarito |

Configuração das figuras: 5 000 clientes, 10 000 terminais, 365 dias a partir de
2023-01-01, `random_state=42`, `DriftConfig()` padrão, `FraudConfig.realistic()`.

### Números
| medida | valor | configuração |
|---|---|---|
| taxa de fraude de `FraudConfig.realistic()` | 0,440% (cenário 1: 0,014%; 2: 0,097%; 3: 0,207%; 4: 0,078%; 5: 0,044%) | 10 000 clientes, 20 000 terminais, 180 dias, `random_state=42`, sem *drift* |
| mesma taxa com outro tamanho | 0,435% | 1 000 clientes, 2 000 terminais, 180 dias, `random_state=42` |
| taxa com *drift* | 0,416% | 5 000 clientes, 10 000 terminais, 365 dias, `random_state=42`, `DriftConfig()` |
| campanhas e transações por campanha | cenário 1: 488 campanhas, 1 transação cada; 2: 237 terminais, 14,7 transações em média; 3: 724 campanhas, 10,3; 4: 555, 5,1; 5: 256, 6,2 | 10 000 clientes × 180 dias, como acima |
| origem das campanhas do cenário 3 | 534 diretas, 72 de cartões vazados no cenário 2, 118 depois de *card testing* | idem |
| *card testing* seguido do cenário 3 | 46% das campanhas (configurado: 50%) | idem |
| cenário 4: gasto e duração | R$ 809 por campanha e 1,3 h em média; 72% das compras são CP | idem |
| valor mediano | legítimas: R$ 46,15; cenário 1: R$ 238,29; 2: R$ 42,10; 3: R$ 100,22; 4: R$ 142,91; 5: R$ 2,95 | idem |
| fração de madrugada (0–6 h) | legítimas: 12,6%; cenário 3: 42%; cenário 5: 44% | idem |
| cenário 1 e regime B | transações legítimas no regime B: 13,5%; vítimas do cenário 1 no regime B: 13,5% (peso médio de B no dia da fraude, 513 fraudes); com o limiar fixo de R$ 220 nos mesmos dados, 85,9% das transações rotuladas viriam do regime B | 5 000 clientes, 365 dias, `random_state=42`, `DriftConfig()`, `FraudConfig.realistic()` |
| transações legítimas rotuladas pelo cenário 1 | 0 | idem |
| tempo de geração no v2 | injeção (`realistic`): 23,6 s; adulteração (`FraudConfig()`): 25,4 s | 10 000 clientes, 20 000 terminais, 180 dias, sem *drift* |

### Testes que comprovam o comportamento
- `tests/golden/test_legacy_reproducibility.py`: o legacy continua reproduzindo a v0.1.0.
- `tests/engine/test_injection.py`:
  - validação de `FraudConfig` e `IntensitySchedule`; a injeção exige o motor v2 (o
    `auto` escolhe o v2; o legacy recusa); `FraudConfig()` continua com adulteração;
  - formas do multiplicador (constante, degrau, rampa, pulso);
  - `test_step_schedule_changes_daily_rate_on_the_planned_day`: no modo absoluto, o
    degrau muda a contagem exatamente no dia previsto; no proporcional, a taxa média cai
    pela metade (tolerância de 0,02);
  - `test_schedule_in_generated_data`: o primeiro *card testing* acontece no dia 27 de 90
    (30% do período);
  - `test_legitimate_transactions_are_unchanged`: as transações legítimas são idênticas,
    coluna a coluna, às geradas com a fraude desligada, e toda fraude que não é do
    cenário 2 é uma transação nova;
  - `test_scenario_1_is_far_above_the_customer_history`: todo valor do cenário 1 fica
    acima de fator × quantil das compras dos 30 dias anteriores da vítima, e abaixo do
    teto $(1 + 0{,}5)$ vezes esse valor;
  - `test_scenario_1_with_drift_does_not_label_regime_b`: com *drift*, nenhuma transação
    legítima é rotulada, e a fração do cenário 1 no regime B difere em menos de 0,05 da
    fração das transações legítimas no regime B;
  - cenário 2 (só compras CP, uma campanha por terminal, cartões vazados usados depois
    pelo mesmo fraudador); cenário 3 (CNP, endereço do fraudador); cenário 4 (até 24 h,
    até o limite e o número máximo de compras, compras CP perto da vítima); cenário 5
    (valores de R$ 1 a 5, CNP, de 3 a 10 testes; quando escala, mesma campanha, cliente e
    fraudador, e o cenário 3 vem depois do último teste);
  - os cinco cenários funcionam sem comerciantes; gabarito; reprodutibilidade;
    `test_realistic_fraud_rate`: taxa entre 0,2% e 1%.
- `tests/examples/test_notebooks.py` (`slow`): executa também `examples/frauds.ipynb`.

### Desvios do plano
- `FraudConfig(engine="v2")` virou `FraudConfig(mechanism="inject")` (ver "Decisões").
- O cenário 2 continua rotulando compras legítimas, como na tabela 3.2; a seção 3.1, que
  dizia que todos os cenários injetam transações, foi corrigida.
- O preset `realistic` é da Fase 5; o critério de taxa foi verificado com
  `FraudConfig.realistic()`, que o preset vai usar.
- `fraud_target_bias` (opcional) não foi implementado.
- O gabarito das transações injetadas deixa `GT_REGIME` e `GT_REGIME_WEIGHT` vazios.
- O teste do cenário 4 verifica a distância das compras CP à vítima, e não uma fração
  mínima de compras CP: com poucos terminais (2 000), o fraudador muitas vezes não tem
  terminal perto e só faz tentativas CNP (53% de CP no conjunto do teste, contra 72% com
  20 000 terminais).

### Pendências / ideias
- Com comerciantes (Fase 4): o cenário 1 em categorias de valor alto, o cenário 2
  comprometendo o comerciante inteiro, o cenário 3 em categorias revendáveis ou
  digitais e o cenário 4 em categorias revendáveis, como na tabela 3.2; e a migração de
  categoria do cenário 3 no `intensity_schedule`.
- `fraud_target_bias`: fraudadores com preferência por clientes de alto gasto.
- Na curva semanal do cenário 2, a taxa sobe nos primeiros 28 dias (os terminais
  comprometidos se acumulam até a primeira duração completa) e varia bastante entre
  semanas, porque o número de compras por terminal é muito desigual; a queda do degrau
  aparece, mas para um efeito mais limpo valeria média de várias seeds.
- Os cartões vazados no cenário 2 são usados pelo mesmo fraudador (mesmo
  `GT_FRAUDSTER_ID` do terminal); um modelo de revenda atribuiria outro fraudador a cada
  campanha vazada.

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
