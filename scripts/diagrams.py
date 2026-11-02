"""Architecture diagrams of the simulator (dissertation, cap. 4, and the Sphinx docs).

- ``fluxo_geracao``: the generation flow of the v2 engine, from the seed to the saved
  dataset, with the configuration that drives each optional step;
- ``arquitetura_modulos``: the modules of the package and who calls whom.

Each diagram is written as PDF (dissertation) and SVG (documentation). Called by
``scripts/make_figures.py``; can also be run alone:
``python scripts/diagrams.py docs/figures/arquitetura``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Fixed ids in the SVG files, so that an unchanged diagram is written the same
matplotlib.rcParams["svg.hashsalt"] = "synccfd"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
# Validated categorical palette, fixed order: core step, optional component,
# configuration, output
CORE, OPTIONAL, CONFIG, OUTPUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FILL = {CORE: "#e3eefb", OPTIONAL: "#fde8dd", CONFIG: "#ddf3ea", OUTPUT: "#fcf0d2"}


def _box(ax, x, y, w, h, title, detail="", color=CORE, dashed=False):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=FILL[color],
        edgecolor=color,
        lw=1.4,
        ls=(0, (4, 2)) if dashed else "-",
    )
    ax.add_patch(patch)
    if detail:
        ax.text(
            x,
            y + 0.13,
            title,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color=INK,
        )
        ax.text(
            x,
            y - 0.15,
            detail,
            ha="center",
            va="center",
            fontsize=8,
            color=SECONDARY,
            family="monospace",
        )
    else:
        ax.text(
            x, y, title, ha="center", va="center", fontsize=10, weight="bold", color=INK
        )


def _arrow(ax, start, end, color=SECONDARY, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            lw=1.1,
            ls=(0, (3, 2)) if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
        )
    )


def _legend(ax, x, y, items):
    for i, (color, label, dashed) in enumerate(items):
        yy = y - 0.32 * i
        ax.add_patch(
            FancyBboxPatch(
                (x, yy - 0.09),
                0.3,
                0.18,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=FILL[color],
                edgecolor=color,
                lw=1.2,
                ls=(0, (4, 2)) if dashed else "-",
            )
        )
        ax.text(x + 0.42, yy, label, va="center", fontsize=8.5, color=SECONDARY)


def generation_flow():
    """The generation flow of the v2 engine (``engine.simulate``)."""
    fig, ax = plt.subplots(figsize=(10, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    x, w, h = 5.6, 4.6, 0.62
    steps = [
        (9.0, "Fluxos aleatórios", "RandomStreams(random_state)", CORE, False),
        (8.0, "Localizações", "LocationFeatures: cidades, população", CORE, False),
        (7.0, "Clientes e terminais", "customers.py · terminals.py", CORE, False),
        (
            6.0,
            "Drift de clientes (regime B)",
            "drift.py · assign_drift",
            OPTIONAL,
            True,
        ),
        (
            5.0,
            "Fatores diários",
            "daily_factors.py: sazonalidade, choques",
            OPTIONAL,
            True,
        ),
        (
            4.0,
            "Comerciantes e preferências",
            "merchants.py · build_merchants",
            OPTIONAL,
            True,
        ),
        (3.0, "Transações legítimas", "transactions.py ou merchants.py", CORE, False),
        (
            2.0,
            "Fraudes e gabarito",
            "fraud.py (tamper) · injection.py (inject)",
            CORE,
            False,
        ),
    ]
    for y, title, detail, color, dashed in steps:
        _box(ax, x, y, w, h, title, detail, color, dashed)
    for (y0, *_), (y1, *_) in zip(steps, steps[1:]):
        _arrow(ax, (x, y0 - h / 2), (x, y1 + h / 2))

    configs = [
        (6.0, "DriftConfig"),
        (5.0, "SeasonalityConfig\nDriftConfig.global_drift"),
        (4.0, "MerchantConfig"),
        (2.0, "FraudConfig"),
    ]
    for y, label in configs:
        _box(ax, 1.55, y, 2.4, 0.62 if "\n" not in label else 0.78, "", "", CONFIG)
        ax.text(
            1.55,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            family="monospace",
            color=INK,
        )
        _arrow(ax, (2.75, y), (x - w / 2, y), color=CONFIG, dashed=True)
    # The merchants also change the drift (category preferences) and the frauds
    _arrow(
        ax, (x + w / 2, 4.0), (x + w / 2, 5.95), color=OPTIONAL, dashed=True, rad=0.55
    )
    ax.text(
        8.85, 5.0, "preferência\nno regime B", fontsize=8, color=SECONDARY, va="center"
    )
    _arrow(
        ax, (x + w / 2, 3.95), (x + w / 2, 2.05), color=OPTIONAL, dashed=True, rad=-0.55
    )
    ax.text(
        8.85, 3.0, "categorias\ndas fraudes", fontsize=8, color=SECONDARY, va="center"
    )

    outputs = [
        (2.35, "customer_profiles"),
        (4.65, "terminal_profiles"),
        (6.95, "merchant_profiles"),
        (9.0, "transactions"),
    ]
    # One bar from the frauds to the four outputs
    ax.plot([x, x], [2.0 - h / 2, 1.3], color=SECONDARY, lw=1.1)
    ax.plot([outputs[0][0], outputs[-1][0]], [1.3, 1.3], color=SECONDARY, lw=1.1)
    for ox, _ in outputs:
        _arrow(ax, (ox, 1.3), (ox, 1.0))
    for ox, label in outputs:
        _box(
            ax,
            ox,
            0.75,
            2.0 if ox < 9 else 1.6,
            0.5,
            "",
            "",
            OUTPUT,
            dashed=label == "merchant_profiles",
        )
        ax.text(
            ox,
            0.75,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            family="monospace",
            color=INK,
        )
    ax.text(
        5.6,
        0.2,
        "save(): os perfis, as transações (um arquivo por dia) e config.json",
        ha="center",
        fontsize=8.5,
        color=SECONDARY,
    )

    _legend(
        ax,
        0.3,
        9.25,
        [
            (CORE, "etapa sempre presente", False),
            (OPTIONAL, "componente opcional", True),
            (CONFIG, "configuração", False),
            (OUTPUT, "saída", False),
        ],
    )
    fig.tight_layout()
    return fig


def module_architecture():
    """The modules of ``synccfd`` and their dependencies."""
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Package frame
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.2),
            10.6,
            6.5,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="none",
            edgecolor=MUTED,
            lw=1,
        )
    )
    ax.text(
        0.4,
        6.5,
        "synccfd",
        fontsize=10,
        color=SECONDARY,
        family="monospace",
        va="center",
    )

    _box(ax, 5.5, 5.85, 4.0, 0.75, "DatasetGenerator", "generate · generate_all · save")
    _box(
        ax, 1.75, 5.85, 2.6, 0.75, "presets.py", "legacy · realistic · ablation", CONFIG
    )
    _box(
        ax, 9.25, 5.85, 2.6, 0.75, "config.py", "FraudConfig, DriftConfig, ...", CONFIG
    )
    _arrow(ax, (3.05, 5.85), (3.5, 5.85), color=CONFIG)
    _arrow(ax, (7.95, 5.85), (7.5, 5.85), color=CONFIG)

    # v2 engine
    ax.add_patch(
        FancyBboxPatch(
            (0.45, 1.45),
            6.85,
            3.55,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="none",
            edgecolor=CORE,
            lw=1.2,
        )
    )
    ax.text(
        0.65,
        4.78,
        "simulator/engine/ (motor v2): simulate()",
        fontsize=9.5,
        color=CORE,
        weight="bold",
        va="center",
    )
    v2 = [
        (1.75, 4.1, "rng.py", "fluxos nomeados", CORE, False),
        (4.0, 4.1, "customers.py", "perfis", CORE, False),
        (6.2, 4.1, "terminals.py", "perfis", CORE, False),
        (1.75, 3.05, "drift.py", "regime B", OPTIONAL, True),
        (4.0, 3.05, "daily_factors.py", "sazonalidade", OPTIONAL, True),
        (6.2, 3.05, "merchants.py", "categorias", OPTIONAL, True),
        (1.75, 2.0, "transactions.py", "legítimas", CORE, False),
        (4.0, 2.0, "fraud.py", "tamper", CORE, False),
        (6.2, 2.0, "injection.py", "inject", OPTIONAL, True),
    ]
    for bx, by, title, detail, color, dashed in v2:
        _box(ax, bx, by, 1.95, 0.7, title, detail, color, dashed)

    # legacy engine
    ax.add_patch(
        FancyBboxPatch(
            (7.85, 2.55),
            2.75,
            2.45,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor="none",
            edgecolor=MUTED,
            lw=1.2,
        )
    )
    ax.text(
        8.0,
        4.78,
        "simulator/legacy/ (v0.1)",
        fontsize=9.5,
        color=SECONDARY,
        weight="bold",
        va="center",
    )
    _box(ax, 9.22, 4.05, 2.3, 0.62, "generator.py", "", CORE)
    _box(ax, 9.22, 3.15, 2.3, 0.62, "fraud.py", "", CORE)

    _arrow(ax, (4.3, 5.47), (3.9, 5.02))
    ax.text(2.55, 5.2, 'engine="v2"', fontsize=8.5, color=SECONDARY, family="monospace")
    _arrow(ax, (7.0, 5.47), (8.9, 5.02))
    ax.text(
        8.0, 5.3, 'engine="legacy"', fontsize=8.5, color=SECONDARY, family="monospace"
    )

    # Shared modules and saving
    _box(
        ax,
        2.2,
        0.75,
        3.2,
        0.7,
        "simulator/location/",
        "cidades e sorteio de pontos (os dois motores)",
        CORE,
    )
    _box(
        ax,
        5.6,
        0.75,
        2.9,
        0.7,
        "simulator/ground_truth.py",
        "compromise_table (análise)",
        CORE,
    )
    _box(ax, 9.25, 0.75, 2.6, 0.7, "utils/data.py", "gravar e ler", OUTPUT)
    _arrow(ax, (2.2, 1.45), (2.2, 1.1))
    ax.plot([7.575, 7.575], [5.47, 0.75], color=SECONDARY, lw=1.1)
    _arrow(ax, (7.575, 0.75), (7.95, 0.75))
    ax.text(7.65, 1.25, "save()", fontsize=8.5, color=SECONDARY, family="monospace")

    for i, (color, label, dashed) in enumerate(
        [
            (CORE, "módulo", False),
            (OPTIONAL, "componente opcional", True),
            (CONFIG, "configuração", False),
            (OUTPUT, "saída", False),
        ]
    ):
        _legend(ax, 0.3 + 2.7 * i, 7.25, [(color, label, dashed)])
    fig.tight_layout()
    return fig


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("fluxo_geracao", generation_flow),
        ("arquitetura_modulos", module_architecture),
    ):
        fig = draw()
        for suffix in ("pdf", "svg"):
            fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)
        print(f"  {out / name}.pdf / .svg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/figures/arquitetura")
