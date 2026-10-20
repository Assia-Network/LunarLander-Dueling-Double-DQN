# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Assia-Network
"""Diagramas con tipografía y fórmulas compiladas por LaTeX mediante Matplotlib.
Requiere latex, dvipng y los paquetes amsmath/amssymb. Exporta PDF, SVG y PNG.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent / "fig"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
    }
)


def canvas(title, height):
    fig, ax = plt.subplots(figsize=(8.2, height * 0.98))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.95)
    ax.set(xlim=(0, 10), ylim=(0, height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=15)
    return fig, ax


def node(ax, x, y, w, h, text, color="#eff6ff"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=color,
            edgecolor="#94a3b8",
            linewidth=1,
        )
    )
    lines = text.split("\n")
    spacing = 0.28
    for index, line in enumerate(lines):
        ax.text(
            x,
            y + ((len(lines) - 1) / 2 - index) * spacing,
            line,
            ha="center",
            va="center",
            fontsize=11,
        )


def arrow(ax, a, b, text=None):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color="#64748b", lw=1.1)
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2 + 0.14,
            (a[1] + b[1]) / 2,
            text,
            fontsize=10,
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )


def export(fig, name):
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"{name}.{fmt}", dpi=190, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def arquitectura():
    f, a = canvas("Arquitectura Dueling Q-Network con ReLU", 11)
    node(
        a,
        5,
        10.35,
        8,
        0.95,
        r"Estado $s=(x,y,v_x,v_y,\theta,\omega,c_L,c_R)\in\mathbb{R}^{8}$"
        + "\n"
        + r"$c_L,c_R$: contacto de las patas",
        "#dcfce7",
    )
    node(
        a,
        5,
        8.9,
        6.5,
        1.2,
        r"Capa densa $8\rightarrow256$"
        + "\n"
        + r"$h_1=\operatorname{ReLU}(W_1s+b_1)$"
        + "\n"
        + r"$W_1\in\mathbb{R}^{256\times8},\quad b_1\in\mathbb{R}^{256}$",
    )
    node(
        a,
        5,
        7.25,
        6.5,
        1.2,
        r"Capa densa $256\rightarrow256$"
        + "\n"
        + r"$h_2=\operatorname{ReLU}(W_2h_1+b_2)$"
        + "\n"
        + r"$W_2\in\mathbb{R}^{256\times256},\quad b_2\in\mathbb{R}^{256}$",
    )
    node(
        a, 2.6, 5.55, 4.2, 1.1, r"Valor: $256\rightarrow1$" + "\n" + r"$V(s)=W_Vh_2+b_V$", "#f3e8ff"
    )
    node(
        a,
        7.4,
        5.55,
        4.2,
        1.1,
        r"Ventaja: $256\rightarrow4$" + "\n" + r"$A(s,\cdot)=W_Ah_2+b_A$",
        "#f3e8ff",
    )
    node(
        a,
        5,
        3.7,
        8.5,
        1.3,
        r"Combinación dueling"
        + "\n"
        + r"$Q(s,a)=V(s)+A(s,a)-\frac{1}{4}\sum_{b=0}^{3}A(s,b)$"
        + "\n"
        + r"Cuatro estimaciones de retorno futuro",
        "#fef3c7",
    )
    node(
        a,
        5,
        1.95,
        8.5,
        1.1,
        r"Acción: $a^*=\operatorname*{arg\,max}_{a}Q(s,a)$"
        + "\n"
        + r"0: reposo; 1: izquierdo; 2: principal; 3: derecho",
        "#dcfce7",
    )
    a.text(
        5,
        0.55,
        r"$69\,381$ parámetros entrenables\qquad $\operatorname{ReLU}(x)=\max(0,x)$",
        ha="center",
        va="center",
        fontsize=11,
    )
    for p, q in [
        ((5, 9.85), (5, 9.57)),
        ((5, 8.23), (5, 7.93)),
        ((5, 6.60), (2.6, 6.13)),
        ((5, 6.60), (7.4, 6.13)),
        ((2.6, 4.97), (4, 4.38)),
        ((7.4, 4.97), (6, 4.38)),
        ((5, 3.02), (5, 2.53)),
    ]:
        arrow(a, p, q)
    export(f, "arquitectura")


def entrenamiento():
    f, a = canvas("Entrenamiento por generaciones: Double DQN", 12)
    labels = [
        (11.3, 1.0, "Inicializar red online y red objetivo\nAdam y memoria de experiencias"),
        (
            9.8,
            1.0,
            "Iniciar entornos con semillas distintas\nMantener pesos fijos durante los vuelos",
        ),
        (
            8.3,
            1.0,
            r"Inferencia en lote y exploración $\varepsilon$-greedy"
            + "\n"
            + r"Guardar $(s,a,r,s\prime,d)$ en la memoria",
        ),
        (6.95, 0.65, "¿Terminaron todos los vuelos?"),
        (
            4.95,
            2.05,
            "Pausar vuelos y muestrear minibatches\n"
            + r"$a\prime=\operatorname*{arg\,max}_{a}Q_\theta(s\prime,a)$"
            + "\n"
            + r"$y=r+\gamma(1-d)Q_{\theta^-}(s\prime,a\prime)$"
            + "\n"
            + r"Minimizar $\operatorname{SmoothL1}(Q_\theta(s,a),y)$"
            + "\n"
            + r"Recortar gradiente y sincronizar $\theta^-$ periódicamente",
        ),
        (2.8, 1.0, "Validar cuando corresponda\nGuardar último modelo y conservar el mejor"),
        (1.4, 0.65, "¿Se alcanzó el límite de generaciones?"),
    ]
    for y, h, label in labels:
        node(
            a, 5, y, 6, h, label, "#f3e8ff" if y == 4.95 else "#fef3c7" if h == 0.65 else "#eff6ff"
        )
    for (y, h, _), (yy, hh, _) in zip(labels, labels[1:]):
        arrow(a, (5, y - h / 2 - 0.05), (5, yy + hh / 2 + 0.05), "Sí" if y == 6.95 else None)
    # Retornos fuera de los nodos y de sus etiquetas.
    a.plot([1.96, 1.1, 1.1, 1.96], [6.95, 6.95, 8.3, 8.3], color="#64748b", lw=1.1)
    arrow(a, (1.1, 8.3), (1.96, 8.3))
    a.text(0.85, 7.6, "No", rotation=90, ha="center")
    a.plot([8.04, 8.9, 8.9, 8.04], [1.4, 1.4, 9.8, 9.8], color="#64748b", lw=1.1)
    arrow(a, (8.9, 9.8), (8.04, 9.8))
    a.text(9.15, 5.3, "No", rotation=90, ha="center")
    arrow(a, (5, 1.02), (5, 0.55), "Sí")
    a.text(5, 0.3, "Finalizar y conservar los checkpoints", ha="center", fontsize=11)
    export(f, "flujo_entrenamiento")


def evaluacion():
    f, a = canvas("Protocolo de evaluación", 8.6)
    node(
        a,
        5,
        7.85,
        8,
        1.0,
        "Congelar checkpoint y registrar SHA-256\nReconstruir la red y sus pesos",
        "#dcfce7",
    )
    node(
        a,
        5,
        6.25,
        8,
        1.2,
        r"$5\times100$ semillas nuevas; máximo $1\,000$ pasos"
        + "\n"
        + r"LunarLander-v3; política $a=\arg\max Q(s,\cdot)$"
        + "\n"
        + "Registrar retorno, duración, acciones y resultado",
    )
    node(
        a,
        2.55,
        4.2,
        4.4,
        1.5,
        "Tasa de aterrizajes\nIntervalo de Wilson al 95\\%\nDesglose de fallos\nDistribución de retornos",
        "#f3e8ff",
    )
    node(
        a,
        7.45,
        4.2,
        4.4,
        1.5,
        "Referencia aleatoria\nMismas 100 semillas\ndel primer grupo\nComparar resultados",
        "#f3e8ff",
    )
    node(
        a,
        5,
        2.05,
        8,
        1.3,
        "Elegir el mayor retorno entre aterrizajes exitosos\nRepetir la semilla y verificar el resultado\nExportar GIF y trazas de decisiones",
        "#fef3c7",
    )
    arrow(a, (5, 7.3), (5, 6.9))
    arrow(a, (4, 5.6), (2.55, 5.0))
    arrow(a, (6, 5.6), (7.45, 5.0))
    arrow(a, (2.55, 3.4), (4, 2.75))
    arrow(a, (7.45, 3.4), (6, 2.75))
    a.text(
        5,
        0.6,
        "El GIF ilustra un caso seleccionado; las métricas resumen los 500 vuelos.",
        ha="center",
        fontsize=10,
    )
    export(f, "protocolo_evaluacion")


if __name__ == "__main__":
    arquitectura()
    entrenamiento()
    evaluacion()
    print("Diagramas compilados con LaTeX")
