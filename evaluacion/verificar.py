# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Assia-Network
"""Evaluación reproducible del checkpoint ReLU congelado. No entrena.
Ejecutar desde la raíz: .venv/bin/python evaluacion/verificar.py
--solo-figuras regenera las figuras y diagramas a partir de datos existentes.
"""

from pathlib import Path
import argparse, csv, hashlib, json, os, platform, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import torch
from torch import nn
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA, FIG, ANIM = ROOT / "datos", ROOT / "fig", ROOT / "animation"
SEED_STARTS = (1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000)
N_PER_GROUP = 100
ACTIONS = ["Sin motor", "Motor izquierdo", "Motor principal", "Motor derecho"]
COLORS = ["#718096", "#d97706", "#059669", "#7c3aed"]


class RedEvaluada(nn.Module):
    """Arquitectura congelada del checkpoint, independiente de cambios posteriores."""

    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(8, 256), nn.Linear(256, 256)])
        self.activations = nn.ModuleList([nn.ReLU(), nn.ReLU()])
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, 4)

    def forward(self, x):
        for layer, activation in zip(self.hidden_layers, self.activations):
            x = activation(layer(x))
        v, a = self.value_head(x), self.advantage_head(x)
        return v + a - a.mean(-1, keepdim=True)


def load_model():
    path = DATA / "modelo_evaluado.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(ck["hidden_sizes"]) == (256, 256) and ck["dueling"] and not ck["layer_norm"]
    assert (
        "return torch.relu(x)" in ck["activation_spec"]["source"]
    ), "Se requiere el checkpoint ReLU documentado."
    model = RedEvaluada().eval()
    model.load_state_dict(ck["weights"], strict=True)
    model.requires_grad_(False)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model, ck, hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(env, state, terminated, truncated):
    # Priorizar choque/salida antes del reposo físico.
    if terminated and env.unwrapped.game_over:
        return "Choque"
    if terminated and abs(float(state[0])) >= 1:
        return "Fuera de zona"
    if terminated and not env.unwrapped.lander.awake:
        return "Aterrizaje"
    if truncated:
        return "Tiempo agotado"
    return "Otra terminación"


def episode(model, seed, random_policy=False, record=False):
    env = gym.make(
        "LunarLander-v3", render_mode="rgb_array" if record else None, max_episode_steps=1000
    )
    rng = np.random.default_rng(seed + 77_000_000)
    states = []
    qs = []
    actions = []
    rewards = []
    frames = []
    try:
        state, _ = env.reset(seed=int(seed))
        if record:
            frames.append((0, env.render(), 0.0, 0))
        for step in range(1000):
            with torch.inference_mode():
                q = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
            assert np.isfinite(q).all()
            action = int(rng.integers(4)) if random_policy else int(q.argmax())
            states.append(state.copy())
            qs.append(q.copy())
            actions.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if record and ((step + 1) % 4 == 0 or terminated or truncated):
                frames.append((step + 1, env.render(), sum(rewards), action))
            if terminated or truncated:
                break
        status = outcome(env, state, terminated, truncated)
        qarray = np.asarray(qs)
        sorted_q = np.sort(qarray, axis=1)
        row = dict(
            seed=int(seed),
            return_total=float(sum(rewards)),
            steps=len(rewards),
            outcome=status,
            success=int(status == "Aterrizaje"),
            return_ge_200=int(sum(rewards) >= 200),
            mean_q_margin=float(np.mean(sorted_q[:, -1] - sorted_q[:, -2])),
        )
        row.update({f"action_{i}_count": actions.count(i) for i in range(4)})
        trace = dict(
            states=np.array(states),
            q=qarray,
            actions=np.array(actions),
            rewards=np.array(rewards),
            final_state=state,
            seed=np.array(seed),
        )
        return row, trace, frames
    finally:
        env.close()


def wilson(successes, n):
    z = 1.95996398454
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def summarize(rows):
    vals = np.array([r["return_total"] for r in rows])
    wins = sum(r["success"] for r in rows)
    n = len(rows)
    return dict(
        n=n,
        successes=wins,
        failures=n - wins,
        success_rate=wins / n,
        ci95_wilson=wilson(wins, n),
        mean_return=float(vals.mean()),
        std_return=float(vals.std(ddof=1)),
        median_return=float(np.median(vals)),
        return_ge_200=sum(r["return_ge_200"] for r in rows),
        outcomes={
            key: sum(r["outcome"] == key for r in rows)
            for key in sorted({r["outcome"] for r in rows})
        },
    )


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate():
    model, ck, digest = load_model()
    rows = []
    baseline = []
    for group, start in enumerate(SEED_STARTS, 1):
        batch = []
        for seed in range(start, start + N_PER_GROUP):
            row, _, _ = episode(model, seed)
            row = {"group": group, **row}
            rows.append(row)
            batch.append(row)
        write_csv(DATA / "resultados.csv", rows)
        print(f'Grupo {group}: {sum(r["success"] for r in batch)}/{N_PER_GROUP}', flush=True)
    for seed in range(SEED_STARTS[0], SEED_STARTS[0] + N_PER_GROUP):
        row, _, _ = episode(model, seed, random_policy=True)
        baseline.append(row)
    write_csv(DATA / "azar.csv", baseline)
    successful = [r for r in rows if r["success"]]
    if not successful:
        raise RuntimeError("La muestra no contiene aterrizajes exitosos para generar la animación.")
    best = max(successful, key=lambda r: r["return_total"])
    worst = min(rows, key=lambda r: r["return_total"])
    for label, row in [("mejor", best), ("peor", worst)]:
        repeat, trace, frames = episode(model, row["seed"], record=label == "mejor")
        assert (
            repeat["outcome"] == row["outcome"]
            and abs(repeat["return_total"] - row["return_total"]) < 1e-5
        ), "Repetición no determinista"
        np.savez_compressed(DATA / f"trayectoria_{label}.npz", **trace)
        if label == "mejor":
            make_gif(frames, row)
    result = dict(
        date=datetime.now(ZoneInfo("America/Lima")).isoformat(),
        checkpoint_sha256=digest,
        model=dict(
            hidden_sizes=[256, 256],
            activation="ReLU",
            dueling=True,
            layer_norm=False,
            parameters=sum(p.numel() for p in model.parameters()),
            steps=ck["steps"],
            updates=ck["updates"],
            generation=ck.get("generations_completed"),
            episodes=ck["training_episodes"],
        ),
        protocol=dict(
            seed_starts=SEED_STARTS,
            n_per_group=N_PER_GROUP,
            max_steps=1000,
            epsilon=0,
            gravity=-10.0,
            enable_wind=False,
            deterministic_argmax=True,
            criterion="Terminación, módulo en reposo, sin game_over ni salida lateral",
        ),
        aggregate=summarize(rows),
        groups=[
            dict(group=g, **summarize([r for r in rows if r["group"] == g])) for g in range(1, 6)
        ],
        random_baseline=summarize(baseline),
        paired_policy_group1=summarize(rows[:100]),
        best=best,
        worst=worst,
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            gymnasium=gym.__version__,
            matplotlib=matplotlib.__version__,
            platform=platform.platform(),
        ),
    )
    (DATA / "resumen.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["aggregate"], ensure_ascii=False), flush=True)


def make_gif(frames, row):
    """Compone cuadros de Gymnasium con una cabecera tipográfica LaTeX."""
    images, durations = [], []
    settings = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
    }
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(9, 1), dpi=100, facecolor="#0a1424")
        fig.text(
            0.5,
            0.72,
            f'Lunar RL · Aterrizaje exitoso · Semilla {row["seed"]}',
            ha="center",
            va="center",
            fontsize=17,
            color="#8ce8c5",
        )
        label = fig.text(0.5, 0.30, "", ha="center", va="center", fontsize=14, color="white")
        for j, (step, frame, total, action) in enumerate(frames):
            label.set_text(
                f"Tiempo: {step / 50:.2f} s   |   Retorno: {total:+.1f}   |   {ACTIONS[action]}"
            )
            fig.canvas.draw()
            header = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            image = Image.new("RGB", (900, 700), "#0a1424")
            image.paste(header, (0, 0))
            image.paste(
                Image.fromarray(frame).resize((900, 600), Image.Resampling.LANCZOS), (0, 100)
            )
            images.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
            durations.append((frames[j + 1][0] - step) * 20 if j + 1 < len(frames) else 1200)
        plt.close(fig)
    images[0].save(
        ANIM / "mejor_aterrizaje.gif",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    images[-1].convert("RGB").save(FIG / "mejor_aterrizaje.png")


def save(fig, name):
    from matplotlib.text import Text

    for item in fig.findobj(Text):
        text = (
            item.get_text()
            .replace("%", r"\%")
            .replace("≥", r"$\geq$")
            .replace("−", "-")
            .replace("·", r"$\cdot$")
        )
        item.set_text(text)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.15, facecolor="white")
    fig.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots():
    result = json.loads((DATA / "resumen.json").read_text())
    rows = list(csv.DictReader((DATA / "resultados.csv").open()))
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlepad": 14,
            "axes.labelpad": 9,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    groups = result["groups"]
    means = np.array([g["success_rate"] for g in groups]) * 100
    low = np.array([g["ci95_wilson"][0] for g in groups]) * 100
    high = np.array([g["ci95_wilson"][1] for g in groups]) * 100
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(range(1, 6), means, color="#059669", width=0.55)
    ax.errorbar(
        range(1, 6),
        means,
        yerr=[means - low, high - means],
        fmt="none",
        ecolor="#334155",
        capsize=6,
    )
    for i, g in enumerate(groups, 1):
        ax.text(i, 4, f'{g["successes"]}/100', ha="center", color="white", weight="bold")
    ax.set(
        ylim=(0, 105),
        xticks=range(1, 6),
        xlabel="Grupo de 100 semillas nuevas",
        ylabel="Aterrizajes exitosos (%)",
        title="Un mismo checkpoint en cinco grupos independientes",
    )
    fig.suptitle(
        "Barras: tasa observada · líneas: intervalo de Wilson al 95 %", fontsize=11, color="#64748b"
    )
    save(fig, "aciertos_semillas")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    win = [float(r["return_total"]) for r in rows if int(r["success"])]
    fail = [float(r["return_total"]) for r in rows if not int(r["success"])]
    axs[0].hist(
        [win, fail],
        bins=30,
        stacked=True,
        label=["Aterrizaje", "Fallo"],
        color=["#059669", "#e76f51"],
    )
    axs[0].axvline(200, color="#475569", ls="--", label="Retorno 200 (criterio distinto)")
    axs[0].legend(fontsize=9)
    axs[0].set(title="Distribución del retorno", xlabel="Recompensa acumulada", ylabel="Vuelos")
    fig.get_layout_engine().set(wspace=0.15)
    counts = result["aggregate"]["outcomes"]
    axs[1].barh(
        list(counts),
        list(counts.values()),
        color=["#059669" if k == "Aterrizaje" else "#e76f51" for k in counts],
    )
    for i, v in enumerate(counts.values()):
        axs[1].text(v + 2, i, str(v), va="center")
    axs[1].set(
        title="Cómo terminó cada vuelo",
        xlabel="Número de vuelos",
        xlim=(0, max(counts.values()) * 1.16),
    )
    save(fig, "resultados")
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), layout="constrained")
    pair = [result["paired_policy_group1"], result["random_baseline"]]
    labels = ["Red entrenada", "Acciones aleatorias"]
    axs[0].bar(labels, [d["success_rate"] * 100 for d in pair], color=["#059669", "#94a3b8"])
    axs[0].set(ylabel="Aterrizajes (%)", ylim=(0, 105), title="Mismas 100 semillas de entorno")
    for i, d in enumerate(pair):
        axs[0].text(i, d["success_rate"] * 100 + 2, f'{d["successes"]}/100', ha="center")
    axs[1].bar(labels, [d["mean_return"] for d in pair], color=["#059669", "#94a3b8"])
    axs[1].axhline(0, color="#64748b", lw=1)
    axs[1].set(ylabel="Retorno medio", title="La referencia aleatoria no aprende")
    save(fig, "comparacion_azar")
    trace = np.load(DATA / "trayectoria_mejor.npz")
    t = np.arange(len(trace["actions"])) / 50
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), layout="constrained", sharex=True)
    for i in range(4):
        axs[0].plot(t, trace["q"][:, i], color=COLORS[i], label=ACTIONS[i], lw=1.2)
    axs[0].set(
        title=f'Decisiones del aterrizaje · semilla {result["best"]["seed"]}', ylabel="Q estimado"
    )
    axs[0].legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False)
    axs[0].set_ylim(top=trace["q"].max() * 1.38)
    axs[1].scatter(t, trace["actions"], c=[COLORS[i] for i in trace["actions"]], s=7)
    axs[1].set(yticks=range(4), yticklabels=ACTIONS, ylabel="Acción ejecutada")
    q = np.sort(trace["q"], axis=1)
    axs[2].plot(t, q[:, -1] - q[:, -2], color="#2563eb")
    axs[2].set(
        xlabel="Tiempo simulado (s)",
        ylabel="Margen Q",
        title=r"Margen entre las dos acciones mejor valoradas: $Q_{(1)}-Q_{(2)}$",
    )
    fig.get_layout_engine().set(hspace=0.13)
    save(fig, "decisiones_mejor")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for label, color in [("mejor", "#059669"), ("peor", "#e76f51")]:
        tr = np.load(DATA / f"trayectoria_{label}.npz")
        st = tr["states"]
        tt = np.arange(len(st)) / 50
        axs[0].plot(st[:, 0], st[:, 1], color=color, label=f"{label.capitalize()} retorno")
        axs[0].scatter(st[0, 0], st[0, 1], color=color, marker="o")
        axs[0].scatter(tr["final_state"][0], tr["final_state"][1], color=color, marker="x", s=65)
        axs[1].plot(tt, np.cumsum(tr["rewards"]), color=color, label=label.capitalize())
    axs[0].set(
        title="Trayectorias observadas",
        xlabel="Posición x (observación normalizada)",
        ylabel="Posición y (observación normalizada)",
    )
    axs[0].legend()
    axs[1].set(
        title="Acumulación de recompensa", xlabel="Tiempo simulado (s)", ylabel="Retorno acumulado"
    )
    axs[1].legend()
    save(fig, "trayectorias")
    history = list(csv.DictReader((DATA / "historial_generaciones.csv").open()))
    validations = [
        json.loads(line)
        for line in (DATA / "validacion_entrenamiento.jsonl").read_text().splitlines()
        if line.strip()
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.2), layout="constrained")
    gen = np.array([int(r["generation"]) for r in history])
    returns = np.array([float(r["mean_return"]) for r in history])
    smooth = np.array([returns[max(0, i - 19) : i + 1].mean() for i in range(len(returns))])
    ax.plot(gen, returns, color="#cbd5e1", lw=0.6, label="Retorno por generación")
    ax.plot(gen, smooth, color="#2563eb", lw=1.4, label="Media móvil (20 generaciones)")
    ax.plot(
        [v["training_episodes"] / 20 for v in validations],
        [v["mean_reward"] for v in validations],
        color="#059669",
        lw=1,
        label="Validación (20 semillas fijas)",
    )
    ax.axvline(
        result["model"]["generation"],
        color="#9333ea",
        ls="--",
        label=f"Checkpoint (generación {result['model']['generation']})",
    )
    ax.set(
        title="Evolución del retorno durante el entrenamiento",
        xlabel="Generación (20 entornos)",
        ylabel="Retorno medio",
    )
    legend = ax.legend(
        fontsize=13, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=1,
        labelcolor="#0f172a", handlelength=3, handletextpad=0.8,
        columnspacing=2, labelspacing=0.8, borderpad=0.9,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.8)
        handle.set_alpha(1)
    legend.legend_handles[0].set_color("#94a3b8")
    save(fig, "historial_entrenamiento")
    subprocess.run([sys.executable, str(ROOT / "diagramas.py")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-figuras", action="store_true")
    parser.add_argument("--solo-animacion", action="store_true")
    args = parser.parse_args()
    for folder in (DATA, FIG, ANIM):
        folder.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    if args.solo_animacion:
        model, _, _ = load_model()
        best = json.loads((DATA / "resumen.json").read_text())["best"]
        repeated, _, frames = episode(model, best["seed"], record=True)
        assert repeated["outcome"] == best["outcome"]
        assert abs(repeated["return_total"] - best["return_total"]) < 1e-5
        make_gif(frames, best)
        sys.exit(0)
    if not args.solo_figuras:
        evaluate()
    plots()
    subprocess.run([sys.executable, str(ROOT / "documentar.py")], check=True)
