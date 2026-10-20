# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Assia-Network
"""Selector de modelos y evaluación de LunarLander. Nunca entrena ni modifica pesos.
Abrir: .venv/bin/python evaluar_modelos.py
Sin ventana: ... --model experiments/.../model.pt --attempts 10 --headless
"""

from pathlib import Path
import argparse, csv, hashlib, io, json, os, re, time, uuid
import numpy as np
import torch
from torch import nn
import gymnasium as gym
import pygame
from ui_orbital import UI, content, columns, BG, TEXT, MUTED, MINT, RED, PURPLE
from lunar_rl import (
    QNetwork,
    MiActivacion,
    activation_compatible,
    classify,
    INPUT_NAMES,
    ACTION_NAMES,
)

ROOT = Path(__file__).resolve().parent


def discover_models(root=None):
    root = Path(root) if root is not None else ROOT
    paths = []
    for folder in ("experiments", "models"):
        if (root / folder).exists():
            paths.extend((root / folder).rglob("*.pt"))

    def order(path):
        try:
            cfg = json.loads((path.parent / "config.json").read_text())
            smoke = bool(cfg.get("smoke", cfg.get("smoke_test", False)))
        except (OSError, ValueError):
            smoke = False
        return (not smoke, path.stat().st_mtime)

    return sorted(paths, key=order, reverse=True)


def load_model(path):
    """Reconstruye cada checkpoint a partir de sus metadatos."""
    path = Path(path)
    data = path.read_bytes()  # Lectura única para evitar cambios del archivo durante la carga.
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("El archivo no contiene un checkpoint reconocido.")
    sidecar = path.parent / "config.json"
    config = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    config = {**config, **checkpoint.get("config", {})}
    environment = checkpoint.get("environment", config.get("environment", "LunarLander-v3"))
    if environment != "LunarLander-v3":
        raise ValueError("El modelo no corresponde a LunarLander-v3.")
    weights = checkpoint.get("weights", checkpoint.get("state_dict"))
    if (
        weights is None
        and checkpoint
        and all(isinstance(v, torch.Tensor) for v in checkpoint.values())
    ):
        weights = checkpoint
    if not isinstance(weights, dict):
        raise ValueError("No se encontraron pesos. Usa un archivo .pt del proyecto.")
    spec = checkpoint.get("activation_spec", config.get("activation_spec"))
    if spec is not None:
        if not activation_compatible(spec):
            raise ValueError(
                "La clase MiActivacion cambió desde el entrenamiento. Restaura la definición de activacion_usada.txt para evaluar este modelo."
            )
        description = "MiActivacion · fórmula personalizada"
        factory = MiActivacion
    else:
        legacy_classes = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "SiLU": nn.SiLU,
            "GELU": nn.GELU,
            "ELU": nn.ELU,
            "LeakyReLU": nn.LeakyReLU,
        }
        description = str(
            checkpoint.get(
                "activation_name", checkpoint.get("activation", config.get("activation", ""))
            )
        )
        names = [name for name in legacy_classes if re.search(r"\b" + name + r"\b", description)]
        if len(names) != 1:
            raise ValueError(
                "No se puede reconstruir la activación. Falta metadata o es una función personalizada."
            )
        name = names[0]
        factory = legacy_classes[name]
        # Conserva los parámetros presentes en metadatos de versiones anteriores.
        if name == "LeakyReLU":
            match = re.search(r"negative_slope=([0-9.eE+-]+)", description)
            slope = float(match.group(1)) if match else 0.01
            factory = lambda: nn.LeakyReLU(slope)
        if name == "ELU":
            match = re.search(r"alpha=([0-9.eE+-]+)", description)
            alpha = float(match.group(1)) if match else 1.0
            factory = lambda: nn.ELU(alpha)
        if name == "GELU" and "approximate='tanh'" in description:
            factory = lambda: nn.GELU(approximate="tanh")
    if "hidden_layers.0.weight" in weights:
        indices = sorted(
            int(k.split(".")[1]) for k in weights if re.fullmatch(r"hidden_layers\.\d+\.weight", k)
        )
        if indices != list(range(len(indices))):
            raise ValueError("Índices de capas inválidos.")
        hidden = tuple(weights[f"hidden_layers.{i}.weight"].shape[0] for i in indices)
        input_dim = weights["hidden_layers.0.weight"].shape[1]
        dueling = "advantage_head.weight" in weights
        output_key = "advantage_head.weight" if dueling else "output_layer.weight"
        output_dim = weights[output_key].shape[0]
        layer_norm = any(k.startswith("norms.") for k in weights)
        model = QNetwork(
            input_dim, hidden, output_dim, factory, dueling=dueling, layer_norm=layer_norm
        )
    elif "layers.0.weight" in weights:
        # Compatibilidad con checkpoints de arquitectura secuencial.
        indices = sorted(
            int(k.split(".")[1]) for k in weights if re.fullmatch(r"layers\.\d+\.weight", k)
        )
        if indices != list(range(0, 2 * len(indices), 2)):
            raise ValueError("Arquitectura secuencial no reconocida.")
        input_dim = weights["layers.0.weight"].shape[1]
        output_dim = weights[f"layers.{indices[-1]}.weight"].shape[0]
        hidden = tuple(weights[f"layers.{i}.weight"].shape[0] for i in indices[:-1])
        modules = []
        for i in indices:
            w = weights[f"layers.{i}.weight"]
            modules.append(nn.Linear(w.shape[1], w.shape[0]))
            if i != indices[-1]:
                modules.append(factory())
        model = nn.Module()
        model.layers = nn.Sequential(*modules)
        model.forward = model.layers.forward
        dueling = False
        layer_norm = False
    else:
        raise ValueError(
            "Arquitectura no reconocida. Los metadatos no permiten reconstruir las capas y activaciones."
        )
    if input_dim != 8 or output_dim != 4:
        raise ValueError("LunarLander discreto necesita 8 entradas y 4 salidas.")
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.requires_grad_(False)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("El modelo contiene pesos no finitos.")
    with torch.inference_mode():
        q = model(torch.zeros(1, 8))
        if q.shape != (1, 4) or not torch.isfinite(q).all():
            raise ValueError("Salida de red inválida.")
    info = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hidden_sizes": list(hidden),
        "activation": description,
        "dueling": dueling,
        "layer_norm": layer_norm,
        "environment": environment,
        "parameters": sum(p.numel() for p in model.parameters()),
        "smoke_test": bool(config.get("smoke", config.get("smoke_test", False))),
    }
    return model, info


class Evaluation:
    def __init__(self, model, info, attempts=10, seed=10000, max_steps=1000, output_root=None):
        if attempts < 1 or max_steps < 1:
            raise ValueError("Intentos y pasos deben ser positivos.")
        self.model = model
        self.info = info
        self.attempts = attempts
        self.base_seed = seed
        self.max_steps = max_steps
        self.rows = []
        self.actions = []
        self.last_q = np.zeros(4)
        self.completed = False
        self.pending_reset = False
        self.status = "EN VUELO"
        self.folder = (Path(output_root) if output_root else ROOT / "evaluations") / (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        self.folder.mkdir(parents=True)
        (self.folder / "attempts").mkdir()
        self.env = gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=max_steps)
        self.state, _ = self.env.reset(seed=seed)
        self.total = 0.0
        self.steps = 0
        self.frame = self.env.render()
        self.write_summary("en curso")

    def step(self):
        if self.completed:
            return
        if self.pending_reset:
            self.state, _ = self.env.reset(seed=self.base_seed + len(self.rows))
            self.total = 0.0
            self.steps = 0
            self.actions = []
            self.pending_reset = False
        with torch.inference_mode():
            q = self.model(torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0))[0]
        if not torch.isfinite(q).all():
            raise ValueError("Valores Q no finitos durante la evaluación.")
        self.last_q = q.numpy().copy()
        action = int(q.argmax())
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        self.actions.append(action)
        self.total += reward
        self.steps += 1
        self.status = classify(self.env, self.state, terminated, truncated)
        if terminated or truncated:
            self.frame = self.env.render()
            number = len(self.rows) + 1
            row = {
                "attempt": number,
                "seed": self.base_seed + number - 1,
                "reward": float(self.total),
                "steps": self.steps,
                "outcome": self.status,
                "success": self.status == "ATERRIZAJE",
                "return_ge_200": bool(self.total >= 200),
            }
            self.rows.append(row)
            (self.folder / "attempts" / f"{number:03}.json").write_text(
                json.dumps({**row, "actions": self.actions}, indent=2)
            )
            pygame.image.save(
                pygame.surfarray.make_surface(self.frame.transpose(1, 0, 2)),
                str(self.folder / "attempts" / f"{number:03}.png"),
            )
            with (self.folder / "results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            self.completed = len(self.rows) >= self.attempts
            self.pending_reset = True
            self.write_summary("completada" if self.completed else "en curso")

    def write_summary(self, status):
        successes = sum(r["success"] for r in self.rows)
        n = len(self.rows)
        ranked = sorted(self.rows, key=lambda r: r["reward"], reverse=True)
        summary = {
            "status": status,
            "model": self.info,
            "requested_attempts": self.attempts,
            "completed_attempts": n,
            "successes": successes,
            "failures": n - successes,
            "success_fraction": f"{successes}/{n}" if n else "0/0",
            "success_rate": successes / n if n else None,
            "mean_reward": float(np.mean([r["reward"] for r in self.rows])) if n else None,
            "base_seed": self.base_seed,
            "max_steps": self.max_steps,
            "epsilon": 0,
            "criterion": "Aterrizaje: terminación con módulo en reposo según Box2D; resto = fallo.",
            "best_attempt": ranked[0] if n else None,
            "worst_attempt": ranked[-1] if n else None,
            "ranking_by_reward": [r["attempt"] for r in ranked],
            "failures_detail": [r for r in self.rows if not r["success"]],
        }
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        report = [
            "# Evaluación LunarLander",
            f'Modelo: {self.info["path"]}',
            f"Estado: {status}",
            f"**Aciertos: {successes}/{n} evaluados. Fallos: {n-successes}/{n}. Objetivo: {self.attempts}.**",
            "Éxito = módulo en reposo según el entorno. Evaluación sin exploración ni entrenamiento.",
            "Mejor/peor se ordenan por recompensa total, no por confianza de una predicción.",
            "",
            "| Intento | Semilla | Retorno | Resultado |",
            "|---|---|---|---|",
        ]
        report.extend(
            f'| {r["attempt"]} | {r["seed"]} | {r["reward"]:.1f} | {r["outcome"]} |' for r in ranked
        )
        (self.folder / "INFORME.md").write_text("\n\n".join(report), encoding="utf-8")

    def close(self):
        self.write_summary("completada" if self.completed else "interrumpida")
        self.env.close()


class EvaluatorUI:
    def __init__(self):
        pygame.init()
        torch.set_num_threads(1)
        self.screen = pygame.display.set_mode((1280, 900), pygame.RESIZABLE)
        pygame.display.set_caption("ORBITAL · Selector y evaluación de modelos")
        self.font = pygame.font.SysFont("DejaVu Sans", 16)
        self.small = pygame.font.SysFont("DejaVu Sans", 13)
        self.title = pygame.font.SysFont("DejaVu Sans", 26, bold=True)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.paths = discover_models()
        self.index = 0
        self.offset = 0
        self.attempts = 10
        self.seed = 10000
        self.model = None
        self.info = None
        self.session = None
        self.replay_env = None
        self.replay_data = None
        self.replay_position = 0
        self.replay_done = False
        self.hold_until = 0
        self.speed = 1
        self.buttons = []
        self.error = ""
        self.page = "selector"
        self.select(0)

    def select(self, index):
        self.index = max(0, min(index, len(self.paths) - 1))
        self.error = ""
        self.model = None
        self.info = None
        if not self.paths:
            self.error = "No hay modelos .pt en experiments/ o models/."
            return
        try:
            self.model, self.info = load_model(self.paths[self.index])
        except Exception as exc:
            self.error = "Modelo no compatible: " + str(exc)

    def text(self, text, x, y, color="#dae6f4", small=False, big=False, width=None):
        font = self.title if big else self.small if small else self.font
        text = str(text)
        if width:
            while len(text) > 3 and font.size(text)[0] > width:
                text = text[:-4] + "…"
        self.screen.blit(font.render(text, True, color), (x, y))

    def button(self, label, rect, action, enabled=True):
        rect = pygame.Rect(rect)
        self.ui.card(rect, "#203e38" if enabled else "#182433")
        self.ui.text(
            label, rect.inflate(-16, 0), 12, MINT if enabled else MUTED, True, align="center"
        )
        if enabled:
            self.buttons.append((rect, action))

    def begin(self):
        if self.model is None:
            return
        if self.session:
            self.session.close()
        self.session = Evaluation(self.model, self.info, self.attempts, self.seed)
        self.page = "evaluation"
        self.paused = False
        self.error = ""
        self.hold_until = 0

    def to_selector(self):
        self.stop_replay()
        if self.session:
            self.session.close()
            self.session = None
        self.page = "selector"
        self.paths = discover_models()
        self.select(self.index)

    def change_attempts(self, amount):
        self.attempts = max(1, min(1000, self.attempts + amount))

    def start_replay(self, best=True):
        if not self.session or not self.session.rows:
            return
        self.stop_replay()
        row = (max if best else min)(self.session.rows, key=lambda r: r["reward"])
        self.replay_data = json.loads(
            (self.session.folder / "attempts" / f'{row["attempt"]:03}.json').read_text()
        )
        self.replay_env = gym.make(
            "LunarLander-v3", render_mode="rgb_array", max_episode_steps=self.session.max_steps
        )
        self.replay_state, _ = self.replay_env.reset(seed=row["seed"])
        self.replay_position = 0
        self.replay_done = False
        self.paused = False
        self.replay_total = 0.0
        self.page = "replay"

    def stop_replay(self):
        if self.replay_env:
            self.replay_env.close()
        self.replay_env = None
        self.replay_data = None
        self.page = "evaluation" if self.session else "selector"

    def events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (max(980, e.w), max(760, e.h)), pygame.RESIZABLE
                )
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for rect, action in list(self.buttons):
                    if rect.collidepoint(e.pos):
                        action()
                        break
            elif e.type == pygame.MOUSEWHEEL and self.page == "selector":
                self.offset = max(0, min(max(0, len(self.paths) - 1), self.offset - e.y))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if self.page == "selector":
                        self.running = False
                    else:
                        self.to_selector()
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_RETURN and self.page == "selector":
                    self.begin()
                elif self.page == "selector" and e.key in (pygame.K_UP, pygame.K_DOWN):
                    self.select(self.index + (-1 if e.key == pygame.K_UP else 1))
                    self.offset = max(0, self.index - 3)
                elif e.key == pygame.K_r and self.page == "selector":
                    self.paths = discover_models()
                    self.select(0)
                    self.offset = 0
                elif e.key == pygame.K_1:
                    self.speed = 1
                elif e.key == pygame.K_2:
                    self.speed = 10

    def draw_selector(self):
        ui = self.ui
        area = content(self.screen)
        left, right = columns(area, 0.59)
        ui.card(left)
        ui.card(right)
        ui.text("Modelos guardados", (left.x + 20, left.y + 16, left.w - 40, 28), 18, TEXT, True)
        ui.text(
            f"{len(self.paths)} disponibles · selecciona un archivo",
            (left.x + 20, left.y + 48, left.w - 40, 20),
            12,
            MUTED,
        )
        visible = max(1, (left.h - 142) // 68)
        self.offset = min(self.offset, max(0, len(self.paths) - visible))
        for slot, index in enumerate(
            range(self.offset, min(len(self.paths), self.offset + visible))
        ):
            path = self.paths[index]
            rect = pygame.Rect(left.x + 16, left.y + 86 + 68 * slot, left.w - 32, 58)
            ui.card(rect, "#203e38" if index == self.index else "#172538")
            ui.text(
                path.parent.name,
                (rect.x + 14, rect.y + 7, rect.w - 28, 22),
                13,
                MINT if index == self.index else TEXT,
                True,
            )
            ui.text(path.name, (rect.x + 14, rect.y + 31, rect.w - 28, 18), 11, MUTED)
            self.buttons.append((rect, lambda i=index: self.select(i)))
        if not self.paths:
            ui.text(
                "Todavía no hay modelos guardados",
                (left.x + 20, left.centery - 12, left.w - 40, 24),
                13,
                MUTED,
                align="center",
            )
        self.button(
            "Actualizar lista · R",
            (left.x + 16, left.bottom - 54, left.w - 32, 38),
            lambda: (
                setattr(self, "paths", discover_models()),
                self.select(0),
                setattr(self, "offset", 0),
            ),
        )
        x = right.x + 22
        room = right.w - 44
        y = right.y
        ui.text("Preparar evaluación", (x, y + 16, room, 28), 18, TEXT, True, align="center")
        if self.info:
            ui.text(
                "8 → " + " → ".join(map(str, self.info["hidden_sizes"])) + " → 4",
                (x, y + 60, room, 28),
                18,
                MINT,
                True,
                align="center",
            )
            ui.text(
                self.info["activation"].replace("\n", " "),
                (x, y + 92, room, 20),
                11,
                MUTED,
                align="center",
            )
            flags = ("Dueling" if self.info["dueling"] else "DQN") + (
                " · LayerNorm" if self.info["layer_norm"] else ""
            )
            ui.text(
                f'{self.info["parameters"]:,} parámetros · {flags}',
                (x, y + 119, room, 20),
                11,
                MUTED,
                align="center",
            )
            ui.text(
                (
                    "Modelo de prueba corta"
                    if self.info["smoke_test"]
                    else "Pesos listos para evaluar"
                ),
                (x, y + 144, room, 20),
                11,
                MINT,
                align="center",
            )
        else:
            ui.text(
                "Selecciona un modelo compatible", (x, y + 82, room, 24), 12, MUTED, align="center"
            )
        ui.rule(x, y + 183, room)
        ui.text("NÚMERO DE INTENTOS", (x, y + 198, room, 20), 11, MUTED, align="center")
        ui.text(self.attempts, (x, y + 222, room, 40), 30, TEXT, True, align="center")
        gap = 10
        bw = (room - 2 * gap) // 3
        for i, (label, amount) in enumerate([("− 1", -1), ("+ 1", 1), ("+ 10", 10)]):
            self.button(
                label,
                (x + i * (bw + gap), y + 278, bw, 38),
                lambda a=amount: self.change_attempts(a),
            )
        ui.text(f"Semilla inicial · {self.seed}", (x, y + 338, room, 24), 13, TEXT, align="center")
        bw = (room - 10) // 2
        self.button(
            "− 100", (x, y + 376, bw, 38), lambda: setattr(self, "seed", max(0, self.seed - 100))
        )
        self.button(
            "+ 100",
            (x + bw + 10, y + 376, bw, 38),
            lambda: setattr(self, "seed", min(2**31 - 2000, self.seed + 100)),
        )
        self.button(
            "Iniciar evaluación →",
            (x, right.bottom - 112, room, 46),
            self.begin,
            self.model is not None,
        )
        ui.text(
            "Acierto = aterrizaje con módulo en reposo.",
            (x, right.bottom - 54, room, 18),
            10,
            MUTED,
            align="center",
        )
        ui.text(
            "Sin exploración · hasta 1.000 pasos por intento",
            (x, right.bottom - 32, room, 18),
            10,
            MUTED,
            align="center",
        )
        ui.text(
            "RUEDA / ↑ ↓  elegir modelo     ENTER  iniciar     R  actualizar     ESC  salir",
            (area.x, area.bottom + 12, area.w, 20),
            12,
            MUTED,
            align="center",
        )
        ui.text(
            self.error or "Cada evaluación guarda sus resultados y las repeticiones de los vuelos.",
            (area.x, area.bottom + 37, area.w, 20),
            11,
            RED if self.error else MUTED,
            align="center",
        )

    def draw_evaluation(self):
        ui = self.ui
        area = content(self.screen)
        s = self.session
        n = len(s.rows)
        wins = sum(r["success"] for r in s.rows)
        gap = 18
        kw = (area.w - 2 * gap) // 3
        for i, (label, value, detail, color) in enumerate(
            [
                ("ACIERTOS", f"{wins}/{n}", "Aterrizajes en los vuelos evaluados", MINT),
                ("FALLOS", f"{n-wins}/{n}", "Choques, salidas o límite de pasos", RED),
                (
                    "PROGRESO",
                    f"{n}/{s.attempts}",
                    "Evaluación finalizada" if s.completed else "Intentos completados",
                    PURPLE,
                ),
            ]
        ):
            ui.metric((area.x + i * (kw + gap), area.y, kw, 100), label, value, detail, color)
        middle = pygame.Rect(area.x, area.y + 116, area.w, area.h - 242)
        left, right = columns(middle, 0.66)
        replay = self.page == "replay"
        if replay:
            frame = self.replay_env.render()
            title = f'Repetición · vuelo {self.replay_data["attempt"]:03}'
            status = "Repetición finalizada" if self.replay_done else "Reproduciendo"
            total = self.replay_total
            steps = self.replay_position
        else:
            frame = s.env.render()
            title = f"Vuelo {min(n if s.pending_reset else n+1,s.attempts):03} / {s.attempts}"
            status = s.status
            total = s.total
            steps = s.steps
        ui.flight(
            left,
            frame,
            title,
            status,
            [
                ("RECOMPENSA", f"{total:+.1f}"),
                ("PASO", steps),
                ("MODO", "Repetición" if replay else "Evaluación"),
            ],
        )
        ui.card(right)
        x = right.x + 16
        room = right.w - 32
        ui.text("Resultados por vuelo", (x, right.y + 12, room, 24), 14, TEXT, True)
        ui.text("INTENTO / RESULTADO", (x, right.y + 43, room - 75, 18), 10, MUTED)
        ui.text("RETORNO", (right.right - 86, right.y + 43, 70, 18), 10, MUTED, align="right")
        count = max(1, (right.h - 211) // 24)
        for i, row in enumerate(s.rows[-count:]):
            y = right.y + 70 + i * 24
            c = MINT if row["success"] else RED
            ui.text(f'{row["attempt"]:03}  {row["outcome"]}', (x, y, room - 80, 22), 11, c)
            ui.text(f'{row["reward"]:+.1f}', (right.right - 86, y, 70, 22), 12, c, align="right")
        if not s.rows:
            ui.text(
                "Esperando el primer resultado…",
                (x, right.y + 82, room, 24),
                11,
                MUTED,
                align="center",
            )
        qy = right.bottom - 128
        ui.rule(x, qy - 8, room)
        ui.text("Q · valor estimado de cada acción", (x, qy, room, 20), 10, MUTED)
        if replay:
            ui.text(
                "Se repiten las acciones guardadas.",
                (x, qy + 34, room, 22),
                11,
                MUTED,
                align="center",
            )
            ui.text(
                "Este vuelo no suma intentos.", (x, qy + 60, room, 22), 11, MINT, align="center"
            )
        else:
            for i, (name, value) in enumerate(zip(ACTION_NAMES, s.last_q)):
                color = MINT if i == int(np.argmax(s.last_q)) else MUTED
                ui.text(name, (x, qy + 26 + i * 23, room - 80, 21), 11, color)
                ui.text(
                    f"{value:+.2f}",
                    (right.right - 86, qy + 26 + i * 23, 70, 21),
                    12,
                    color,
                    align="right",
                )
        summary = pygame.Rect(area.x, middle.bottom + 12, area.w, 58)
        for rect, best in zip(columns(summary, 0.5), (True, False)):
            ui.card(rect)
            label = "MEJOR RETORNO" if best else "MENOR RETORNO"
            ui.text(label, (rect.x + 16, rect.y + 6, rect.w - 32, 18), 10, MUTED, align="center")
            row = (max if best else min)(s.rows, key=lambda r: r["reward"]) if s.rows else None
            value = (
                f'Vuelo {row["attempt"]:03} · {row["reward"]:+.1f} · {row["outcome"]}'
                if row
                else "Disponible al finalizar un vuelo"
            )
            ui.text(
                value,
                (rect.x + 16, rect.y + 29, rect.w - 32, 22),
                12,
                MINT if best else RED,
                True,
                align="center",
            )
        actions = [
            (
                "Continuar" if self.paused else "Pausar",
                lambda: setattr(self, "paused", not self.paused),
                not s.completed or replay,
            ),
            ("Ver mejor vuelo", lambda: self.start_replay(True), s.completed),
            ("Ver peor vuelo", lambda: self.start_replay(False), s.completed),
            ("Otro modelo", self.to_selector, True),
        ]
        if replay:
            actions.append(("Volver a resultados", self.stop_replay, True))
        bw = (area.w - 12 * (len(actions) - 1)) // len(actions)
        for i, (label, action, enabled) in enumerate(actions):
            self.button(label, (area.x + i * (bw + 12), area.bottom - 44, bw, 44), action, enabled)
        ui.text(
            "ESPACIO  pausa     1  velocidad normal     2  rápida     ESC  volver al selector",
            (area.x, area.bottom + 12, area.w, 20),
            12,
            MUTED,
            align="center",
        )
        ui.text(
            self.error or "Resultados: " + str(s.folder),
            (area.x, area.bottom + 37, area.w, 20),
            11,
            RED if self.error else MUTED,
            align="center",
        )

    def draw(self):
        self.screen.fill(BG)
        self.buttons = []
        self.ui = UI(self.screen)
        area = content(self.screen)
        status = (
            "Selección de modelo"
            if self.page == "selector"
            else (
                "En pausa"
                if self.paused
                else ("Repetición" if self.page == "replay" else "Evaluación")
            )
        )
        self.ui.header(
            area,
            "Evaluar modelos",
            "Comprueba los aterrizajes, revisa los fallos y repite cada resultado.",
            status,
        )
        if self.page == "selector":
            self.draw_selector()
        else:
            self.draw_evaluation()
        self.ui_audit = self.ui.audit
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                self.events()
                if not self.running:
                    break
                try:
                    if (
                        not self.paused
                        and self.page == "evaluation"
                        and not self.session.completed
                        and time.monotonic() >= self.hold_until
                    ):
                        previous = len(self.session.rows)
                        for _ in range(self.speed):
                            self.session.step()
                            if len(self.session.rows) > previous:
                                self.hold_until = time.monotonic() + 0.8
                                break
                    elif not self.paused and self.page == "replay" and not self.replay_done:
                        action = self.replay_data["actions"][self.replay_position]
                        self.replay_state, reward, terminated, truncated, _ = self.replay_env.step(
                            action
                        )
                        self.replay_total += reward
                        self.replay_position += 1
                        self.replay_done = (
                            terminated
                            or truncated
                            or self.replay_position >= len(self.replay_data["actions"])
                        )
                except Exception as exc:
                    self.error = str(exc)
                    self.paused = True
                self.draw()
                self.clock.tick(50)
        finally:
            self.stop_replay()
            if self.session:
                self.session.close()
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.seed < 0:
        parser.error("Usa intentos positivos y semilla no negativa.")
    torch.set_num_threads(1)
    if args.headless:
        if not args.model:
            parser.error("--headless necesita --model")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        pygame.init()
        model, info = load_model(args.model)
        session = Evaluation(model, info, args.attempts, args.seed)
        try:
            while not session.completed:
                session.step()
        finally:
            session.close()
            pygame.quit()
        print("Resultados:", session.folder)
        print("Aciertos:", sum(r["success"] for r in session.rows), "/", len(session.rows))
    else:
        ui = EvaluatorUI()
        ui.attempts = args.attempts
        ui.seed = args.seed
        if args.model:
            ui.model, ui.info = load_model(args.model)
            ui.begin()
        ui.run()


if __name__ == "__main__":
    main()
