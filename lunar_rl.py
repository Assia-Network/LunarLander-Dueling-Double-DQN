# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Assia-Network
"""Lunar RL — PyTorch + Pygame + Gymnasium LunarLander-v3.
Ejecución: .venv/bin/python lunar_rl.py
Configuración de arquitectura, recolección de experiencias y entrenamiento por generaciones.
"""

from pathlib import Path
from datetime import datetime
import re
import ast
from collections import deque
from types import SimpleNamespace
import argparse, copy, csv, json, os, random, time, inspect, hashlib
import numpy as np
import torch
from torch import nn
import gymnasium as gym
import pygame
from ui_orbital import VistaGeneraciones

INPUT_NAMES = (
    "posición x",
    "posición y",
    "velocidad x",
    "velocidad y",
    "ángulo",
    "velocidad angular",
    "contacto pata izq.",
    "contacto pata der.",
)
ACTION_NAMES = ("Sin motor", "Motor izquierdo", "Motor principal", "Motor derecho")
TRAINING_NAME = "lunar_dueling_relu"  # Identificador del experimento.
HIDDEN_SIZES = (256, 256)  # Neuronas por capa oculta.
LEARNING_RATE = 1e-4
NUM_GENERATIONS = 50_000  # Todos vuelan; luego la red aprende una vez por generación.
NUM_ENVS = 20  # Cohetes activos simultáneamente con una red compartida.
DUELING = True
LAYER_NORM = False
VISIBLE_ENVS = 5  # Cohetes por página del mosaico; 0 oculta los vuelos.
EVAL_EVERY_GENERATIONS = 10  # Validación tras aprender; primera y última también.
EVAL_EPISODES = 20
EVAL_SEED = 100_000  # Mismos escenarios de validación; distintos del entrenamiento.
SEED = 0
DEVICE = torch.device("cpu")  # CPU para reducir transferencias en una red pequeña.
ROOT = Path(__file__).resolve().parent


class MiActivacion(nn.Module):
    """Activación ReLU utilizada por las capas ocultas de la red."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


# Captura la definición al iniciar; no ejecuta código extraído de modelos.
_ACTIVATION_SOURCE = inspect.getsource(MiActivacion)
_ACTIVATION_SHA256 = hashlib.sha256(_ACTIVATION_SOURCE.encode()).hexdigest()


def activation_metadata():
    return {
        "kind": "custom_class",
        "class_name": "MiActivacion",
        "source": _ACTIVATION_SOURCE,
        "sha256": _ACTIVATION_SHA256,
    }


def activation_compatible(spec):
    """Compara la implementación sin ejecutar el código almacenado en el checkpoint.

    Los cambios de comentarios, formato y docstrings no invalidan los pesos.
    El texto original debe corresponder a la huella registrada en los metadatos.
    """
    if not isinstance(spec, dict) or spec.get("kind") != "custom_class":
        return False
    source = spec.get("source")
    if not isinstance(source, str) or hashlib.sha256(source.encode()).hexdigest() != spec.get(
        "sha256"
    ):
        return False

    def implementation(source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body.pop(0)
        return ast.dump(tree, include_attributes=False)

    try:
        return implementation(source) == implementation(_ACTIVATION_SOURCE)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        output_dim: int,
        activation_cls: type[nn.Module],
        dueling: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        if not isinstance(hidden_sizes, tuple) or not hidden_sizes:
            raise ValueError("hidden_sizes debe ser una tupla, por ejemplo (128, 256).")
        if any(not isinstance(n, int) or n <= 0 for n in hidden_sizes):
            raise ValueError("La cantidad de neuronas debe ser positiva.")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.activation_cls = activation_cls
        self.dueling = dueling
        self.layer_norm = layer_norm
        self.norms = nn.ModuleList()

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        previous_width = input_dim
        for neurons in hidden_sizes:
            self.hidden_layers.append(nn.Linear(previous_width, neurons))
            self.norms.append(nn.LayerNorm(neurons) if layer_norm else nn.Identity())
            self.activations.append(activation_cls())  # una instancia nueva por capa
            previous_width = neurons
        if dueling:
            self.value_head = nn.Linear(previous_width, 1)
            self.advantage_head = nn.Linear(previous_width, output_dim)
        else:
            self.output_layer = nn.Linear(previous_width, output_dim)

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Se esperaban {self.input_dim} entradas; llegaron {x.shape[-1]}.")
        for layer, norm, activation in zip(self.hidden_layers, self.norms, self.activations):
            x = activation(norm(layer(x)))
        if self.dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.output_layer(x)  # Un valor Q por acción.


class DQNAgent:
    def __init__(
        self,
        network,
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=128,
        memory_size=50_000,
        warmup=1_000,
        train_every=4,
        target_every=63,
        epsilon_decay=60_000,
        seed=42,
        collect_episodes=5,
    ):
        self.online = network
        self.target = copy.deepcopy(network).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.warmup = warmup
        self.train_every = train_every
        self.target_every = target_every
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.steps = 0
        self.updates = 0
        self.collect_episodes = collect_episodes
        self.block_episodes = 0
        self.block_steps = 0
        self.pending_updates = 0
        self.block_updates = 0
        self.learned_blocks = 0
        self._learning_memory = None
        self.last_action_random = False
        assert memory_size >= batch_size > 0
        assert min(train_every, target_every, epsilon_decay, collect_episodes) > 0

    @property
    def epsilon(self):
        return max(0.1, 1 - 0.9 * self.steps / self.epsilon_decay)

    def action(self, state, explore=True):
        self.last_action_random = bool(explore and self.rng.random() < self.epsilon)
        if self.last_action_random:
            return self.rng.randrange(self.online.output_dim)
        was_training = self.online.training
        self.online.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = self.online(x).argmax(dim=1).item()
        self.online.train(was_training)
        return action

    def observe(self, state, action, reward, next_state, terminated):
        # Recolección solamente: los pesos se mantienen fijos durante todos los vuelos.
        if self.pending_updates:
            raise RuntimeError("Termina el aprendizaje del bloque antes de recoger otro paso.")
        self.memory.append(
            (
                np.array(state, copy=True),
                action,
                reward,
                np.array(next_state, copy=True),
                terminated,
            )
        )
        self.steps += 1
        self.block_steps += 1

    def end_episode(self):
        """Solo cuentan los vuelos de entrenamiento terminados, incluido TimeLimit."""
        self.block_episodes += 1
        if self.block_episodes >= self.collect_episodes:
            self.queue_learning()

    def queue_learning(self):
        """Programa el bloque; también permite vaciar el último bloque al llegar al límite."""
        if self.pending_updates:
            return
        self.block_episodes = 0
        if not self.block_steps or len(self.memory) < max(self.warmup, self.batch_size):
            return  # Mantener experiencias hasta superar el calentamiento.
        # Una actualización por cada cuatro pasos recogidos, agrupadas entre vuelos.
        self.block_updates = max(1, self.block_steps // self.train_every)
        self.pending_updates = self.block_updates
        self.block_steps = 0
        self._learning_memory = tuple(self.memory)

    def learn_one(self):
        """Un minibatch por llamada: la ventana sigue respondiendo durante el aprendizaje."""
        if not self.pending_updates:
            return None
        batch = self.rng.sample(self._learning_memory, self.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        to_tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
        states, next_states = to_tensor(states), to_tensor(next_states)
        actions = torch.as_tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards, terminals = to_tensor(rewards), to_tensor(terminals)
        self.online.train()
        predicted_q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN: online elige; target valora.
            self.online.eval()
            next_actions = self.online(next_states).argmax(1, keepdim=True)
            self.online.train()
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + self.gamma * (1 - terminals) * next_q
        loss = self.loss_fn(predicted_q, expected_q)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida no finita.")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.pending_updates -= 1
        if self.pending_updates == 0:
            self.learned_blocks += 1
            self._learning_memory = None
        return float(loss.detach().cpu())


def classify(env, state, terminated, truncated):
    if terminated and not env.unwrapped.lander.awake:
        return "ATERRIZAJE"
    if terminated and env.unwrapped.game_over:
        return "CHOQUE"
    if terminated and abs(float(state[0])) >= 1:
        return "FUERA DE ZONA"
    if truncated:
        return "TIEMPO AGOTADO"
    return "EN VUELO"


class FlightGroup:
    """Entornos independientes, inferencia en lote y pesos fijos durante cada generación.

    La física avanza por turnos en un proceso; no crea un proceso por cohete.
    Los vuelos terminados esperan al resto sin reiniciarse ni aportar pasos extra.
    """

    def __init__(self, count):
        self.envs = [
            gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=1000)
            for _ in range(count)
        ]
        self.count = count
        self.reset(0)

    def reset(self, generation):
        self.states = np.stack(
            [
                env.reset(seed=SEED + generation * self.count + i)[0]
                for i, env in enumerate(self.envs)
            ]
        )
        self.done = np.zeros(self.count, dtype=bool)
        self.rewards = np.zeros(self.count)
        self.steps = np.zeros(self.count, dtype=int)
        self.randoms = np.zeros(self.count, dtype=int)
        self.actions = np.zeros(self.count, dtype=int)
        self.status = ["EN VUELO"] * self.count
        self.frames = [None] * self.count

    def step(self, agent):
        active = np.flatnonzero(~self.done)
        if not len(active):
            return
        was_training = agent.online.training
        agent.online.eval()
        with torch.no_grad():
            batch = torch.as_tensor(self.states[active], dtype=torch.float32, device=DEVICE)
            actions = agent.online(batch).argmax(1).cpu().tolist()
        agent.online.train(was_training)
        for i, action in zip(active, actions):
            random_action = agent.rng.random() < agent.epsilon
            if random_action:
                action = agent.rng.randrange(4)
            state, reward, terminated, truncated, _ = self.envs[i].step(action)
            agent.observe(self.states[i], action, reward, state, terminated)
            self.states[i] = state
            self.actions[i] = action
            self.randoms[i] += random_action
            self.rewards[i] += float(reward)
            self.steps[i] += 1
            self.frames[i] = None
            if terminated or truncated:
                self.done[i] = True
                self.status[i] = classify(self.envs[i], state, terminated, truncated)

    def close(self):
        for env in self.envs:
            env.close()


class LunarApp:
    """Recoger generación → aprender → validar cuando corresponda → siguiente."""

    def __init__(self, smoke=False, load=None):
        torch.set_num_threads(1)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        pygame.init()
        self.screen = pygame.display.set_mode((1440, 960), pygame.RESIZABLE)
        pygame.display.set_caption("Lunar RL · Generaciones")
        self.vista = VistaGeneraciones()
        self.clock = pygame.time.Clock()
        self.red = QNetwork(8, HIDDEN_SIZES, 4, MiActivacion, DUELING, LAYER_NORM).to(DEVICE)
        self.agent = DQNAgent(
            self.red,
            learning_rate=LEARNING_RATE,
            seed=SEED,
            batch_size=16 if smoke else 128,
            warmup=32 if smoke else 1000,
            collect_episodes=NUM_ENVS,
        )
        self.flights = FlightGroup(NUM_ENVS)
        self.generation = 0  # Generaciones recogidas; no equivale a pasos ni episodios.
        self.training_episodes = 0
        self.phase = "VUELOS"
        self.validation = None
        self.best_score = -float("inf")
        self.last_validation = None
        self.validated_updates = -1
        self.next_validation_generation = EVAL_EVERY_GENERATIONS
        self.training_complete = False
        self.paused = False
        self.running = True
        self.speed = 1
        self.page = 0
        self.show_flights = VISIBLE_ENVS > 0
        self.smoke = smoke
        self.history = []
        self.loss_history = deque(maxlen=1000)
        self.last_loss = None
        self.notice = "Una red compartida · cada cohete tiene su propio escenario"
        self.training_name = TRAINING_NAME.strip() or "entrenamiento"
        started = datetime.now().astimezone()
        self.created_at = started.isoformat(timespec="microseconds")
        name = re.sub(r"[^\w-]+", "-", self.training_name).strip("-_")[:48] or "entrenamiento"
        self.run_dir = ROOT / "experiments" / f"{name}__{started:%Y-%m-%d_%H-%M-%S-%f}"
        self.run_dir.mkdir(parents=True)
        self.log = (self.run_dir / "episodes.csv").open("w", newline="")
        fields = [
            "episode",
            "generation",
            "env",
            "mode",
            "reward",
            "outcome",
            "steps",
            "epsilon",
            "random_actions",
            "random_fraction",
        ]
        self.writer = csv.DictWriter(self.log, fieldnames=fields)
        self.writer.writeheader()
        config = dict(
            training_name=self.training_name,
            created_at=self.created_at,
            environment="LunarLander-v3",
            hidden_sizes=HIDDEN_SIZES,
            activation="MiActivacion",
            activation_spec=activation_metadata(),
            seed=SEED,
            learning_rate=LEARNING_RATE,
            num_generations=NUM_GENERATIONS,
            num_envs=NUM_ENVS,
            visible_envs=VISIBLE_ENVS,
            eval_every_generations=EVAL_EVERY_GENERATIONS,
            eval_episodes=EVAL_EPISODES,
            eval_seed=EVAL_SEED,
            smoke=smoke,
            dueling=DUELING,
            layer_norm=LAYER_NORM,
            train_every=self.agent.train_every,
        )
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))
        if load:
            self.load_weights(Path(load))

    def complete_flights(self):
        self.generation += 1
        f = self.flights
        for i in range(NUM_ENVS):
            self.training_episodes += 1
            self.writer.writerow(
                dict(
                    episode=self.training_episodes,
                    generation=self.generation,
                    env=i + 1,
                    mode="ENTRENAR",
                    reward=float(f.rewards[i]),
                    outcome=f.status[i],
                    steps=int(f.steps[i]),
                    epsilon=self.agent.epsilon,
                    random_actions=int(f.randoms[i]),
                    random_fraction=float(f.randoms[i] / max(1, f.steps[i])),
                )
            )
        self.log.flush()
        self.history.append((self.generation, float(f.rewards.mean())))
        self.agent.queue_learning()  # Una actualización por cada cuatro transiciones.
        self.phase = "APRENDIENDO"
        self.notice = "Todos terminaron. Vuelos detenidos mientras se actualizan los pesos."

    def advance_generation(self):
        if self.generation >= NUM_GENERATIONS:
            self.training_complete = True
            self.phase = "COMPLETADO"
            self.paused = True
            self.save_weights()
            self.notice = (
                f"Finalizado · {self.training_episodes} vuelos · best.pt y last.pt guardados"
            )
        else:
            self.flights.reset(self.generation)
            self.phase = "VUELOS"

    def tick(self):
        if self.validation is not None:
            self.validation_slice()
        elif self.phase == "VUELOS":
            self.flights.step(self.agent)
            if self.flights.done.all():
                self.complete_flights()
        elif self.phase == "APRENDIENDO":
            start = time.perf_counter()
            while self.agent.pending_updates and time.perf_counter() - start < 0.012:
                self.last_loss = self.agent.learn_one()
                self.loss_history.append((self.agent.updates, self.last_loss))
            if not self.agent.pending_updates:
                self.save_weights()
                if (
                    self.last_validation is None
                    or self.generation >= self.next_validation_generation
                    or self.generation >= NUM_GENERATIONS
                ):
                    self.start_validation()
                if self.validation is not None:
                    self.phase = "VALIDANDO"
                else:
                    self.advance_generation()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (max(980, event.w), max(760, event.h)), pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE and not self.training_complete:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.save_weights()
                elif event.key == pygame.K_v:
                    self.show_flights = not self.show_flights
                elif event.key == pygame.K_1:
                    self.speed = 1
                elif event.key == pygame.K_2:
                    self.speed = 4
                elif event.key == pygame.K_3:
                    self.speed = 20
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    pages = (NUM_ENVS + max(1, VISIBLE_ENVS) - 1) // max(1, VISIBLE_ENVS)
                    self.page = (self.page + (1 if event.key == pygame.K_RIGHT else -1)) % pages

    def draw(self):
        f = self.flights
        page_size = max(1, VISIBLE_ENVS)
        indices = list(range(self.page * page_size, min(NUM_ENVS, (self.page + 1) * page_size)))
        tiles = []
        for i in indices:
            if self.show_flights and f.frames[i] is None:
                f.frames[i] = f.envs[i].render()
            tiles.append(
                dict(
                    index=i + 1,
                    frame=f.frames[i] if self.show_flights else None,
                    reward=f.rewards[i],
                    steps=f.steps[i],
                    status=f.status[i],
                )
            )
        datos = SimpleNamespace(
            tiles=tiles,
            num_envs=NUM_ENVS,
            generation=(
                min(self.generation + 1, NUM_GENERATIONS)
                if self.phase == "VUELOS"
                else self.generation
            ),
            max_generations=NUM_GENERATIONS,
            phase=self.phase,
            paused=self.paused,
            finished=int(f.done.sum()),
            landings=f.status.count("ATERRIZAJE"),
            epsilon=self.agent.epsilon,
            random_fraction=float(f.randoms.sum() / max(1, f.steps.sum())),
            training_steps=self.agent.steps,
            updates=self.agent.updates,
            hidden_sizes=HIDDEN_SIZES,
            pending_updates=self.agent.pending_updates,
            block_updates=self.agent.block_updates,
            best_score=self.best_score,
            last_validation=self.last_validation,
            history=self.history,
            loss_history=list(self.loss_history),
            notice=self.notice,
            page=self.page + 1,
            pages=(NUM_ENVS + page_size - 1) // page_size,
            speed=self.speed,
        )
        self.ui_audit = self.vista.draw(self.screen, datos)

    def run(self, limit=None, screenshot=None):
        frames = 0
        next_draw = 0.0
        try:
            while self.running:
                self.events()
                if not self.running:
                    break
                fast = self.phase != "VUELOS" or not self.show_flights or self.smoke
                if not self.paused:
                    start = time.perf_counter()
                    for _ in range(10000 if fast else self.speed * 2):
                        phase = self.phase
                        self.tick()
                        if (
                            self.paused
                            or self.phase != phase
                            or time.perf_counter() - start >= 0.012
                        ):
                            break
                now = time.monotonic()
                if now >= next_draw:
                    self.draw()
                    next_draw = now + (0.1 if fast else 0.04)
                frames += 1
                if limit and frames >= limit:
                    break
                self.clock.tick(25 if not fast or self.paused else 0)
            if screenshot:
                self.draw()
                pygame.image.save(self.screen, str(screenshot))
        finally:
            if self.validation is not None:
                self.validation.close()
            self.save_weights()
            self.log.close()
            self.flights.close()
            pygame.quit()

    def save_weights(self, kind="last"):
        if kind not in ("best", "last"):
            raise ValueError(kind)
        path = self.run_dir / f"{kind}.pt"
        temp = path.with_suffix(".tmp")
        torch.save(
            {
                "training_name": self.training_name,
                "created_at": self.created_at,
                "weights": self.red.state_dict(),
                "hidden_sizes": HIDDEN_SIZES,
                "activation": "MiActivacion",
                "activation_spec": activation_metadata(),
                "environment": "LunarLander-v3",
                "steps": self.agent.steps,
                "format_version": 4,
                "checkpoint_kind": kind,
                "selection_metric": "mean_reward",
                "validation": (
                    self.last_validation if self.validated_updates == self.agent.updates else None
                ),
                "training_episodes": self.training_episodes,
                "max_episodes": NUM_GENERATIONS * NUM_ENVS,
                "generations_completed": self.generation,
                "num_generations": NUM_GENERATIONS,
                "num_envs": NUM_ENVS,
                "updates": self.agent.updates,
                "collect_episodes": NUM_ENVS,
                "learned_blocks": self.agent.learned_blocks,
                "pending_updates": self.agent.pending_updates,
                "dueling": DUELING,
                "layer_norm": LAYER_NORM,
            },
            temp,
        )
        temp.replace(path)
        (self.run_dir / "activacion_usada.txt").write_text(_ACTIVATION_SOURCE, encoding="utf-8")
        self.notice = "Pesos guardados en " + str(path.relative_to(ROOT))
        return path

    def start_validation(self):
        """Validación independiente y por pasos: no modifica replay, epsilon ni entrenamiento."""
        if self.validation is None and self.validated_updates != self.agent.updates:
            self.validation = self.validate_weights()

    def validate_weights(self):
        model = copy.deepcopy(self.red).eval()
        env = gym.make("LunarLander-v3", max_episode_steps=1000)
        rewards, landings = [], 0
        try:
            for episode in range(EVAL_EPISODES):
                state, _ = env.reset(seed=EVAL_SEED + episode)
                total = 0.0
                while True:
                    with torch.no_grad():
                        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        action = model(x).argmax(1).item()
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    self.notice = f"Validando sin azar: vuelo {episode + 1}/{EVAL_EPISODES} · entrenamiento detenido"
                    yield
                    if terminated or truncated:
                        landings += classify(env, state, terminated, truncated) == "ATERRIZAJE"
                        rewards.append(float(total))
                        break
        finally:
            env.close()
        score = float(np.mean(rewards))
        self.validated_updates = self.agent.updates
        self.last_validation = dict(
            training_episodes=self.training_episodes,
            updates=self.agent.updates,
            mean_reward=score,
            std_reward=float(np.std(rewards)),
            landings=landings,
            attempts=EVAL_EPISODES,
            seed=EVAL_SEED,
            rewards=rewards,
        )
        with (self.run_dir / "validation.jsonl").open("a") as log:
            log.write(json.dumps(self.last_validation) + "\n")
        if score > self.best_score:
            self.best_score = score
            self.save_weights("best")
        self.save_weights("last")
        self.next_validation_generation = self.generation + EVAL_EVERY_GENERATIONS
        self.notice = f"Validación: {landings}/{EVAL_EPISODES} aterrizajes · retorno {score:+.1f} · mejor {self.best_score:+.1f}"

    def validation_slice(self):
        start = time.perf_counter()
        while time.perf_counter() - start < 0.012:
            try:
                next(self.validation)
            except StopIteration:
                self.validation = None
                self.advance_generation()
                break

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        if (
            checkpoint.get("environment") != "LunarLander-v3"
            or tuple(checkpoint.get("hidden_sizes", ())) != HIDDEN_SIZES
            or not activation_compatible(checkpoint.get("activation_spec"))
            or bool(checkpoint.get("dueling", False)) != DUELING
            or bool(checkpoint.get("layer_norm", False)) != LAYER_NORM
        ):
            raise ValueError("La arquitectura o activación del archivo no coincide con el código.")
        candidate = copy.deepcopy(self.red)
        candidate.load_state_dict(checkpoint["weights"])
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            raise ValueError("Pesos no finitos.")
        self.red.load_state_dict(candidate.state_dict())
        self.agent.target.load_state_dict(self.red.state_dict())
        self.agent.steps = int(checkpoint["steps"])
        # Cargar pesos inicia un experimento nuevo; conserva pasos para epsilon.
        self.training_episodes = 0
        self.notice = "Pesos cargados. Memoria y optimizador nuevos; no es reanudación exacta."
        if "training_episodes" not in checkpoint:
            self.notice += " Modelo antiguo: contador de episodios desde cero."


def main():
    global NUM_GENERATIONS, NUM_ENVS, VISIBLE_ENVS, TRAINING_NAME, HIDDEN_SIZES, LEARNING_RATE, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--envs", type=int, default=NUM_ENVS)
    parser.add_argument("--visible-envs", type=int, default=VISIBLE_ENVS)
    parser.add_argument("--name", default=TRAINING_NAME)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(HIDDEN_SIZES))
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if (
        min(args.generations, args.envs, *args.hidden) <= 0
        or args.visible_envs < 0
        or args.learning_rate <= 0
    ):
        parser.error(
            "Generaciones, entornos, capas y learning rate deben ser positivos; visibles puede ser cero."
        )
    if not args.name.strip():
        parser.error("Escribe un nombre de entrenamiento.")
    NUM_GENERATIONS = args.generations
    NUM_ENVS = args.envs
    VISIBLE_ENVS = args.visible_envs
    TRAINING_NAME = args.name.strip()
    HIDDEN_SIZES = tuple(args.hidden)
    LEARNING_RATE = args.learning_rate
    SEED = args.seed
    if args.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    app = LunarApp(smoke=args.smoke, load=args.load)
    app.run(limit=args.frames or (240 if args.smoke else None), screenshot=args.screenshot)
    print("Resultados:", app.run_dir)


if __name__ == "__main__":
    main()
