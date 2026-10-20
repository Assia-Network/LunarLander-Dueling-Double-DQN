# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Assia-Network
"""Componentes gráficos compartidos: distribución, tipografía y dibujo con Pygame."""

import pygame
import numpy as np

BG = "#0a101b"
CARD = "#111d2d"
BORDER = "#26374c"
TEXT = "#e6edf7"
MUTED = "#91a3bb"
MINT = "#8ce8c5"
PURPLE = "#bda6ff"
RED = "#ff9da6"
FONTS = {}


def font(size=14, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = pygame.font.SysFont("DejaVu Sans", size, bold=bold)
    return FONTS[key]


class UI:
    def __init__(self, surface):
        self.s = surface
        self.audit = []

    def text(self, value, box, size=14, color=TEXT, bold=False, align="left"):
        box = pygame.Rect(box)
        value = str(value)
        f = font(size, bold)
        if f.size(value)[0] > box.w:
            while value and f.size(value + "…")[0] > box.w:
                value = value[:-1]
            value += "…"
        image = f.render(value, True, color)
        pos = image.get_rect(midleft=box.midleft)
        if align == "center":
            pos.center = box.center
        elif align == "right":
            pos.midright = box.midright
        self.audit.append((value, box.copy(), pos.copy()))
        previous = self.s.get_clip()
        self.s.set_clip(box.clip(previous))
        self.s.blit(image, pos)
        self.s.set_clip(previous)

    def card(self, box, color=CARD):
        pygame.draw.rect(self.s, color, box, border_radius=14)
        pygame.draw.rect(self.s, BORDER, box, width=1, border_radius=14)

    def rule(self, x, y, w):
        pygame.draw.line(self.s, BORDER, (x, y), (x + w, y))

    def progress(self, box, fraction, color=MINT):
        box = pygame.Rect(box)
        pygame.draw.rect(self.s, "#213249", box, border_radius=4)
        if fraction > 0:
            pygame.draw.rect(
                self.s,
                color,
                (box.x, box.y, max(1, int(box.w * min(1, fraction))), box.h),
                border_radius=4,
            )

    def header(self, rect, title, subtitle, status):
        self.text("ORBITAL", (rect.x, 18, 160, 28), 21, MINT, True)
        self.text(title, (rect.x + 178, 18, rect.w - 460, 28), 21, TEXT, True)
        self.text(status, (rect.right - 270, 18, 270, 28), 13, MINT, align="right")
        self.text(subtitle, (rect.x, 54, rect.w, 24), 13, MUTED)
        self.rule(rect.x, 88, rect.w)

    def flight(
        self,
        rect,
        frame,
        title,
        status,
        metrics,
        empty="La red sigue entrenando · progreso en los paneles",
    ):
        rect = pygame.Rect(rect)
        self.card(rect)
        self.text(title, (rect.x + 18, rect.y + 10, rect.w * 0.52, 26), 14, TEXT, True)
        self.text(
            status,
            (rect.x + rect.w * 0.54, rect.y + 10, rect.w * 0.46 - 18, 26),
            12,
            RED if "CHOQUE" in status else MINT,
            align="right",
        )
        area = pygame.Rect(rect.x + 12, rect.y + 48, rect.w - 24, rect.h - 106)
        pygame.draw.rect(self.s, "#000000", area, border_radius=8)
        if frame is None:
            self.text(
                "Vuelo sin renderizar",
                (area.x + 16, area.centery - 26, area.w - 32, 26),
                16,
                MINT,
                True,
                align="center",
            )
            self.text(
                empty, (area.x + 16, area.centery + 8, area.w - 32, 22), 11, MUTED, align="center"
            )
        else:
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scale = min(area.w / surface.get_width(), area.h / surface.get_height())
            scaled = pygame.transform.smoothscale(
                surface,
                (
                    max(1, int(surface.get_width() * scale)),
                    max(1, int(surface.get_height() * scale)),
                ),
            )
            self.s.blit(scaled, scaled.get_rect(center=area.center))
        width = (rect.w - 36) / len(metrics)
        for i, (label, value) in enumerate(metrics):
            x = rect.x + 18 + i * width
            self.text(label, (x, rect.bottom - 49, width, 17), 10, MUTED, align="center")
            self.text(value, (x, rect.bottom - 30, width, 22), 14, TEXT, True, align="center")

    def metric(self, rect, label, value, detail, color=MINT):
        rect = pygame.Rect(rect)
        self.card(rect)
        self.text(label, (rect.x + 16, rect.y + 10, rect.w - 32, 18), 11, MUTED, align="center")
        self.text(
            value, (rect.x + 16, rect.y + 31, rect.w - 32, 33), 25, color, True, align="center"
        )
        self.text(
            detail, (rect.x + 16, rect.bottom - 24, rect.w - 32, 18), 11, MUTED, align="center"
        )

    def graph(
        self, rect, title, series, color=MINT, subtitle="", dots=None, empty="Todavía no hay datos"
    ):
        rect = pygame.Rect(rect)
        self.card(rect)
        self.text(title, (rect.x + 18, rect.y + 10, rect.w - 36, 24), 15, TEXT, True)
        self.text(subtitle, (rect.x + 18, rect.y + 35, rect.w - 36, 18), 11, MUTED)
        if not series:
            self.text(
                empty, (rect.x + 20, rect.centery - 5, rect.w - 40, 24), 13, MUTED, align="center"
            )
            return
        values = [float(y) for _, y in series]
        lo, hi = min(values), max(values)
        padding = max((hi - lo) * 0.12, 0.001 if color == PURPLE else 1.0)
        lo -= padding
        hi += padding
        if color == PURPLE:
            lo = max(0.0, lo)
        plot = pygame.Rect(rect.x + 65, rect.y + 68, rect.w - 88, rect.h - 107)
        for value in np.linspace(lo, hi, 4):
            y = plot.bottom - (value - lo) / (hi - lo) * plot.h
            pygame.draw.line(self.s, BORDER, (plot.left, y), (plot.right, y))
            label = f"{value:.3g}"
            self.text(label, (rect.x + 8, y - 9, 48, 18), 11, MUTED, align="right")
        points = []
        means = []
        for i, (_, value) in enumerate(series):
            x = plot.left + i / max(1, len(values) - 1) * plot.w
            points.append((x, plot.bottom - (value - lo) / (hi - lo) * plot.h))
            avg = np.mean(values[max(0, i - 19) : i + 1])
            means.append((x, plot.bottom - (avg - lo) / (hi - lo) * plot.h))
        if len(points) > 1:
            pygame.draw.lines(self.s, "#567089", False, points, 1)
            pygame.draw.lines(self.s, color, False, means, 2)
        for i, point in enumerate(points):
            if dots is not None or len(points) == 1:
                c = (MINT if dots[i] else RED) if dots is not None else color
                pygame.draw.circle(self.s, c, tuple(map(round, point)), 3)
        self.text(series[0][0], (plot.left, plot.bottom + 10, 70, 18), 11, MUTED)
        if len(series) > 1:
            self.text(
                series[-1][0], (plot.right - 70, plot.bottom + 10, 70, 18), 11, MUTED, align="right"
            )


def content(surface):
    w, h = surface.get_size()
    width = min(1560, w - 48)
    return pygame.Rect((w - width) // 2, 104, width, h - 174)


def columns(rect, fraction=0.68, gap=18):
    rect = pygame.Rect(rect)
    left = pygame.Rect(rect.x, rect.y, int((rect.w - gap) * fraction), rect.h)
    right = pygame.Rect(left.right + gap, rect.y, rect.right - left.right - gap, rect.h)
    return left, right


class VistaGeneraciones:
    """Mosaico de vuelos y métricas; no contiene física ni entrenamiento."""

    def draw(self, screen, d):
        import math

        screen.fill(BG)
        ui = UI(screen)
        area = content(screen)
        phase = {
            "VUELOS": "Recolectando vuelos · pesos fijos",
            "APRENDIENDO": "Vuelos detenidos · entrenando la red",
            "VALIDANDO": "Validación independiente · sin exploración",
            "COMPLETADO": "Entrenamiento completado",
        }[d.phase]
        ui.header(
            area,
            "Flota lunar",
            phase,
            f"Generación {d.generation}/{d.max_generations}" + (" · PAUSA" if d.paused else ""),
        )
        summary = pygame.Rect(area.x, area.y, area.w, 46)
        ui.card(summary)
        labels = [
            f"Finalizados {d.finished}/{d.num_envs}",
            f"Aterrizajes {d.landings}/{d.num_envs}",
            f"Azar {d.random_fraction:.0%} · ε {d.epsilon:.0%}",
            f"Pasos {d.training_steps:,}",
        ]
        for i, label in enumerate(labels):
            ui.text(
                label,
                (area.x + i * area.w / 4, area.y + 9, area.w / 4, 28),
                13,
                MINT if i == 1 else TEXT,
                align="center",
            )
        graph_h = 205
        mosaic = pygame.Rect(area.x, summary.bottom + 12, area.w, area.h - graph_h - 82)
        count = len(d.tiles)
        cols = max(1, math.ceil(math.sqrt(count * mosaic.w / max(1, mosaic.h) / 1.55)))
        cols = min(cols, max(1, count))
        rows = max(1, math.ceil(count / cols))
        gap = 8
        cell_w = (mosaic.w - (cols - 1) * gap) // cols
        cell_h = (mosaic.h - (rows - 1) * gap) // rows
        for j, t in enumerate(d.tiles):
            row_count = min(cols, count - (j // cols) * cols)
            row_left = mosaic.x + (mosaic.w - (row_count * cell_w + (row_count - 1) * gap)) // 2
            box = pygame.Rect(
                row_left + (j % cols) * (cell_w + gap),
                mosaic.y + (j // cols) * (cell_h + gap),
                cell_w,
                cell_h,
            )
            ui.card(box)
            color = (
                MINT if t["status"] == "ATERRIZAJE" else MUTED if t["status"] == "EN VUELO" else RED
            )
            ui.text(
                f"#{t['index']:02} · {t['status']}",
                (box.x + 8, box.y + 3, box.w - 16, 20),
                11,
                color,
                True,
            )
            view = pygame.Rect(box.x + 5, box.y + 25, box.w - 10, max(1, box.h - 49))
            pygame.draw.rect(screen, "#000000", view)
            if t["frame"] is not None:
                surface = pygame.surfarray.make_surface(np.transpose(t["frame"], (1, 0, 2)))
                scale = min(view.w / surface.get_width(), view.h / surface.get_height())
                surface = pygame.transform.smoothscale(
                    surface,
                    (
                        max(1, int(surface.get_width() * scale)),
                        max(1, int(surface.get_height() * scale)),
                    ),
                )
                screen.blit(surface, surface.get_rect(center=view.center))
            else:
                ui.text(
                    "Sin render", (view.x, view.centery - 9, view.w, 18), 11, MUTED, align="center"
                )
            ui.text(
                f"Retorno {t['reward']:+.0f} · {t['steps']} pasos",
                (box.x + 8, box.bottom - 22, box.w - 16, 19),
                11,
                TEXT,
                align="center",
            )
        graphs = pygame.Rect(area.x, area.bottom - graph_h, area.w, graph_h)
        left, right = columns(graphs, 0.5)
        validation = d.last_validation
        result = (
            f"Validación {validation['landings']}/{validation['attempts']} · mejor retorno {d.best_score:+.1f}"
            if validation
            else "La primera validación elegirá best.pt"
        )
        ui.graph(left, "Recompensa media por generación", d.history[-150:], subtitle=result)
        progress = (
            f"Actualizando {d.block_updates-d.pending_updates}/{d.block_updates}"
            if d.pending_updates
            else f"{d.updates:,} actualizaciones"
        )
        ui.graph(right, "Pérdida de la red", d.loss_history[-300:], PURPLE, subtitle=progress)
        ui.text(
            f"ESPACIO pausa · V mostrar/ocultar · ← → página {d.page}/{d.pages} · 1/2/3 velocidad · S guardar · ESC salir",
            (area.x, area.bottom + 10, area.w, 22),
            12,
            MUTED,
            align="center",
        )
        ui.text(d.notice, (area.x, area.bottom + 35, area.w, 22), 11, MUTED, align="center")
        pygame.display.flip()
        return ui.audit
