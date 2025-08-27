"""
Multi-agent Flappy arena (single window, shared pipes/ground).
- Same physics/pipe logic as single-bird env.
- Each bird tracks its own (x, y, vy, alive) and score.
- Observations are vs. the same closest upcoming pipe.
- Call `step(decisions)` with a list[bool] (flap or not) per living bird.
"""
from __future__ import annotations
import random
import pygame as pg
from typing import List, Tuple

from .assets import load_assets
from .constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    GAP_MAX, GAP_MIN, MIN_PIPE_VISIBLE, FPS,
)
from .sprites import Ground, Bird, PipePair


class PopulationArena:
    def __init__(self, n_birds: int = 32, pipe_speed: int = -7, seed: int = 42):
        random.seed(seed)
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pg.display.set_caption("Flappy GA — Population Viewer")

        # Assets
        self.assets = load_assets()
        self.bg_img = self.assets["bg"]
        self.gnd_img = self.assets["ground"]
        self.pipe_img = self.assets["pipe"]
        self.bird_frames = [self.assets["bird1"], self.assets["bird2"], self.assets["bird3"]]
        self.font = self.assets.get("font") or pg.font.Font(None, 32)
        self.small_font = self.assets.get("small_font") or pg.font.Font(None, 20)

        # Ground / Pipes 
        ground_y = SCREEN_HEIGHT - self.gnd_img.get_height()
        self.ground = Ground(self.gnd_img, y=ground_y, speed=7)
        self.pipes: List[PipePair] = []
        self.pipe_speed = pipe_speed

        # Pipe spacing
        self.allowed_frame_gaps = [60, 65, 70, 75]
        self.pixel_jitter = 0
        self._rebuild_spacing_table()
        self.dist_to_next_spawn = self._pick_gap_px()

        # Birds (population)
        self.n_birds = n_birds
        self.birds: List[Bird] = []
        self.alive: List[bool] = []
        self.score: List[int] = []

        # Game flow
        self.running = True
        self.generation = 1

        self.reset_population()

    # pipe helpers 
    def _rebuild_spacing_table(self):
        spd = max(1, abs(int(self.pipe_speed)))
        self.allowed_px_gaps = [max(1, f * spd) for f in self.allowed_frame_gaps]

    def _pick_gap_px(self) -> int:
        base = random.choice(self.allowed_px_gaps)
        if self.pixel_jitter:
            j = random.randint(-self.pixel_jitter, self.pixel_jitter)
            base = max(40, base + j)
        return base

    def _safe_gap_and_center(self) -> Tuple[int, int]:
        ground_y = self.ground.y
        max_gap_by_visible = max(40, ground_y - 2 * MIN_PIPE_VISIBLE)
        gap_hi = max(0, min(GAP_MAX, max_gap_by_visible))
        gap_lo = min(GAP_MIN, gap_hi)
        gap = random.randint(gap_lo, gap_hi)
        min_center = (gap // 2) + MIN_PIPE_VISIBLE
        max_center = (ground_y - MIN_PIPE_VISIBLE) - (gap // 2)
        if min_center > max_center:
            return int(gap_lo), int(ground_y // 2)
        gap_center = random.randint(int(min_center), int(max_center))
        return int(gap), int(gap_center)

    def _spawn_pipe(self, x=None):
        if x is None:
            x = SCREEN_WIDTH
        gap, gap_center = self._safe_gap_and_center()
        self.pipes.append(PipePair(self.pipe_img, x=x, gap_center=gap_center, gap=gap))

    # population control
    def reset_population(self):
        self.ground.x = 0
        self.pipes.clear()
        self._spawn_pipe(x=SCREEN_WIDTH)
        self._rebuild_spacing_table()
        self.dist_to_next_spawn = self._pick_gap_px()

        self.birds = [Bird(self.bird_frames, x=80, y=SCREEN_HEIGHT // 2) for _ in range(self.n_birds)]
        self.alive = [True] * self.n_birds
        self.score = [0] * self.n_birds

    def living_indices(self) -> List[int]:
        return [i for i, a in enumerate(self.alive) if a]

    def closest_pipe(self) -> PipePair | None:
        closest = None
        min_dx = 1e9
        bx = self.birds[0].x
        for p in self.pipes:
            dx = p.x + p.w - bx
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                closest = p
        return closest

    def observe(self, i: int) -> Tuple[float, float, float]:
        b = self.birds[i]
        p = self.closest_pipe()
        if p is None:
            return 300.0, 0.0, b.vy
        dx = (p.x + p.w) - b.x
        dy = (p.gap_center - b.y)
        return float(dx), float(dy), float(b.vy)

    def step(self, flap_mask: List[bool]):
        # control
        for i in self.living_indices():
            if flap_mask[i]:
                self.birds[i].flap()

        # physics
        self.ground.update()
        for b in self.birds:
            b.update()

        # pipes spawn/move
        self.dist_to_next_spawn -= abs(self.pipe_speed)
        while self.dist_to_next_spawn <= 0:
            self._spawn_pipe()
            self.dist_to_next_spawn += self._pick_gap_px()
        for p in self.pipes:
            p.update(speed=self.pipe_speed)
        self.pipes = [p for p in self.pipes if not p.offscreen()]

        # score & collisions per bird
        for idx in self.living_indices():
            b = self.birds[idx]
            # score
            for p in self.pipes:
                if not getattr(p, "_passed_idx_" + str(idx), False) and (p.x + p.w) < b.x:
                    setattr(p, "_passed_idx_" + str(idx), True)
                    self.score[idx] += 1
            # collision
            for p in self.pipes:
                for r in p.rects:
                    if b.rect.colliderect(r):
                        self.alive[idx] = False
            # bounds
            if b.y < 0 or b.y > SCREEN_HEIGHT:
                self.alive[idx] = False

    
    def render(self, header: str = ""):
        self.screen.blit(self.bg_img, (0, 0))
        for p in self.pipes:
            p.draw(self.screen)
        self.ground.draw(self.screen)

        for i, b in enumerate(self.birds):
            if not self.alive[i]:
                continue
            b.draw(self.screen)  

        hud = self.small_font.render(header, True, (255, 255, 255))
        self.screen.blit(hud, (8, 8))

        # population stats
        alive_cnt = sum(self.alive)
        hud2 = self.small_font.render(f"Gen {self.generation} | Alive {alive_cnt}/{self.n_birds}", True, (200, 255, 200))
        self.screen.blit(hud2, (8, 30))
        pg.display.flip()


    def tick(self, fps: int = FPS):
        self.clock.tick(fps)

    def any_alive(self) -> bool:
        return any(self.alive)

    def handle_quit(self) -> bool:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return True
            if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return True
        return False

    def close(self):
        pg.quit()
