import random
import pygame as pg
from pygame.locals import *
from .assets import load_assets
from .constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GAP_MAX,
    GAP_MIN,
    MIN_PIPE_VISIBLE,
    FPS,META_PATH
)
from .sprites import Ground, Bird, PipePair
from src.ai.torch_model import load_policy
import json
import torch
import os
class GameMode:
    HUMAN = "HUMAN"
    AI = "AI"


class GameState:
    MENU = "MENU"
    PLAYING = "PLAYING"
    GAME_OVER = "GAME_OVER"


class FlappyEnv:
   

    def __init__(self, pipe_speed=-7, seed=42):
        random.seed(seed)

        # Screen
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pg.display.set_caption("Flappy Bird with AI")

        # Assets
        self.assets = load_assets()
        self.bg_img = self.assets["bg"]
        self.gnd_img = self.assets["ground"]
        self.pipe_img = self.assets["pipe"]
        self.bird_frames = [
            self.assets["bird1"],
            self.assets["bird2"],
            self.assets["bird3"],
        ]

        # Restart button
        self.restart_img = self.assets.get("restart")

        # Font
        self.font = self.assets.get("font") or pg.font.Font(None, 42)
        self.small_font = self.assets.get("small_font") or pg.font.Font(None, 28)

        # Sprites
        ground_y = SCREEN_HEIGHT - self.gnd_img.get_height()
        self.ground = Ground(self.gnd_img, y=ground_y, speed=7)
        self.bird = Bird(self.bird_frames, x=80, y=SCREEN_HEIGHT // 2)

        # Pipes
        self.pipes = []
        self.pipe_speed = pipe_speed
        self.allowed_frame_gaps = [ 60, 65, 70, 75]
        self.pixel_jitter = 0
        self._rebuild_spacing_table()
        self.dist_to_next_spawn = self._pick_gap_px()

        # flow
        self.running = False
        self.state = GameState.MENU
        self.mode = GameMode.HUMAN

        # Score
        self.score = 0
        #self.high_score = 0
        self.high_score_human = 0
        self.high_score_ai = 0
        self.ai_model = None
        self.ai_threshold = 0.5
        self.ai_sx, self.ai_sy, self.ai_sv = 1/300.0, 1/300.0, 1/10.0
        self.show_ai_hud = True
        self.ai_artifacts = META_PATH 
        self.scores_path = os.path.join(self.ai_artifacts, "scores.json")
        # load persisted scores if available
        try:
            if os.path.isfile(self.scores_path):
                with open(self.scores_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.high_score_human = int(data.get("human", 0))
                    self.high_score_ai = int(data.get("ai", 0))
        except Exception as e:
            print("[scores load warning]", e)
    
         
    def load_ai(self, meta_json_path: str):
        if not os.path.isfile(meta_json_path):
            raise FileNotFoundError(f"Path Couldn't Find: {meta_json_path}")

        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.ai_threshold = float(meta.get("threshold", 0.5))
        self.ai_sx = float(meta.get("sx", 1/300.0))
        self.ai_sy = float(meta.get("sy", 1/300.0))
        self.ai_sv = float(meta.get("sv", 1/10.0))
        self.ai_threshold = max(0.1, min(0.9, self.ai_threshold))

        model_path = meta.get("model_path") or os.path.join(self.ai_artifacts, "best_.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Invalid Model Path: {model_path}")

        self.ai_model = load_policy(model_path, map_location="cpu")
        self.ai_model.eval()



    def ai_bird(self) -> bool:
        """Return True to flap, False to wait."""
        if self.ai_model is None:
            return False 

        dx, dy, vy = self._observe()  
        x = torch.tensor(
            [self.ai_sx * dx, self.ai_sy * dy, self.ai_sv * vy],
            dtype=torch.float32
        ).unsqueeze(0) 

        with torch.no_grad():
            p = float(self.ai_model(x).view(-1)[0].item())

        return p > self.ai_threshold

   
    def _closest_pipe(self) -> PipePair | None:
        closest = None
        min_dx = 1e9
        bx = self.bird.x
        for p in self.pipes:
            dx = p.x + p.w - bx
            if dx >= 0 and dx < min_dx:
                min_dx = dx
                closest = p
        return closest
    
    def _observe(self):
        """Raw (dx, dy, vy)."""
        p = self._closest_pipe()
        if p is None:
            return 300.0, 0.0, self.bird.vy
        dx = (p.x + p.w) - self.bird.x
        dy = (p.gap_center - self.bird.y)
        return float(dx), float(dy), float(self.bird.vy)

    def _observe_norm(self):
        dx, dy, vy = self._observe()
        return dx / 300.0, dy / 300.0, vy / 10.0

    def _rebuild_spacing_table(self):
        spd = max(1, abs(int(self.pipe_speed)))
        self.allowed_px_gaps = [max(1, f * spd) for f in self.allowed_frame_gaps]

    def _pick_gap_px(self) -> int:
        base = random.choice(self.allowed_px_gaps)
        if self.pixel_jitter:
            j = random.randint(-self.pixel_jitter, self.pixel_jitter)
            base = max(40, base + j)
        return base

    def _safe_gap_and_center(self) -> tuple[int, int]:
        ground_y = self.ground.y  # SCREEN_HEIGHT - ground_img_height
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
        self.pipes.append(
            PipePair(self.pipe_img, x=x, gap_center=gap_center, gap=gap)
        )

   
    def handle_events(self):
        for e in pg.event.get():
            if e.type == QUIT:
                self.running = False
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    self.running = False
                # MENU
                if self.state == GameState.MENU:
                    if e.key in (K_1, K_KP1):
                        self.mode = GameMode.HUMAN
                        self.start_game()
                    elif e.key in (K_2, K_KP2):
                        try:
                            meta_path = os.path.join(self.ai_artifacts, "best_.json")
                            self.load_ai(meta_path)
                            self.mode = GameMode.AI
                            self.start_game()
                        except Exception as ex:
                            print("[AI Load Error]", ex)
                elif self.state == GameState.PLAYING:
                    if self.mode == GameMode.HUMAN:
                        if e.key in (K_SPACE, K_UP):
                            self.bird.flap()
                elif self.state == GameState.GAME_OVER:
                    if e.key in (K_r, K_RETURN):
                        self.start_game()
                    elif e.key == K_m:
                        self.state = GameState.MENU

    def reset_world(self):
        self.ground.x = 0
        self.bird.x, self.bird.y = 80.0, float(SCREEN_HEIGHT // 2)
        self.bird.vy = 0.0
        self.pipes = []
        self.score = 0
        self._spawn_pipe(x=SCREEN_WIDTH)
        self._rebuild_spacing_table()
        self.dist_to_next_spawn = self._pick_gap_px()

    def start_game(self):
        self.reset_world()
        self.state = GameState.PLAYING

    def game_over(self):
        if self.mode == GameMode.AI:
            self.high_score_ai = max(self.high_score_ai, self.score)
        else:
            self.high_score_human = max(self.high_score_human, self.score)
        self.state = GameState.GAME_OVER
        try:
            with open(self.scores_path, "w", encoding="utf-8") as f:
                json.dump({"human": self.high_score_human, "ai": self.high_score_ai}, f, indent=2)
        except Exception as e:
            print("[scores save warning]", e)

    def update_ai(self):
        if self.state == GameState.PLAYING and self.mode == GameMode.AI:
            if self.ai_bird():
                self.bird.flap()

    def update(self):
        if self.state != GameState.PLAYING:
            return

        self.ground.update()
        self.bird.update()

        if self.mode == GameMode.AI:
            self.update_ai()

        # Pipe spawn
        self.dist_to_next_spawn -= abs(self.pipe_speed)
        while self.dist_to_next_spawn <= 0:
            self._spawn_pipe()
            self.dist_to_next_spawn += self._pick_gap_px()

        # Pipe Moves
        for p in self.pipes:
            p.update(speed=self.pipe_speed)
        self.pipes = [p for p in self.pipes if not p.offscreen()]

        # Scoring
        for p in self.pipes:
            if not p.passed and (p.x + p.w) < self.bird.x:
                p.passed = True
                self.score += 1

        # Collisions
        for p in self.pipes:
            for r in p.rects:
                if self.bird.rect.colliderect(r):
                    self.game_over()
                    return

        # Bounds
        if self.bird.y < 0 or self.bird.y > SCREEN_HEIGHT:
            self.game_over()


    def _blit_center(self, surf: pg.Surface, y: int):
        x = (SCREEN_WIDTH - surf.get_width()) // 2
        self.screen.blit(surf, (x, y))

    def _draw_score(self):
        score_surf = self.font.render(str(self.score), True, (255, 255, 255))
        self._blit_center(score_surf, 40)

    def _draw_hud(self):
        if self.state == GameState.MENU:
            title = self.font.render("FLAPPY BIRD", True, (255, 255, 255))
            self._blit_center(title, 120)
            if self.small_font:
                line1 = self.small_font.render("1: Normal Game", True, (255, 255, 255))
                line2 = self.small_font.render("2: AI Mode", True, (255, 255, 255))
                line3 = self.small_font.render("ESC: Exit", True, (200, 200, 200))
                self._blit_center(line1, 200)
                self._blit_center(line2, 240)
                self._blit_center(line3, 280)
        elif self.state == GameState.PLAYING:
            self._draw_score()
            if self.mode == GameMode.AI and self.show_ai_hud and self.small_font:
                dx, dy, vy = self._observe()
                info = self.small_font.render(
                    f"AI dx={int(dx)} dy={int(dy)} vy={int(vy)} thr={self.ai_threshold:.2f}", True, (200, 255, 200)
                )
                self.screen.blit(info, (10, 10))
        elif self.state == GameState.GAME_OVER:
            if self.restart_img:
                y = (SCREEN_HEIGHT - self.restart_img.get_height()) // 2 - 40
                self._blit_center(self.restart_img, y)
            over = self.font.render("GAME OVER", True, (255, 100, 100))
            self._blit_center(over, 120)

            if self.mode == GameMode.AI:
                best = self.high_score_ai
                mode_lbl = "AI Best"
            else:
                best = self.high_score_human
                mode_lbl = "Best"

            score_txt = self.small_font.render(
                f"Score: {self.score}   {mode_lbl}: {best}", True, (255, 255, 255)
            )
            self._blit_center(score_txt, 180)

            hint1 = self.small_font.render("R/Enter: Restart", True, (200, 200, 200))
            hint2 = self.small_font.render("M: Menu", True, (200, 200, 200))
            self._blit_center(hint1, 220)
            self._blit_center(hint2, 252)

    def render(self):
        self.screen.blit(self.bg_img, (0, 0))
        if self.state in (GameState.PLAYING, GameState.GAME_OVER):
            for p in self.pipes:
                p.draw(self.screen)
            self.ground.draw(self.screen)
            self.bird.draw(self.screen)
        else:
            self.ground.draw(self.screen)
        self._draw_hud()
        pg.display.flip()

   
    def run(self):
        self.running = True
        self.state = GameState.MENU
        self.reset_world()
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.render()
        pg.quit()
