import pygame 

class Bird:
    def __init__(self, frames: list[pygame.Surface], x: int, y: int):
        self.frames = frames
        self.frame_idx = 0
        self.anim_speed = 0.15
        self.image = self.frames[0]
        self.x, self.y = float(x), float(y)
        self.vy = 0.0
        self.gravity = 0.35
        self.flap_impulse = -7.5

    @property
    def rect(self) -> pygame.Rect:
        r = self.image.get_rect()
        r.center = (int(self.x), int(self.y))
        return r

    def flap(self):
        self.vy = self.flap_impulse

    def update(self):
        self.vy += self.gravity
        self.y  += self.vy
        self.frame_idx = (self.frame_idx + self.anim_speed) % len(self.frames)
        self.image = self.frames[int(self.frame_idx)]

    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)

class Ground:
    def __init__(self, img: pygame.Surface, y: int, speed: int = 5):
        self.img = img
        self.y = y
        self.speed = speed
        self.x = 0
        self.w = img.get_width()

    def update(self):
        self.x = (self.x - self.speed) % self.w

    def draw(self, screen: pygame.Surface):
        screen.blit(self.img, (self.x - self.w, self.y))
        screen.blit(self.img, (self.x,self.y))


class PipePair:
    def __init__(self, img: pygame.Surface, x: int, gap_center: int, gap: int):
   
        self.img = img
        self.x = x
        self.gap_center = gap_center
        self.gap = gap
        self.w = img.get_width()
        self.h = img.get_height()

        self.top_img = pygame.transform.flip(img, False, True)
        self.bottom_img = img

        self.passed = False  

    def update(self, speed: int):
        self.x += speed

    def draw(self, screen: pygame.Surface):
        top_y = (self.gap_center - self.gap // 2) - self.top_img.get_height()
        bot_y = (self.gap_center + self.gap // 2)
        screen.blit(self.top_img, (self.x, top_y))
        screen.blit(self.bottom_img, (self.x, bot_y))

    def offscreen(self) -> bool:
        return self.x + self.w < 0

    @property
    def rects(self):
        top_y = self.gap_center - self.gap//2 - self.h
        bot_y = self.gap_center + self.gap//2
        rect_top = pygame.Rect(self.x, top_y, self.w, self.h)
        rect_bot = pygame.Rect(self.x, bot_y, self.w, self.h)
        return rect_top, rect_bot
