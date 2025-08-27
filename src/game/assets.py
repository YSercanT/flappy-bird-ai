import pygame
from .constants import BIRD_PATH1,BIRD_PATH2,BIRD_PATH3, PIPE_PATH, BG_PATH,GROUND_PATH,RESTART_PATH

def load_assets() -> dict:
    assets = {}
    assets["bird1"] = pygame.image.load(BIRD_PATH1).convert_alpha()
    assets["bird2"] = pygame.image.load(BIRD_PATH2).convert_alpha()
    assets["bird3"] = pygame.image.load(BIRD_PATH3).convert_alpha()

    assets["pipe"] = pygame.image.load(PIPE_PATH).convert_alpha()
    assets["ground"] = pygame.image.load(GROUND_PATH).convert()
    assets["bg"]   = pygame.image.load(BG_PATH).convert()
    assets["restart"]   = pygame.image.load(RESTART_PATH).convert()

    return assets
