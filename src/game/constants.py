import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "..", "..", "assets")

SCREEN_WIDTH  = 864
SCREEN_HEIGHT = 936
FPS = 60

GAP_MIN = 130
GAP_MAX = 180
MIN_PIPE_VISIBLE = 90   


BIRD_PATH1  = os.path.join(ASSET_DIR, "bird1.png");BIRD_PATH2  = os.path.join(ASSET_DIR, "bird2.png");BIRD_PATH3  = os.path.join(ASSET_DIR, "bird3.png")
GROUND_PATH = os.path.join(ASSET_DIR,"ground.png")
PIPE_PATH  = os.path.join(ASSET_DIR, "pipe.png")
BG_PATH    = os.path.join(ASSET_DIR, "bg.png")
RESTART_PATH = os.path.join(ASSET_DIR,"restart.png")


META_PATH = "artifacts"