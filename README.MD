# Flappy Bird AI (PyTorch + Genetic Algorithm)

This project makes the classic **Flappy Bird** game playable both by humans and by an AI agent.
The AI is powered by a **PyTorch-based neural network** trained with a **Genetic Algorithm (GA)**.
During training, many birds play simultaneously, the best performers are selected, and new generations are created.

---

## Features

* **Two game modes**

  * **Human Mode:** Play with keyboard controls.
  * **AI Mode:** A trained model plays automatically.
* **Genetic Algorithm training** with elitism, crossover, and Gaussian mutation.
* **Population visualization:** Watch all birds in a single shared window.
* **Model saving/loading:** Best individual is stored as `.pt` and `.json`.

---

## Installation

```bash
git clone https://github.com/YSercanT/flappy-bird-ai.git
cd flappy-bird-ai
pip install -r requirements.txt
```

### Requirements

* Python 3.9+
* PyTorch
* pygame

---

## Running the Game

Start the game:

```bash
python -m src.main
```

### Menu

* `1` → **Human Mode** (play yourself)

  * Controls: `SPACE` or `↑` → flap, `ESC` → exit
* `2` → **AI Mode** (trained model plays automatically)

---

## Training with Genetic Algorithm

Start training with:

```bash
python -m src.ai.ga
```

If you don’t provide any parameters, the following **default values** are used:

```json
{
  "pop": 32,
  "gens": 20,
  "elites": 10,
  "mut_p": 0.9,
  "mut_sigma": 0.08,
  "thr_sigma": 0.05,
  "norm_sigma": 0.02,
  "fps": 120,
  "scheme": "uniform",
  "blend_alpha": 0.5,
  "hidden1": 64,
  "hidden2": 32,
  "activation": "relu",
  "threshold": 0.5,
  "seed": 123,
  "artifacts_dir": "artifacts"
}
```

This means you can simply run:

```bash
python -m src.ai.ga
```

and the training will proceed without any additional setup.

At the end of training:

* Best model → `artifacts/best_.pt`
* Model metadata (threshold, input scales, etc.) → `artifacts/best_.json`
* Run configuration → `artifacts/run_info.json`

### Custom Parameters

You can override parameters via CLI. Example:

```bash
python -m src.ai.ga --pop 100 --gens 30 --elites 15 --scheme blend
```

All training options can be changed via CLI:

* `--pop` → population size  
* `--gens` → number of generations  
* `--elites` → number of elite individuals preserved  
* `--mut_p` → mutation probability  
* `--mut_sigma` → mutation strength  
* `--fps` → simulation framerate (default 120)  
* `--threshold` → initial decision threshold for AI (clamped during evolution)  
* `--scheme` → crossover type (`uniform` or `blend`)  
* `--artifacts` → output directory for saving models and run metadata  
* `--blend_alpha` → mixing factor for blend crossover (default 0.5)  
* `--hidden1`, `--hidden2` → hidden layer sizes  
* `--activation` → activation function (`relu` or `tanh`)  
* `--thr_sigma` → mutation sigma for threshold  
* `--norm_sigma` → mutation sigma for input normalization scales  

---

## 📂 Project Structure

```
src/
 ├── ai/
 │   ├── __init__.py
 │   ├── ga.py      # GA training loop
 │   ├── torch_model.py     # Policy network
 └── game/
     ├── __init__.py
     ├── flappy_env.py      # Human/AI game environment
     ├── multi_env.py       # Population environment for GA
     ├── sprites.py         # Bird, pipe, ground classes
     └── assets.py          # Asset loading
```

---
