"""
Genetic Algorithm with single-window population view.

- Evaluates all agents together in one arena.
- Selects top-K elites; children via layer-wise crossover (uniform or blend).
- Applies Gaussian mutation to weights + lightweight behavior hparams (threshold, sx/sy/sv).
- Runs for N generations and tracks global best.

Example:
    python -m src.ai.ga_layered --pop 32 --gens 20 --elites 10 --fps 120 \
        --hidden1 64 --hidden2 32 --activation relu
"""
from __future__ import annotations
import argparse
import os
import random
from dataclasses import dataclass
from typing import List
import json
import torch
from .torch_model import build_policy, save_policy, Policy
from src.game.multi_env import PopulationArena
from src.game.constants import META_PATH
from datetime import datetime
@dataclass
class GAConf:
    pop_size: int = 32
    elites: int = 10
    generations: int = 20
    mutation_prob: float = 0.9
    mutation_sigma: float = 0.08
    blend_alpha: float = 0.5  
    fps: int = 120
    threshold: float = 0.5  # initial threshold seed
    seed: int = 123
    artifacts: str = "artifacts"


class Genome:
    def __init__(self, model: Policy, *, init_threshold: float = 0.5, init_norm=(1/300.0, 1/300.0, 1/10.0)):
        # model parameters (per-parameter tensors)
        self.params: List[torch.Tensor] = []
        with torch.no_grad():
            for p in model.parameters():
                self.params.append(p.detach().clone())
        self.fitness: float = float("-inf")
        # evolved behavior hparams
        self.threshold: float = float(init_threshold)  #  sigmoid cutoff
        self.sx: float = float(init_norm[0])          # input scaling for dx
        self.sy: float = float(init_norm[1])          # input scaling for dy
        self.sv: float = float(init_norm[2])          # input scaling for vy

    def clone(self) -> "Genome":
        g = Genome.__new__(Genome)
        g.params = [p.clone() for p in self.params]
        g.fitness = self.fitness
        g.threshold = self.threshold
        g.sx, g.sy, g.sv = self.sx, self.sy, self.sv
        return g

    def load_into(self, model: Policy):
        with torch.no_grad():
            for p, src in zip(model.parameters(), self.params):
                p.copy_(src)


def crossover_layerwise(a: Genome, b: Genome, scheme: str = "uniform",alpha: float =0.5) -> Genome:
    """Layer-wise crossover; optionally blend weights; average behavior hparams.
    scheme="uniform": randomly pick each layer from a or b (50/50)
    scheme="blend":   child_layer = alpha*a + (1-alpha)*b
    """
    child = a.clone()
    with torch.no_grad():
        # weights
        for i, (pa, pb) in enumerate(zip(a.params, b.params)):
            if scheme == "uniform":
                take_a = random.random() < 0.5
                child.params[i] = pa.clone() if take_a else pb.clone()
            elif scheme == "blend":
                child.params[i] = alpha * pa + (1.0 - alpha) * pb
            else:
                raise ValueError("Unknown crossover scheme")
        # average behavior hyperparams 
        child.threshold = 0.5 * (a.threshold + b.threshold)
        child.sx = 0.5 * (a.sx + b.sx)
        child.sy = 0.5 * (a.sy + b.sy)
        child.sv = 0.5 * (a.sv + b.sv)
    return child


def mutate_gaussian(g: Genome, prob: float, sigma: float,
                    *, thr_sigma: float = 0.05, norm_sigma: float = 0.02) -> Genome:
    child = g.clone()
    with torch.no_grad():
        # weights
        for i, layer in enumerate(child.params):
            mask = (torch.rand_like(layer) < prob).float()
            noise = torch.randn_like(layer) * sigma
            child.params[i] = layer + mask * noise
        #  threshold ∈ [0.1, 0.9]
        child.threshold = float(max(0.1, min(0.9, child.threshold + random.gauss(0.0, thr_sigma))))
        child.sx = float(max(1/600.0, min(1/150.0, child.sx + random.gauss(0.0, norm_sigma))))
        child.sy = float(max(1/600.0, min(1/150.0, child.sy + random.gauss(0.0, norm_sigma))))
        child.sv = float(max(1/20.0,  min(1/5.0,   child.sv + random.gauss(0.0, norm_sigma))))
    return child


def tournament(pop: List[Genome], k: int = 5) -> Genome:
    cand = random.sample(pop, k)
    cand.sort(key=lambda x: x.fitness, reverse=True)
    return cand[0]


def evaluate_population(model_tpl: Policy, genomes: List[Genome], conf: GAConf, current_gen: int) -> bool:
    """Runs one episode for the whole population. Returns True if user quits.[ESC]"""
    # different world each generation
    gen_seed = random.randint(0, 2**31 - 1)
    arena = PopulationArena(n_birds=len(genomes), pipe_speed=-7, seed=gen_seed)
    arena.generation = current_gen

    # build models for all genomes 
    models: List[Policy] = []
    for g in genomes:
        m = build_policy(**model_tpl.cfg.__dict__)
        g.load_into(m)
        m.eval()
        models.append(m)

    survived = [0 for _ in genomes]
    user_stopped = False

    while arena.any_alive():
        if arena.handle_quit():
            user_stopped = True
            break
        arena.tick(conf.fps)

        # Decisions for all living birds
        decisions = [False] * len(genomes)
        for i in arena.living_indices():
            dx, dy, vy = arena.observe(i)
            # per-genome input scaling + per-genome threshold
            x = torch.tensor([genomes[i].sx*dx, genomes[i].sy*dy, genomes[i].sv*vy], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                p = float(models[i](x)[0].item())
            if p > genomes[i].threshold:
                decisions[i] = True
            survived[i] += 1

        arena.step(decisions)
        arena.render(header="GA — viewing whole population")

    # Compute fitness 
    # fitness = pipes*10 + survival*0.01
    for i, g in enumerate(genomes):
        sc = arena.score[i]
        fit = sc * 10.0 + survived[i] * 0.01
        g.fitness = fit

    arena.close()
    return user_stopped

def pick_parents(pop: List[Genome], k: int = 5):
    a = tournament(pop, k)
    b = tournament(pop, k)
    if b is a:
        others = [x for x in pop if x is not a]
        if others:
            b = random.choice(others)
    return a, b

def main():
    parser = argparse.ArgumentParser(description="GA with population viewer")
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--mut_p", type=float, default=0.9)
    parser.add_argument("--mut_sigma", type=float, default=0.08)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scheme", choices=["uniform", "blend"], default="uniform")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--blend_alpha", type=float, default=0.5)
    # architecture
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")

    # behavior hyperparameter mutation
    parser.add_argument("--thr_sigma", type=float, default=0.05)
    parser.add_argument("--norm_sigma", type=float, default=0.02)

    args = parser.parse_args()

    os.makedirs(META_PATH, exist_ok=True)
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pop": args.pop,
        "gens": args.gens,
        "elites": args.elites,
        "mut_p": args.mut_p,
        "mut_sigma": args.mut_sigma,
        "thr_sigma": args.thr_sigma,
        "norm_sigma": args.norm_sigma,
        "fps": args.fps,
        "scheme": args.scheme,
        "blend_alpha": args.blend_alpha,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "activation": args.activation,
        "seed": 123,
        "artifacts_dir": META_PATH,
    }
    with open(os.path.join(META_PATH, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    random.seed(123)
    torch.manual_seed(123)


    template = build_policy(
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        activation=args.activation,
    )

    pop: List[Genome] = []
    for _ in range(args.pop):
        g = Genome(template, init_threshold=args.threshold)
        with torch.no_grad():
            for i in range(len(g.params)):
                g.params[i].normal_(0.0, 0.5)
        pop.append(g)

    best: Genome | None = None

    for gen in range(1, args.gens + 1):
        user_stopped = evaluate_population(
            template,
            pop,
            GAConf(
                pop_size=args.pop, elites=args.elites, generations=args.gens,
                mutation_prob=args.mut_p, mutation_sigma=args.mut_sigma,
                fps=args.fps, threshold=args.threshold
            ),
            current_gen=gen,
        )

        if user_stopped:
            print("[User requested stop] Exiting GA loop.")
            break

        pop.sort(key=lambda x: x.fitness, reverse=True)
        if best is None or pop[0].fitness > best.fitness:
            best = pop[0].clone()
            # Save best            
            m = build_policy(**template.cfg.__dict__)
            best.load_into(m)
            save_policy(m, os.path.join(META_PATH, "best_.pt"))            
            meta_path = os.path.join(META_PATH, "best_.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "threshold": best.threshold,
                    "sx": best.sx, "sy": best.sy, "sv": best.sv,
                    "model_path": os.path.join(META_PATH, "best_.pt"),
                }, f, indent=2)
        evaluated_mean = sum(g.fitness for g in pop) / len(pop)
        print(f"[Gen {gen:02d}] best={pop[0].fitness:.2f} mean={evaluated_mean:.2f} thr={pop[0].threshold:.2f}")

       

        # Selection
        elites = [pop[i].clone() for i in range(min(args.elites, len(pop)))]
        children: List[Genome] = []
        while len(elites) + len(children) < args.pop:
            #a, b = tournament(pop, 5), tournament(pop, 5)
            a, b = pick_parents(pop, 5)
            child = crossover_layerwise(a, b, scheme=args.scheme, alpha=args.blend_alpha)
            child = mutate_gaussian(child, prob=args.mut_p, sigma=args.mut_sigma,
                                     thr_sigma=args.thr_sigma, norm_sigma=args.norm_sigma)
            child.fitness = float("-inf") 
            children.append(child)
        pop = elites + children


    print("Saved:", os.path.join(META_PATH, "best_.pt"))


if __name__ == "__main__":
    main()
