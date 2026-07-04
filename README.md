# StarWAM: A Generalizable Codebase for World-Action Models

StarWAM is a research codebase for building **World-Action Models (WAMs)**: robot policies that combine generative video/world models with action prediction modules. It is designed for modular experimentation with world-model backbones, action representations, and training recipes.

> This repository is an early research release. More WAM variants, benchmarks, model checkpoints, and technical details will be added.

## News

- **2026/07**: Initial StarWAM codebase prepared with Wan2.2 and Cosmos-Predict2 backbone adapters, LIBERO training/rollout recipes, MoT WAM, Shared-DiT WAM, and feature-conditioned WAM support.

## Highlights

- **World-model backbones**: reuse pretrained video generation models as robot world models, e.g., Wan2.2 and Cosmos-Predict2.
- **Multiple WAM families**:
  - `mot_wam`: multi-stream video/action experts with mixed attention, e.g., Motus / FastWAM-style world-action modeling.
  - `shared_dit_wam`: shared-DiT/register-token video-action prediction, e.g., DreamZero / LingBot-VA-style shared-token formulations.
  - `feature_conditioned_action_model`: action prediction conditioned on video/world-model features, e.g., Video-IDM, Mimic-Video / World2Action, and StarVLA-WM4A-style variants.
- **Benchmark recipes**: benchmark-specific data loading, normalization, text embedding caches, training recipes, and rollout utilities.
- **Typed recipe system**: YAML recipes are loaded into Python dataclasses without requiring Hydra.

## Repository Layout

```text
starwam/                     # Core Python package
  backbone/                  # Wan2.2 / Cosmos-Predict2 backbone adapters
  wam/                       # WAM wrappers and taxonomy-level model families
  modules/                   # DiT blocks, MoT, scheduler, shared-DiT modules
  action_model/              # Action expert builders
  data/                      # LeRobot dataset and text-cache utilities
  training/                  # Trainer, losses, flow utilities, entrypoint
  tools/                     # Preprocessing utilities
  utils/                     # Checkpoint and config helpers
configs/                     # Accelerate / DeepSpeed configs
examples/                    # Benchmark-specific recipes, rollout scripts, and launch scripts
pyproject.toml               # Python package metadata
```

## Model Families

StarWAM organizes WAM methods by taxonomy-level model families. The taxonomy is separated from the video/world-model backbone, so the same WAM family can be instantiated with different backbones.

### `mot_wam`

`mot_wam` uses separate video and action experts and mixes their Q/K/V streams through MoT-style attention. It supports first-frame and full-video action conditioning. This is the first functional LIBERO path in this codebase.

### `shared_dit_wam`

`shared_dit_wam` uses a shared DiT token space for clean video, noisy video, action tokens, and state tokens. Wan Shared-DiT is currently supported; additional backbones can implement the same `build_shared_dit_core(...)` interface.

### `feature_conditioned_action_model`

This family covers action models conditioned on video/world-model features. A single Wan DiT forward extracts observation tokens that condition an ActionDiT flow-matching expert. It is the intended home for Video-IDM, Mimic-Video/World2Action, and StarVLA-WM4A-style variants where a video generation model provides hidden states or generated-video features to an action decoder.

## Examples

Benchmark-specific setup, training, and evaluation instructions are maintained under `examples/`.

- [LIBERO examples](examples/libero/LIBERO.md)

## Roadmap

- [x] Wan2.2 backbone adapter.
- [x] Cosmos-Predict2 backbone adapter.
- [x] MoT WAM training and action rollout path.
- [x] Shared-DiT WAM path.
- [x] Feature-conditioned Video-IDM / WM4A action model path.
- [ ] Additional benchmark integrations.
- [ ] Technical report and model zoo.

## Citation

If you find StarWAM useful in your research, please consider citing it. A formal BibTeX entry will be added once the technical report is released.

## Acknowledgements

This project draws inspiration and references from several notable open-source initiatives, including:

- [StarVLA](https://github.com/starVLA/starVLA) — a primary reference for this project; its VLA/WM4A designs and training recipes directly informed StarWAM's action modeling.
- [DreamZero](https://github.com/dreamzero0/dreamzero)
- [LingBot-VA](https://github.com/robbyant/lingbot-va)
- [FastWAM](https://github.com/yuantianyuan01/FastWAM)
- [Mimic-Video](https://github.com/mimic-video/mimic-video)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Wan](https://github.com/Wan-Video/Wan2.2)
- [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
