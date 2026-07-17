# RoboTwin 2.0 Examples

RoboTwin 2.0 workflow for StarWAM: setup, training, and rollout against the official RoboTwin harness. See root [README.md](../../README.md) and [../libero/LIBERO.md](../libero/LIBERO.md).

SAPIEN needs an NVIDIA Vulkan stack; StarWAM inference needs Torch/CUDA. Pick one of two modes depending on your environment:

- **local** (`local_policy.py`, `policy_mode: local`): when the Vulkan stack and Torch/StarWAM are available in the **same env** — inference runs in-process, simplest.
- **client/server** (`client_policy.py` + `policy_server.py`, `policy_mode: client`): when the two stacks **can't coexist** (e.g. the inference box has no Vulkan, or the render box has no suitable Torch) — the server runs inference in the Torch env, the client renders in the SAPIEN env, communicating over a socket.

## 1. Layout

```text
examples/robotwin/
  deploy_policy.py          # RoboTwin entry point; dispatches by policy_mode (local|client)
  local_policy.py           # in-process adapter (SAPIEN + Torch/StarWAM in one env)
  client_policy.py          # socket client adapter (SAPIEN-only env)
  policy_server.py          # StarWAM inference server (Torch/StarWAM env)
  deploy_policy.yml         # config for local mode
  deploy_policy_client.yml  # config for client mode
  configs/recipes/          # training recipes
  scripts/                  # launch scripts (train / server / rollout / text cache)
```

RoboTwin imports a policy by symlinking this directory into its `policy/` tree:

```bash
# local (single env):
ln -s /ABS/PATH/starWAM/examples/robotwin  RoboTwin/policy/starwam_policy
# client (split env):
ln -s /ABS/PATH/starWAM/examples/robotwin  RoboTwin/policy/starwam_client
```

## 2. Environment

### 2.1 Training / server (Torch/StarWAM)

```bash
conda create -n starwam python=3.11 -y
conda activate starwam
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### 2.2 Client (RoboTwin / SAPIEN)

Follow the official RoboTwin 2.0 install (SAPIEN, curobo, mplib, assets). The
StarWAM client adapter itself needs only `numpy` + the standard library. On
headless servers SAPIEN needs a valid NVIDIA Vulkan ICD; if `vulkaninfo` does
not list the GPU, point SAPIEN at one before running:

```bash
export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
```

## 3. Recipe

```text
examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml
```

| Recipe | Model family | Backbone | Notes |
| --- | --- | --- | --- |
| `starwam_robotwin_mot_wan22_5b.yaml` | `mot_wam` | Wan2.2-TI2V-5B | Fast-WAM-aligned dual-arm MoT recipe. Requires a preprocessed ActionDiT init payload. |

Key RoboTwin-specific settings (already in the recipe):

- `framework.action_dim: 14`, `framework.proprio_dim: 14` — dual-arm qpos.
- `framework.chunk_size: 32`.
- `data.video_keys: [cam_high, cam_left_wrist, cam_right_wrist]`, `concat_multi_camera: robotwin` — head 256x320 on top, [left|right] 128x160 each on the bottom → 384x320.
- `data.num_frames: 33`, `action_freq_ratio: 4` → 32 action steps, 9 sampled video frames.
- z-score normalization for actions and states.
- `text_prompt_template: "A video recorded from a robot's point of view executing the following instruction: {task}"`.

### Results

MoT (`mot_wam`), Wan2.2-TI2V-5B, checkpoint-27500. 100 episodes per task-setting
over all 50 tasks; `instruction_type=unseen`, `num_inference_steps=4`, `replan_steps=24`.

| Setting | Success rate |
| --- | ---: |
| demo_clean | 89.28% |
| demo_randomized | 89.68% |
| Overall (micro) | 89.48% (8948/10000) |

Per-task breakdown is in the appendix ([Section 9](#9-per-task-results)).

## 4. Paths you must set

Release recipes use placeholder paths. Replace them in the YAML or pass via `--override`.

| Field | Required for | What to set |
| --- | --- | --- |
| `backbone.pretrained_model_id` | all | Local Wan2.2-TI2V-5B directory. |
| `framework.action_expert_init_from` | training | Output of Section 6.1 (`preprocess_action_dit_init`). |
| `training.output_dir` | training | Run output directory. |
| `data.dataset_dirs` | training | LeRobot-format RoboTwin 2.0 dataset dir(s). |
| `data.text_embedding_cache_dir` | training + eval | Text embedding cache dir. |
| `data.action_stats_path` | training + eval | z-score action stats JSON (created if missing). |
| `data.state_stats_path` | training + eval | State stats JSON (can share the action stats file). |

## 5. Data

RoboTwin 2.0 LeRobot v2.1 dataset (Fast-WAM preprocessed): `yuanty/robotwin2.0-fastwam`.
14-D dual-arm qpos state/action (left 6 joints + gripper, right 6 joints + gripper);
cameras `cam_high` / `cam_left_wrist` / `cam_right_wrist`.

Download the RoboTwin harness assets with the official script:

```bash
cd RoboTwin && bash script/_download_assets.sh
```

## 6. Preprocessing

### 6.1 ActionDiT initialization

```bash
python -m starwam.tools.preprocess_action_dit_init \
  --config examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml \
  --source-backbone wan22 \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output /path/to/preprocessed/starwam_action_dit_init_wan22.pt \
  --device cuda:0 --dtype bfloat16
```

Set the generated path in the recipe or pass `--override framework.action_expert_init_from=...`.

### 6.2 Text embedding cache (optional but recommended)

RoboTwin has ~921k unique frame-level instructions; precompute the T5 cache
across GPUs so training/eval don't re-encode on the fly:

```bash
BACKBONE=/path/to/Wan2.2-TI2V-5B \
OUTPUT_DIR=/path/to/output/starwam_robotwin_mot_wan22_5b/text_embedding_cache \
bash examples/robotwin/scripts/precompute_text_cache_8gpu.sh
```

## 7. Training

Run Section 6.1 first, then launch. Fast-WAM target: global batch 1024, lr 1e-4,
5 epochs (~29.7k steps). Set `gradient_accumulation_steps` so
`per_device(8) x grad_accum x num_gpus == 1024` (8 GPUs → 16; 16 GPUs → 8).

Single node, 8 GPUs:

```bash
cd /path/to/starWAM
export CONDA_ENV=starwam
export REPO_DIR=/path/to/starWAM
export TRAIN_OVERRIDES='data.dataset_dirs=["/path/to/robotwin2.0"] backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt training.output_dir=/path/to/output/starwam_robotwin_mot_wan22_5b data.text_embedding_cache_dir=/path/to/output/starwam_robotwin_mot_wan22_5b/text_embedding_cache data.action_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json'

bash examples/robotwin/scripts/launch_starwam_robotwin_mot_wan22_5b_8gpu.sh
```

Two nodes (16 GPUs): use `ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2_multinode.yaml`,
set `NUM_MACHINES=2 NUM_PROCESSES=16 MACHINE_RANK=<0|1>`, and add
`training.gradient_accumulation_steps=8` to `TRAIN_OVERRIDES`.

## 8. Rollout / Evaluation

RoboTwin's `script/eval_policy.py` runs 100 episodes per (task, config). Aligned
to Fast-WAM: all 50 tasks, both `demo_clean` and `demo_randomized`,
`instruction_type=unseen`, `replan_steps=24`, `num_inference_steps=4`.

Do NOT wrap a task-setting in a wall-clock `timeout`: RoboTwin already enforces a
per-episode step limit, so 100 episodes terminate on their own. A group-level
timeout truncates episodes and biases the success rate.

### 8.1 Split env (recommended): servers + client

Step 1 — start the inference servers in the Torch/StarWAM env (one per GPU):

```bash
REPO=/path/to/starWAM \
PY=/path/to/starwam-env/bin/python \
CKPT=/path/to/output/starwam_robotwin_mot_wan22_5b/checkpoint-27500/pytorch_model \
BACKBONE=/path/to/Wan2.2-TI2V-5B \
ACTION_STATS=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json \
TEXTCACHE=/path/to/output/starwam_robotwin_mot_wan22_5b/eval_text_cache \
ACTION_INIT=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
SERVER_BIND=0.0.0.0 SERVER_PORT_BASE=8765 NGPU=8 \
bash examples/robotwin/scripts/launch_starwam_robotwin_policy_server.sh
```

Wait until each `robotwin_server_logs/server_*.log` prints `model ready ... listening on`.

Step 2 — symlink the client adapter and run the rollout in the RoboTwin env:

```bash
ln -s /path/to/starWAM/examples/robotwin  RoboTwin/policy/starwam_client
cd RoboTwin

export VK_ICD_FILENAMES=/path/to/nvidia_icd.json   # if SAPIEN needs it
SERVER_HOST=<server ip> SERVER_PORT_BASE=8765 NSERVERS=8 \
CLIENT_GPUS="0" NWORKER=8 CKPT_TAG=starwam27500 \
bash /path/to/starWAM/examples/robotwin/scripts/launch_starwam_robotwin_mot_rollout.sh
```

- A **single-GPU client** can drive all 8 servers: keep `CLIENT_GPUS="0"` and
  set `NWORKER`/`NSERVERS` to 8 (worker i → port `SERVER_PORT_BASE + i%8`).
- A **co-located 8-GPU box** maps 1:1: `CLIENT_GPUS="0 1 2 3 4 5 6 7"`.

### 8.2 Single command (single env)

If SAPIEN and Torch/StarWAM share one env, run the harness directly with
`policy_mode local` (no server needed):

```bash
ln -s /path/to/starWAM/examples/robotwin  RoboTwin/policy/starwam_policy
cd RoboTwin
python script/eval_policy.py \
  --config policy/starwam_policy/deploy_policy.yml \
  --overrides policy_name starwam_policy task_name adjust_bottle \
    task_config demo_clean instruction_type unseen ckpt_setting starwam27500 seed 0 \
    policy_mode local \
    checkpoint /path/to/output/starwam_robotwin_mot_wan22_5b/checkpoint-27500/pytorch_model \
    overrides "backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B data.action_stats_path=/path/to/output/.../action_stats.json data.state_stats_path=/path/to/output/.../action_stats.json data.text_embedding_cache_dir=/path/to/output/.../text_embedding_cache"
```

### 8.3 Results

RoboTwin writes one result per (task, config) under
`RoboTwin/eval_result/<task>/<policy>/<config>/<ckpt_setting>/<time>/_result.txt`,
and the worker logs (`rollout_logs/worker_*.log`) print each `Success rate: X/Y`.
There is no built-in cross-task aggregation; sum successes across the 100
task-settings for the overall micro-average.

### 8.4 Released ModelScope checkpoint

The trained MoT checkpoint is released at
[`panshaohua/starwam`](https://www.modelscope.cn/models/panshaohua/starwam):

```text
starwam-robotwin/
  mot/starwam_wan225b_robotwin_mot.pt   # dual-arm MoT weights
  action_stats.json                     # z-score action/state stats
```

Download and point `CKPT`/`ACTION_STATS` at it (server side); you still need the
Wan2.2 backbone locally:

```bash
pip install modelscope
modelscope download --model panshaohua/starwam --local_dir /path/to/starwam_ckpts
# CKPT=/path/to/starwam_ckpts/starwam-robotwin/mot/starwam_wan225b_robotwin_mot.pt
# ACTION_STATS=/path/to/starwam_ckpts/starwam-robotwin/action_stats.json
```

The released `.pt` is a plain model `state_dict`, so pass the file path directly
as `--checkpoint` (local mode) or `CKPT=` (server).

## 9. Per-task results

MoT, Wan2.2-TI2V-5B, checkpoint-27500. 100 episodes per task-setting;
`instruction_type=unseen`, `num_inference_steps=4`, `replan_steps=24`.
Overall: **8948 / 10000 = 89.48%**.

| Task | Clean | Rand. |
|---|---:|---:|
| Adjust Bottle | 100% | 99% |
| Beat Block Hammer | 97% | 93% |
| Blocks Ranking Rgb | 100% | 100% |
| Blocks Ranking Size | 91% | 90% |
| Click Alarmclock | 100% | 100% |
| Click Bell | 100% | 100% |
| Dump Bin Bigbin | 93% | 97% |
| Grab Roller | 100% | 100% |
| Handover Block | 78% | 78% |
| Handover Mic | 99% | 100% |
| Lift Pot | 100% | 100% |
| Move Can Pot | 84% | 91% |
| Move Playingcard Away | 99% | 100% |
| Move Stapler Pad | 87% | 83% |
| Hanging Mug | 41% | 41% |
| Open Laptop | 93% | 96% |
| Open Microwave | 33% | 31% |
| Pick Diverse Bottles | 79% | 73% |
| Pick Dual Bottles | 85% | 83% |
| Place A2b Left | 99% | 97% |
| Place A2b Right | 98% | 98% |
| Place Bread Basket | 98% | 94% |
| Place Bread Skillet | 93% | 89% |
| Place Can Basket | 49% | 63% |
| Place Cans Plasticbox | 97% | 97% |
| Place Container Plate | 98% | 99% |
| Place Dual Shoes | 83% | 88% |
| Place Empty Cup | 100% | 100% |
| Place Fan | 93% | 89% |
| Place Burger Fries | 96% | 96% |
| Place Mouse Pad | 83% | 84% |
| Place Object Basket | 87% | 77% |
| Place Object Scale | 96% | 98% |
| Place Object Stand | 97% | 95% |
| Place Phone Stand | 89% | 96% |
| Move Pillbottle Pad | 99% | 100% |
| Place Shoe | 87% | 96% |
| Press Stapler | 94% | 93% |
| Put Bottles Dustbin | 81% | 86% |
| Put Object Cabinet | 94% | 92% |
| Rotate Qrcode | 89% | 86% |
| Scan Object | 92% | 92% |
| Shake Bottle | 100% | 100% |
| Shake Bottle Horizontally | 100% | 100% |
| Stack Blocks Three | 92% | 96% |
| Stack Blocks Two | 100% | 100% |
| Stack Bowls Three | 87% | 77% |
| Stack Bowls Two | 97% | 95% |
| Stamp Seal | 82% | 88% |
| Turn Switch | 55% | 68% |
| **Average (%)** | **89.28%** | **89.68%** |

## 10. Troubleshooting

- SAPIEN `failed to find a rendering device`: NVIDIA Vulkan stack missing; check `vulkaninfo` lists the GPU or set `VK_ICD_FILENAMES`.
- Client can't reach servers: verify `SERVER_HOST`/ports and that servers bound `--host 0.0.0.0`.
- Missing ActionDiT init: run Section 6.1, set `framework.action_expert_init_from`.
- curobo/mplib build errors on old glibc: use conda-forge or manylinux2014 wheels.
