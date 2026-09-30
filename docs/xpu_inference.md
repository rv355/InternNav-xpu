# Optional Intel XPU eager inference

This route targets the InternVLA-N1-DualVLN model only. Native XPU component
execution has been tested; full-checkpoint compatibility has **not** been
established. It does not establish Habitat, Isaac, simulator, sensor, or robot
stack compatibility. There is no compilation, quantization, CPU offload, or
static-cache change.

## Prerequisites and installation

Use a supported Intel GPU, a Linux kernel/Intel GPU driver supporting that GPU,
the Intel Level Zero compute runtime, and permission to access `/dev/dri/render*`.
PCI visibility alone is insufficient. Follow the current
[PyTorch Intel GPU prerequisites](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
for your OS/device. Driver installation is an administrator task; PyTorch wheels
do not install the kernel driver. Do not install IPEX or CUDA-only attention
extensions for this route.

From the repository root, use the existing `venv`. The official stable XPU index
was queried on 2026-09-30 and reported torch 2.14.0 and torchvision 0.29.0:

```bash
venv/bin/python -m pip index versions torch --index-url https://download.pytorch.org/whl/xpu
venv/bin/python -m pip index versions torchvision --index-url https://download.pytorch.org/whl/xpu
venv/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/xpu
venv/bin/python -m pip install -r requirements/internvla_n1_xpu.txt
venv/bin/python -m pip check
venv/bin/python -c "import torch; assert torch.xpu.is_available(); print(torch.__version__); print(torch.xpu.get_device_properties(0)); print(torch.xpu.mem_get_info(0))"
```

Transformers remains 4.51.0, Diffusers 0.33.1, and Accelerate 1.4.0. The separate
inference requirements do not require `flash_attn`, xFormers, or diffusion-policy.
Run from the repository root so imports resolve without installing all simulator
extras. The original CUDA requirements and installation route are unchanged;
do not combine the CUDA extras with this minimal XPU environment. Some existing
agent imports emit warnings about absent Habitat/depth-camera-filtering; those
packages are not needed for this standalone DualVLN route.

## Auxiliary checkpoint

Normal pretrained depth/RGB initialization is retained, including strict
`load_state_dict` checks. There is no skip-loading flag and no random-encoder
fallback, even when restoring a full DualVLN checkpoint. Supply the auxiliary
checkpoint from its upstream source before constructing the model:

```bash
curl -fL --retry 2 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
sha256sum checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

The downloaded file used for validation has SHA256
`b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.
It is a local ignored checkpoint, not part of the source patch. Missing files,
missing weights, and mismatched shapes remain fatal. Partial checkpoints and
training continue to use the normal pretrained initializer.

## Configuration and launch

Both inference loaders accept `attn_implementation` (default `auto`):

| Device | `auto` | Other selections |
| --- | --- | --- |
| CUDA | `flash_attention_2`, unchanged | `sdpa`, `eager`, explicit FA2 |
| XPU | `sdpa` | `eager` for debugging; external FA2 is rejected |

An explicit unavailable XPU or invalid XPU index raises an error, never silently
falls back to CPU. Existing CUDA device defaults are unchanged. BF16 and the
custom model/processor are retained. Policy configuration can set:

```python
device = 'xpu:0'
attn_implementation = 'auto'
model_path = 'checkpoints/InternVLA-N1-DualVLN'
```

On an XPU with enough available memory:

```bash
venv/bin/python -m scripts.realworld.http_internvla_server \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --device xpu:0 --attn_implementation auto --plan_step_gap 4 --skip_warmup
```

The server listens on port 5801. `--skip_warmup` is opt-in; it bypasses the existing
demo warmup, whose call omits the required camera intrinsic argument and uses
floating-point RGB input. The original default warmup branch is unchanged.
The agent is reset before serving when the bypass is selected. This is a local
development server with shared episode state, not a concurrent multi-client or
production server. Its existing hard-coded instruction behavior is unchanged.

## Memory and runtime evidence

DualVLN weights require 16,767,077,782 bytes (15.62 GiB), before KV cache, visual
activations, diffusion buffers, allocator reservations, and runtime overhead.
History increases memory use. More than 15.62 GiB of **available** device memory
is required; no universal sufficient-memory threshold has been established.
Shared system memory and swap are not equivalent to available XPU memory.

Validation on 2026-09-30 used the existing Python 3.12 venv, Linux kernel
7.0.0-31-generic, Intel Arc B390 (PCI 8086:b082), Level Zero V2 driver
1.15.39122+14. PyTorch reported 14,939,189,248 bytes (13.91 GiB) total and
1,558,061,056 bytes free at the initial probe. Full-model loading was therefore
not attempted. CUDA was unavailable.

Exact principal versions: torch 2.14.0+xpu, torchvision 0.29.0+xpu,
transformers 4.51.0, diffusers 0.33.1, accelerate 1.4.0, numpy 1.26.4,
Pillow 11.3.0, imageio 2.37.0, pydantic 2.11.10, pytest 9.1.1.

```bash
venv/bin/python -m pip install pytest
INTERNNAV_FULL_XPU_TEST=1 venv/bin/python -m pytest tests/unit_test/test_xpu_inference.py -q -s
```

Result: **23 passed, 1 skipped**. The opt-in full-checkpoint test skipped on its
memory gate: weights 16,767,077,782 bytes, total 14,939,189,248 bytes, free
712,245,248 bytes during the combined test run. Its two-GiB headroom check is a
minimum safety gate, not a guarantee that a full run will fit.

Tested scenarios:

- Device/attention selection, unavailable XPU, invalid index, and FA2 rejection.
- DINO native CPU/XPU fallback even with simulated xFormers availability;
  non-None nested biases are rejected rather than discarded.
- Strict auxiliary restoration, including missing file/key and shape failures;
  the real pretrained RGB encoder loaded with 22,056,576 finite parameters.
- Actual processor image/instruction handling, custom reduced Qwen prefill,
  cached decoding, and latent extraction on XPU in both SDPA and eager modes.
  Integer tokens/grid metadata stay integer. CPU BF16 eager reference logits
  matched at `rtol=0.05, atol=0.01` with identical weights. Peak XPU allocated
  memory: 36,691,968 bytes per mode.
- Actual full-size DINO/memory/QFormer/diffusion components on XPU with two
  224x224 frames, two diffusion steps, two samples, and four predicted steps.
  Only DINO uses pretrained weights in this component test; other weights are
  deterministic random fixtures. CPU/XPU use the **same noise tensor**, not just
  seeds. BF16 trajectories matched at `rtol=0.05, atol=0.05`. Peak XPU allocated
  memory: 212,505,600 bytes. These peaks are not full-model memory estimates.
- Stubbed STOP/waypoint outputs, look-down history, episode reset, thread-local
  no-grad contexts, and one real asynchronous-worker publication iteration.
- Diffusers checkpoint-toggle API compatibility, CLI help, and `pip check`.

Not verified: complete pretrained DualVLN execution/quality, long-history memory,
full-model peak memory, CUDA FA2 runtime equivalence, real checkpoint STOP/action
agreement, sustained asynchronous scheduling/reset races, or robot/simulator
integration. The full-checkpoint test is ready for a larger XPU but is not a
substitute for recorded navigation inputs and a CUDA baseline. For that comparison,
copy identical diffusion noise across devices and compare intermediate logits,
latents, trajectories, and resulting actions with explicit BF16 tolerances.

## Change boundaries

Shared resolution is used only by the two inference loaders. DINO retains its
existing CUDA xFormers computation. Tensor transfers are confined to inference
methods; RGB normalization and scheduler/guidance math are unchanged. No model
layers or checkpoint keys were added. Gradients are not disabled globally.
Encoder exports are loaded lazily to avoid requiring unrelated LongCLIP just to
import DINO. A backward-compatible checkpoint-toggle signature adapter fixes the
custom diffusion model's constructor under pinned Diffusers 0.33.1; the existing
training checkpoint computation is unchanged. Full training was not tested.