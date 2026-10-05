# Optional Intel XPU eager inference

Eager inference for InternVLA-N1-DualVLN on Intel GPUs.

## Installation

Install the Intel GPU driver and runtime per the
[PyTorch Intel GPU prerequisites](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
Then, from the repository root, in a dedicated virtual environment:

```bash
git submodule update --init internnav/model/basemodel/LongCLIP
venv/bin/python -m pip install -r requirements/internvla_n1_xpu.txt
venv/bin/python -m pip check
venv/bin/python -c "import torch; assert torch.xpu.is_available(); print(torch.__version__); print(torch.xpu.get_device_properties(0)); print(torch.xpu.mem_get_info(0))"
```

Do not install XPU packages into a CUDA environment. Keep `diffusers==0.32.2`;
newer versions fail to load the checkpoint with a `size mismatch` error.

## Auxiliary checkpoint

The DualVLN asynchronous RGB path requires this checkpoint:

```bash
curl -fL --retry 2 -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
sha256sum checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

## Configuration

For policy-based inference, set `device` and, optionally, `attn_implementation` in the
model configuration consumed by `ModelCfg` in `internnav/configs/model/base_encoders.py`.
The real-world HTTP server (`scripts/realworld/http_internvla_server.py`) always uses `auto`.

Set `attn_implementation` (default `auto`):

| Device | `auto` | Other selections |
| --- | --- | --- |
| CUDA | `flash_attention_2` | `sdpa`, `eager` |
| XPU | `sdpa` | `eager`; `flash_attention_2` is not supported |

Policy configuration example:

```python
device = 'xpu:0'
attn_implementation = 'auto'
model_path = 'checkpoints/InternVLA-N1-DualVLN'
```

## Launch

```bash
venv/bin/python -m scripts.realworld.http_internvla_server \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --device xpu:0 --plan_step_gap 4 --skip_warmup
```

The server listens on port 5801.

## Memory

Model weights alone need 15.62 GiB of free GPU memory; inference requires more.

## Tests

```bash
venv/bin/python -m pip install pytest
INTERNNAV_FULL_XPU_TEST=1 venv/bin/python -m pytest tests/unit_test/test_xpu_inference.py -q -s
```