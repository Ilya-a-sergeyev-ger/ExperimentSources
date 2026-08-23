"""FP8 only, vLLM runtime, same-session sweep + B-fit second point.

Analog of qwen38_27b_fp8_dmgpu0.py rewritten to exercise the vLLM path
(CUTLASS W8A8 on Ada, sm_89). Preserves the shape sweep, sample counts,
labelling and data-url conventions of the transformers-based script; only
the loader, generate loop and residency semantics differ.

Placement is inapplicable to vLLM (it manages its own memory pool, not
device_map). The task-id suffix is the runtime tag "vllm" so rows do not
collide with the transformers-based dmauto/dmgpu0 sweeps.

Labels: 62 = FP8, n<num_samples>, p<prompt>, o<output>, vllm.
"""

import asyncio
import os
import uuid

from krauncher import KrauncherClient

client = KrauncherClient()

GROUP = f"qwen38-27b-fp8-vllm-{uuid.uuid4().hex[:8]}"
DATA_URL = "hf://models/Qwen/Qwen3.8-27B-FP8/017b9c7"

DISK_GB = 40
# Explicit per-task pin: env-var pick-up applies only to the first task.
GPU_NAME = os.environ.get("KRAUNCHER_GPU_NAME", "")
WAIT_TIMEOUT = 3600

PLACEMENT = "vllm"


@client.task(disk_gb=DISK_GB, gpu_name=GPU_NAME,
             group_id=GROUP, timeout=120)
def quick_probe():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(512, 512, device=dev)
    return {"probe_sum": float((x @ x).sum())}


@client.task(group_id=GROUP, data_urls=[DATA_URL], timeout=2400,
             dataset_size=0, disk_gb=DISK_GB, gpu_name=GPU_NAME,
             stream_stderr=True)
def warmup():
    """Pays the one-off download of the FP8 checkpoint. Not a measurement point."""
    import os
    n = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk("/data/Qwen__Qwen3.8-27B-FP8") for f in fs)
    return {"downloaded_gb": round(n / 2**30, 2)}


def _body(placement, model_dir, revision, expect_gb,
          num_samples, prompt_tokens, max_new_tokens):
    """One shape, vLLM runtime. Load, report residency, generate."""
    import os as _os
    import time
    import torch
    _os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
        # Blackwell (sm_100/sm_120): vLLM 0.24 DeepGEMM warmup hits "Unknown
        # recipe" on this block-scaled FP8 checkpoint (#47130/#47169). Fall
        # back to CUTLASS/Triton; Ada never reaches DeepGEMM, so it's a no-op there.
        _os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    for _r, _, _fs in _os.walk(model_dir):
        for _f in _fs:
            try:
                _fd = _os.open(_os.path.join(_r, _f), _os.O_RDONLY)
                _os.posix_fadvise(_fd, 0, 0, _os.POSIX_FADV_DONTNEED)
                _os.close(_fd)
            except OSError:
                pass

    tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=revision)
    free_before, total = torch.cuda.mem_get_info()
    t1 = time.monotonic()
    # max_model_len explicit — otherwise vLLM plans a KV pool for the model's
    # full context length and OOMs on Ada 48 GB with a 27.5 GB FP8 checkpoint.
    llm_kwargs = dict(
        model=model_dir,
        revision=revision,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=(prompt_tokens or 0) + (max_new_tokens or 0) + 128,
        gdn_prefill_backend="triton",
    )
    # On Hopper (sm_90) the default max_num_seqs=1024 exceeds the Mamba cache
    # blocks CUDA-graph capture needs; cap it to a moderate value there.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        llm_kwargs["max_num_seqs"] = 512
    llm = LLM(**llm_kwargs)
    load_sec = time.monotonic() - t1
    free_after, _ = torch.cuda.mem_get_info()
    on_gpu_gb = (free_before - free_after) / 2**30

    # Residency: no hf_device_map in vLLM. on_gpu_gb here covers weights AND
    # the KV cache pool AND activation buffers, so it will read materially
    # higher than the transformers `weights_on_gpu_gb` (~27.6 GB). Printed for
    # observability; no assert.
    print(f"placement={placement} on_gpu={on_gpu_gb:.2f} GB "
          f"(weights ~{expect_gb} GB + KV pool + workspace)", flush=True)

    common = {
        "placement": placement, "model_dir": model_dir,
        "load_sec": round(load_sec, 2),
        "gpu_total_gb": round(total / 2**30, 2),
        "weights_on_gpu_gb": round(on_gpu_gb, 2),
    }
    if not num_samples:
        return {**common, "num_samples": 0, "gen_sec": None}

    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 8 + 16)
    ids = tokenizer(filler, return_tensors="pt").input_ids[0][:prompt_tokens]
    prompt_ids = ids.tolist()
    actual_p = len(prompt_ids)
    # ignore_eos=True — hold max_tokens fixed regardless of EOS emissions,
    # matching the transformers script's fixed-output invariant.
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    t_gen = time.monotonic()
    for _ in range(num_samples):
        llm.generate([{"prompt_token_ids": prompt_ids}], sampling_params=sp,
                     use_tqdm=False)
    gen_sec = time.monotonic() - t_gen
    return {**common, "num_samples": num_samples, "prompt_tokens": actual_p,
            "max_new_tokens": max_new_tokens, "gen_sec": round(gen_sec, 2),
            "out_tokens_per_sec": round(num_samples * max_new_tokens / gen_sec, 2)}


def _task(fn):
    return client.task(group_id=GROUP, data_urls=[DATA_URL], timeout=3000,
                       dataset_size=0, disk_gb=DISK_GB, gpu_name=GPU_NAME,
                       stream_stderr=True)(fn)


@_task
def fp8_load(placement: str):
    return _body(placement, "/data/Qwen__Qwen3.8-27B-FP8", "017b9c7", 27.6, 0, 0, 0)


@_task
def fp8_shape(placement: str, num_samples: int,
              prompt_tokens: int, max_new_tokens: int):
    return _body(placement, "/data/Qwen__Qwen3.8-27B-FP8", "017b9c7", 27.6,
                 num_samples, prompt_tokens, max_new_tokens)


SHAPES = [
    ( 2,   1024,   128),
    (10,   1024,   128),
    ( 2,    256,  1024),
    ( 5,    256,  1024),
    ( 2,   4096,   128),
    (10,   4096,   128),
    ( 2,   8192,   128),
    ( 5,   8192,   128),
    ( 2,  16384,   128),
    (10,  16384,   128),
]


def _print_point(label, r):
    exec_sec = r.execution_time_sec - r.download_sec - r.pip_install_sec
    print(f"  {label:26s}  worker={r.worker_id}  gpu={r.actual_gpu}  "
          f"download={r.download_sec:7.1f}s  compute={exec_sec:7.1f}s  "
          f"total={r.execution_time_sec:7.1f}s  actual_cu={r.actual_cu:.4f}",
          flush=True)
    print(f"      output={r.output}", flush=True)


async def _run(label, coro):
    h = await coro
    r = await h.wait(timeout=WAIT_TIMEOUT)
    _print_point(label, r)
    return r


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return
    print(f"Group: {GROUP}   placement={PLACEMENT}")
    print("FP8:  Qwen/Qwen3.8-27B-FP8 @ 017b9c7  (vLLM runtime)")
    print("=" * 100)
    await _run("probe", quick_probe())
    await _run("warmup", warmup())
    await _run(f"62_load_{PLACEMENT}", fp8_load(placement=PLACEMENT))
    for n, p, o in SHAPES:
        await _run(f"62_n{n}_p{p}_o{o}_{PLACEMENT}",
                   fp8_shape(placement=PLACEMENT, num_samples=n,
                             prompt_tokens=p, max_new_tokens=o))


if __name__ == "__main__":
    asyncio.run(main())
