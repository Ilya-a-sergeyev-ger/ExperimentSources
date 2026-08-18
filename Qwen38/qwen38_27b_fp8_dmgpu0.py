"""FP8 only, same-session sweep + B-fit second point, one placement per file.

Formed from qwen38_27b_precision_placed.py and its bfit twin, with the
BF16 side dropped and both sample counts (n=5 or n=10 from the placed
script, n=2 for the B-fit) run in one session on one worker. The dmauto
twin lives in qwen38_27b_fp8_dmauto.py; placement is hardwired per file
so the operator does not toggle a module constant between runs.

Labels: 62 = FP8, n<num_samples>, p<prompt>, o<output>, dm<placement>.
"""

import asyncio
import uuid

from krauncher import KrauncherClient

client = KrauncherClient()

GROUP = f"qwen38-27b-fp8-placement-{uuid.uuid4().hex[:8]}"
DATA_URL = "hf://models/Qwen/Qwen3.8-27B-FP8/017b9c7"

DISK_GB = 40
WAIT_TIMEOUT = 3600

PLACEMENT = "dmgpu0"


@client.task(disk_gb=DISK_GB, group_id=GROUP, timeout=120)
def quick_probe():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(512, 512, device=dev)
    return {"probe_sum": float((x @ x).sum())}


@client.task(group_id=GROUP, data_urls=[DATA_URL], timeout=2400,
             dataset_size=0, disk_gb=DISK_GB, stream_stderr=True)
def warmup():
    """Pays the one-off download of the FP8 checkpoint. Not a measurement point."""
    import os
    n = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk("/data/Qwen__Qwen3.8-27B-FP8") for f in fs)
    return {"downloaded_gb": round(n / 2**30, 2)}


def _body(placement, model_dir, revision, expect_gb,
          num_samples, prompt_tokens, max_new_tokens):
    """One placement, one shape. Load, verify residency, generate."""
    from collections import Counter
    import os as _os
    # Reclaim allocator fragmentation so long prefills fit in 32 GB.
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if placement == "dmauto":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, revision=revision, torch_dtype="auto", device_map="auto")
    elif placement == "dmgpu0":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, revision=revision, torch_dtype="auto", device_map={"": 0})
    else:
        raise ValueError(f"unknown placement {placement!r}")
    model.eval()
    load_sec = time.monotonic() - t1
    free_after, _ = torch.cuda.mem_get_info()
    on_gpu_gb = (free_before - free_after) / 2**30

    def _dev(v):
        return f"cuda:{v}" if isinstance(v, int) else str(v)
    where = Counter(_dev(v) for v in getattr(model, "hf_device_map", {}).values())
    off_by_map = sum(n for k, n in where.items() if k in ("cpu", "disk"))
    print(f"placement={placement} map={dict(where)} on_gpu={on_gpu_gb:.2f} GB "
          f"expect={expect_gb} GB", flush=True)
    if placement != "dmauto" and (off_by_map or on_gpu_gb < 0.9 * expect_gb):
        raise RuntimeError(
            f"placement={placement} but only {on_gpu_gb:.2f} GB of an expected "
            f"{expect_gb} GB landed on the GPU (map={dict(where)})")

    common = {
        "placement": placement, "model_dir": model_dir,
        "load_sec": round(load_sec, 2),
        "device_map": dict(where),
        "gpu_total_gb": round(total / 2**30, 2),
        "weights_on_gpu_gb": round(on_gpu_gb, 2),
    }
    if not num_samples:
        return {**common, "num_samples": 0, "gen_sec": None}

    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 8 + 16)
    ids = tokenizer(filler, return_tensors="pt").input_ids[0][:prompt_tokens]
    input_ids = ids.unsqueeze(0).to(model.device)
    actual_p = int(input_ids.shape[1])
    t_gen = time.monotonic()
    for _ in range(num_samples):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    gen_sec = time.monotonic() - t_gen
    return {**common, "num_samples": num_samples, "prompt_tokens": actual_p,
            "max_new_tokens": max_new_tokens, "gen_sec": round(gen_sec, 2),
            "out_tokens_per_sec": round(num_samples * max_new_tokens / gen_sec, 2)}


def _task(fn):
    return client.task(group_id=GROUP, data_urls=[DATA_URL], timeout=3000,
                       dataset_size=0, disk_gb=DISK_GB, stream_stderr=True)(fn)


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
    print("FP8:  Qwen/Qwen3.8-27B-FP8 @ 017b9c7")
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
