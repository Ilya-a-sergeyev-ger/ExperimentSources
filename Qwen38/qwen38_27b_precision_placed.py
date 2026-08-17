"""FP8 against BF16 on one host, one session — the precision comparison.

Formed from qwen38_27b_fp8_placed.py and qwen38_27b_bf16_placed.py. Both
checkpoints share a group_id, so both precisions land on the same worker and
reuse one /data; provisioning is taken from the BF16 side (vram_gb=80, disk
sized for ~83 GB of weights) because that is the constraint that has to hold
for both. Comparing precisions across two machines is what made the earlier
attempt unreadable: host-to-host spread reached 1.6-1.9x, larger than the
effect being measured.

What the comparison uses: gen_sec, the generation loop alone. exec_sec cannot
answer this question -- it also contains the page-cache evict, the tokenizer,
the weight load (which is itself ~4x longer in FP8) and the prompt build, all
of them CPU- and IO-bound with the GPU idle.

Placement is passed per call, never read from module scope -- a task body ships
without the module around it. Residency is verified against the VRAM the
weights actually occupy, not against hf_device_map: with device_map={"": 0}
transformers leaves that map empty, so counting offloaded entries in it passes
vacuously.

Labels: 62 = FP8, 63 = BF16, then p<prompt tokens>, o<output tokens>,
dm<device_map value>, as in 18_nw2 / 18_nw2_b128.
"""

import asyncio
import uuid

from krauncher import KrauncherClient

client = KrauncherClient()

FP8_ID, FP8_REV, FP8_DIR, FP8_GB = "Qwen/Qwen3.8-27B-FP8", "017b9c7", "/data/Qwen__Qwen3.8-27B-FP8", 27.6
BF16_ID, BF16_REV, BF16_DIR, BF16_GB = "Qwen/Qwen3.8-27B", "1d4bf0f", "/data/Qwen__Qwen3.8-27B", 55.0

GROUP = f"qwen38-27b-precision-{uuid.uuid4().hex[:8]}"

VRAM_GB = 80          # BF16 is the binding constraint; FP8 fits trivially
DISK_GB = 120         # ~83 GB of weights across the two checkpoints, plus room

PLACEMENT = "dmgpu0"  # "dmauto" | "dmgpu0" | "dmnone"

DATA_URLS = [f"hf://models/{FP8_ID}/{FP8_REV}", f"hf://models/{BF16_ID}/{BF16_REV}"]

WAIT_TIMEOUT = 3600


@client.task(vram_gb=VRAM_GB, disk_gb=DISK_GB, group_id=GROUP, timeout=120)
def quick_probe():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(512, 512, device=dev)
    return {"probe_sum": float((x @ x).sum())}


@client.task(vram_gb=VRAM_GB, group_id=GROUP, data_urls=DATA_URLS,
             timeout=3000, dataset_size=0, disk_gb=DISK_GB, stream_stderr=True)
def warmup():
    """Pays the one-off download of both checkpoints. Not a measurement point."""
    import os
    seen = {}
    for d in ("/data/Qwen__Qwen3.8-27B-FP8", "/data/Qwen__Qwen3.8-27B"):
        n = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(d) for f in fs)
        seen[d] = round(n / 2**30, 2)
    return {"downloaded_gb": seen}


def _body(placement, model_dir, revision, expect_gb,
          num_samples, prompt_tokens, max_new_tokens):
    """One precision, one shape, one placement. Load, verify, generate."""
    from collections import Counter
    import os as _os
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
    elif placement == "dmnone":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, revision=revision, torch_dtype="auto").to("cuda")
    else:
        raise ValueError(f"unknown placement {placement!r}")
    model.eval()
    load_sec = time.monotonic() - t1
    free_after, _ = torch.cuda.mem_get_info()
    on_gpu_gb = (free_before - free_after) / 2**30

    # accelerate reports GPUs as bare ints (0), not "cuda:0"; and with an
    # explicit device_map the map is often empty, so it cannot carry the check.
    def _dev(v):
        return f"cuda:{v}" if isinstance(v, int) else str(v)
    where = Counter(_dev(v) for v in getattr(model, "hf_device_map", {}).values())
    off_by_map = sum(n for k, n in where.items() if k in ("cpu", "disk"))
    print(f"placement={placement} map={dict(where)} on_gpu={on_gpu_gb:.2f} GB "
          f"expect={expect_gb} GB", flush=True)
    # Residency is decided by the VRAM the weights actually took.
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

    # Filler is built to the tokens actually needed; the earlier samples
    # tokenised ~225k characters regardless of shape, CPU-bound, GPU idle.
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
    return client.task(vram_gb=VRAM_GB, group_id=GROUP, data_urls=DATA_URLS,
                       timeout=3000, dataset_size=0, disk_gb=DISK_GB,
                       stream_stderr=True)(fn)


@_task
def fp8_load(placement: str = "dmauto"):
    return _body(placement, "/data/Qwen__Qwen3.8-27B-FP8", "017b9c7", 27.6, 0, 0, 0)


@_task
def fp8_shape(placement: str = "dmauto", num_samples: int = 5,
              prompt_tokens: int = 256, max_new_tokens: int = 1024):
    return _body(placement, "/data/Qwen__Qwen3.8-27B-FP8", "017b9c7", 27.6,
                 num_samples, prompt_tokens, max_new_tokens)


@_task
def bf16_load(placement: str = "dmauto"):
    return _body(placement, "/data/Qwen__Qwen3.8-27B", "1d4bf0f", 55.0, 0, 0, 0)


@_task
def bf16_shape(placement: str = "dmauto", num_samples: int = 5,
               prompt_tokens: int = 256, max_new_tokens: int = 1024):
    return _body(placement, "/data/Qwen__Qwen3.8-27B", "1d4bf0f", 55.0,
                 num_samples, prompt_tokens, max_new_tokens)


SHAPES = [
    (5, 256, 1024),
    (5, 8192, 128),
    (10, 1024, 128),
    (10, 4096, 128),
    (10, 16384, 128),
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
    print(f"FP8:  {FP8_ID} @ {FP8_REV}")
    print(f"BF16: {BF16_ID} @ {BF16_REV}")
    print("=" * 100)
    await _run("probe", quick_probe())
    await _run("warmup", warmup())
    for base, load_fn, shape_fn in (("62", fp8_load, fp8_shape),
                                    ("63", bf16_load, bf16_shape)):
        await _run(f"{base}_load_{PLACEMENT}", load_fn(placement=PLACEMENT))
        for n, p, o in SHAPES:
            await _run(f"{base}_p{p}_o{o}_{PLACEMENT}",
                       shape_fn(placement=PLACEMENT, num_samples=n,
                                prompt_tokens=p, max_new_tokens=o))


if __name__ == "__main__":
    asyncio.run(main())
