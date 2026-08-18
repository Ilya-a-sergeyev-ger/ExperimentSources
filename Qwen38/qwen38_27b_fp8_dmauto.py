"""Twin of qwen38_27b_fp8_dmgpu0.py, placement pinned to dmauto."""

import asyncio

from qwen38_27b_fp8_dmgpu0 import (
    client, GROUP,
    fp8_load, fp8_shape, quick_probe, warmup, _run,
    SHAPES,
)

PLACEMENT = "dmauto"


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
