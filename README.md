# Experiment sources

Measurement sources that serve as material for articles. Each file is a
self-contained scenario: tasks are declared with the `@client.task` decorator and
run on cloud GPUs through [krauncher.com](https://krauncher.com) — only the
orchestration stays local.

## Qwen38

Qwen3.8-27B measurements: FP8 against BF16, weight placement, and runtime
(transformers against vLLM).

| File | What it measures |
| --- | --- |
| `qwen38_27b_precision_placed.py` | FP8 vs BF16 in one session on one host (shared `group_id`, shared `/data`) — otherwise host-to-host spread swamps the effect being measured |
| `qwen38_27b_fp8_dmgpu0.py` | FP8 only, shape sweep plus a second B-fit point, placement `device_map={"": 0}` |
| `qwen38_27b_fp8_dmauto.py` | the same scenario with `device_map="auto"`; imports the task bodies from `dmgpu0` |
| `qwen38_27b_fp8_vllm.py` | the same sweep on vLLM (CUTLASS W8A8, sm_89); placement is inapplicable there and survives only as a label |

## Running

```bash
pip install krauncher
echo "CAS_API_KEY=..." > .env
python Qwen38/qwen38_27b_fp8_dmgpu0.py
```

Each scenario prints one line per point: worker, GPU type, download time, compute
time, `actual_cu`, and the result dict.

## Transformers patches

Running Qwen3.8 FP8 on Hopper (sm_90) needs two fixes on top of upstream
`transformers`. Both branches are applied in the krauncher worker image, so the
scenarios above use them transparently.

- [`fix-deepgemm-sm90-non-fp32-scales`](https://github.com/Ilya-a-sergeyev-ger/transformers/commits/fix-deepgemm-sm90-non-fp32-scales/)
  — universal FP32 upcast of block-scaled FP8 weight scales on SM90 (DeepGEMM
  path). Without it the model loads but the first forward errors on UE8M0
  scales packed as int32.
- [`fix-fp8-should-convert-prefix-match`](https://github.com/Ilya-a-sergeyev-ger/transformers/tree/fix-fp8-should-convert-prefix-match)
  — literal prefix matching in `should_convert_module`, replacing a fragile
  regex that skipped the wrong modules for this checkpoint.
