# Q-Catalyst Model Weight & Cache Policy

## 1. Overview and Boundaries

Pretrained protein language models (ESM-1v and ESM-2) provide zero-shot mutation-effect scoring and sequence embeddings. However, full-sized checkpoints are very large:
- **ESM-1v (650M parameters):** ~2.6 GB per checkpoint (5 members = ~13 GB total).
- **ESM-2 (650M parameters):** ~2.6 GB.
- **ESM-2 (150M parameters):** ~600 MB.
- **ESM-2 (35M parameters):** ~140 MB.
- **ESM-2 (8M parameters):** ~30 MB (lightweight smoke testing).

### Strict Home Session Limits
1. **Never download large checkpoints (>200 MB single, >500 MB session) during home sessions.**
2. **Do not download the five-member ESM-1v ensemble at home.**
3. **Use the built-in calibrated mock scorer (`config.mock=True`) for home development, unit tests, and CI.**

---

## 2. Workstation Caching Layout

When running on the designated workstation with high-speed network and approved storage, cache weights in:

```text
env/cache/models/
├── esm1v/
│   ├── config.json
│   ├── pytorch_model.bin  # or model.safetensors (~2.6 GB)
│   ├── tokenizer_config.json
│   └── vocab.txt
└── esm2/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
```

Do not commit files under `env/cache/` or model checkpoints into Git.

---

## 3. Loading Cached Weights in Code

To run inference with a locally cached checkpoint:

```python
from pathlib import Path
from qcatalyst.protein_ai.esm_scoring import ESMScorer, ESMScorerConfig

config = ESMScorerConfig(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    cache_dir=Path("env/cache/models/esm1v"),
    mock=False,
    device="cpu",  # or "cuda"
)

scorer = ESMScorer(config)
result = scorer.score_variant(wt_seq="...", mutant_str="N37P", parent_enzyme="IsPETase")
print(result.score)
```

If `cache_dir` is missing or invalid when `mock=False`, the system raises a descriptive `FileNotFoundError` without attempting any automatic background download.

---

## 4. Scientific Interpretation Notice

**All ESM-1v scores are zero-shot computational log-likelihood ratios representing relative statistical sequence compatibility.** They are computational prioritisation heuristics and **must never be described or treated as experimentally validated activity or thermostability improvements**.
