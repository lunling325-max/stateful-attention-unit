# The Stateful Attention Unit (SAU)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20780300.svg)](https://doi.org/10.5281/zenodo.20780300)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A persistent-state AI architecture prototype. SAU maintains a state vector that survives across forward passes, evolves through gated mechanisms, consolidates memories via a slow channel, and exhibits self-driven idle dynamics.

**Author**: Cao Jing

**Institution**: Nanhang Jincheng College

fan_38324cj@qq.com

---

## Architecture

```
Input → [Perception Layer] → [Working Memory] → [Core State]
              ↑β=0.95              ↑α=β=0.5          ↑α=0.92
           (fast, external)     (balanced)       (slow, self-driven)
                                       ↓
                                  Slow Channel
                              (surprise accumulation)
                                       ↓
                                  Core Injection
```

### Key Properties

| Property | LSTM | Transformer | **SAU** |
|----------|------|-------------|---------|
| State persistence | Within sequence | None | **Across all calls** |
| State update | Multiplicative decay | N/A | **Additive** (S + η·δ) |
| Time scales | Single | Single | **Three layers** |
| Memory consolidation | None | None | **Slow channel** |
| Idle dynamics | Freezes | No mechanism | **Self-driven evolution** |

---

## Experimental Observations (268K parameters, synthetic data)

| Benchmark | Result |
|-----------|--------|
| Familiarity detection | Cohen's d = 3.68 (zero training) |
| Time-scale separation | Perception 0.37 / WM 0.60 / Core 0.994 |
| Slow channel consolidation | Triggers at ~54 steps (threshold 0.001) |
| Idle state drift | Core 0.94 / WM 0.43 / Perception 0.44 |

These are small-scale prototype observations. Whether properties persist at larger scales or on natural language remains an open question.

---

## Known Limitations

- 268K parameters, synthetic data only. Scaling experiments not yet conducted.
- Non-autoregressive text decoder — cannot generate coherent text.
- Batch training conflict from persistent state (workaround: batch_size=1 + gradient accumulation).
- SAU has not been validated on standard memory benchmarks (Copy Memory, Associative Recall).
- Missing comparisons with Mamba, RWKV, and Retentive Networks (2023).
- Homeostatic parameters (target 2.0, gain 1.0) are empirically chosen, not formally derived.
- Multi-unit self-organization experiments (not in paper) indicate SAU's continuous dynamics are poorly suited to Hebbian/STDP plasticity.

---

## Quick Start

```bash
# Requirements
pip install torch

# Run familiarity detection (no training needed)
cd benchmarks
python familiarity_bench.py

# Run time-scale separation test
python three_layer_multi.py

# Run idle state drift test
python idle_drift.py
```

---

## Paper

**"The Stateful Attention Unit"** — Cao Jing, 2026.

[Full Paper (Zenodo)](https://doi.org/10.5281/zenodo.20780300)

---

## License

MIT License — see [LICENSE](LICENSE) file.
