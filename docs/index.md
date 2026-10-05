# Project guide

| I want to… | Read |
|---|---|
| Understand the project | [README](../README.md) |
| Explore the measurements | [Data study](data_study.md) |
| Understand the findings | [Results overview](results.md) |
| Inspect local learning and adaptation | [Adaptive forecasting](adaptive_learning.md) |
| Inspect the targeted residual follow-up | [Residual diagnosis and state-space study](residual_state_space.md) |
| Inspect validation, controllers and activity counts | [Registered operational learning](operational_learning.md) |
| Check whether the data could show a small gain | [Power analysis](power_analysis.md) |
| Read the draft multi-band protocol | [Multi-band protocol](multiband_protocol.md) |
| Read the research summary | [Three-page PDF](research_note/research_note.pdf) |
| Run the code | [Reproduction guide](reproduction.md) |
| Check methods and evidence | [Technical reference](technical/index.md) |

`src/` contains forecasting code, `scripts/` the entry points, `configs/` experiment settings,
`tests/` synthetic checks, and `results/` saved research outputs. Recordings and local runs are
excluded from Git.

## Protocols and historical locks

Historical dates and local locks do not prove independent registration. Some original rural-site
selection overlaps evaluation. These limitations and the pending third-site evaluation are
explained in the [technical reference](technical/index.md).
