# Project guide

This project is a deep dive into one public radio dataset (AERPAW): can recent measurements,
and models trained on them, forecast how busy a Wi-Fi channel will be? Unfamiliar words are
explained in the [glossary](glossary.md).

## Reading paths

- **5 minutes:** the [README](../README.md).
- **15 minutes, no jargon:** the [story](story.md) and [glossary](glossary.md).
- **30 minutes:** [data study](data_study.md), then the [results overview](results.md), then the
  [power analysis](power_analysis.md).
- **Everything:** follow the chapter map below, then the [technical reference](technical/index.md).

## Chapter map

Each chapter answers one plain question. The status column uses the labels defined in the
[glossary](glossary.md#status-labels-used-in-this-repository).

| # | Plain question | Read | Status |
|---|---|---|---|
| 1 | What was recorded, and what does "busy" mean? | [Data study](data_study.md) | Descriptive |
| 2 | Does history beat learned models, and by how much? | [Results overview](results.md) | Exploratory |
| 3 | Can models adapt locally as new measurements arrive? | [Adaptive forecasting](adaptive_learning.md) | Exploratory |
| 4 | Are the leftover errors predictable? | [Residual diagnosis and state-space study](residual_state_space.md) | Exploratory |
| 5 | Does a more realistic validation help, and do activity counts help? | [Operational learning](operational_learning.md) | Declared in the registry |
| 6 | Could these recordings have detected a small gain? | [Power analysis](power_analysis.md) | Exploratory |
| 7 | When most measurements are missing, do learned models help? | [Sparse-regime study](sparse_regime.md) | Declared in the registry |
| 8 | Could pooling many frequency bands help? | [Multi-band protocol](multiband_protocol.md) | Draft, not run |
| 9 | What is the final independent test? | [CC2 protocol](technical/followup_v2_protocol.md) | Frozen and pending |

## Everything else

| I want to… | Read |
|---|---|
| Understand the project | [README](../README.md) |
| Look up a term | [Glossary](glossary.md) |
| See where every number comes from | [Claim-to-evidence table](claims.md) |
| Read the research summary | [Two-page PDF](research_note/research_note.pdf) |
| Run the code | [Reproduction guide](reproduction.md) |
| Check methods and evidence | [Technical reference](technical/index.md) |

`src/` contains forecasting code, `scripts/` the entry points, `configs/` experiment settings,
`tests/` synthetic checks, and `results/` saved research outputs. Recordings and local runs are
excluded from Git.

## Protocols and historical locks

Historical dates and local locks do not prove independent registration. Some original rural-site
selection overlaps evaluation. These limitations and the pending third-site evaluation are
explained in the [technical reference](technical/index.md).
