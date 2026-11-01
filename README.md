# Short-Term Network Latency Prediction

**Research project:** *Can Machine Learning Beat TCP's Estimator? Short-Term Network Latency Prediction from an End Host*

This repository contains the LaTeX research report, supporting figures and bibliography, plus Python reference scripts related to TCP connection-latency measurement and short-term RTT prediction.

> [!IMPORTANT]
> **Reproducibility and provenance:** The original measurement CSV files and original experimental scripts were not present in the project archive supplied for this repository. The Python files currently included are reconstructed reference implementations based on the methodology described in the report. They have not been verified to reproduce the exact numerical results, tables, or figures in the paper. Treat outputs from these scripts as new experimental runs, not as a reproduction of the reported results.

## Project overview

This study asks whether machine-learning models can predict a server's near-future round-trip time (RTT) more accurately than simple history-based estimators, including persistence, moving average, and exponentially weighted moving average (EWMA).

The report describes measurements to 12 public HTTPS hosts at five-second intervals on a home Wi-Fi connection and a mobile hotspot. It compares baseline estimators with regression models and discusses chronological (forward-chaining) evaluation to avoid training on future observations.

## Methods covered

- **Classical baselines:** persistence, five-sample moving average, and EWMA.
- **Machine-learning models:** Ridge Regression, k-nearest neighbours (KNN), Decision Tree, Random Forest, and histogram-based gradient boosting.
- **Feature engineering:** recent RTT lags, rolling statistics, EWMA, server-level history, and network context.
- **Evaluation metrics:** mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE), coefficient of determination (R²), and 95th-percentile absolute error.
- **Time-aware evaluation:** chronological test blocks with a gap between training and test periods.

The paper also describes statistical comparisons and additional experiments. Not every analysis described in the paper is implemented in the reconstructed reference script.

## Repository contents

```text
.
├── analyze_latency.py       # Reconstructed reference analysis script
├── collect_latency.py       # Reconstructed TCP connection-latency collector
├── requirements.txt         # Python dependencies
├── sn-article.tex           # LaTeX source of the research report
├── references.bib           # Bibliography
├── fig_*.pdf                # Figures used by the report
└── README.md
```

## Requirements and setup

- Python 3.10 or later (check package compatibility if using a different version)
- Internet access to reach the measurement hosts when running the collector

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the scripts

First inspect the available command-line options:

```bash
python collect_latency.py --help
python analyze_latency.py --help
```

The collector contacts configured public HTTPS hosts and records TCP connection timing. Use the options shown by its `--help` output to choose a duration, probe interval, and output file.

The analysis script processes a measurement CSV and writes prediction metrics. Use its `--help` output to confirm the required input and output arguments for the version in this repository.

### Expected measurement fields

The analysis workflow is documented around these fields:

| Field | Meaning |
|---|---|
| `timestamp` | Measurement timestamp |
| `server` | Hostname or server label |
| `rtt_ms` | Measured connection latency in milliseconds; blank for failed probes |
| `failed` | `1` for a failed probe, otherwise `0` |
| `load_on` | Optional load-state indicator (`0` or `1`) |

The collector may also record error details and a network label, depending on the script version.

**Background load:** A load-state field is only a label. It does not, by itself, generate background traffic. If you study loaded-link behaviour, arrange and document background traffic separately and only on a network you are authorized to use.

## Findings reported in the paper

The research report gives the following headline results for its original experimental datasets:

- On the reported Wi-Fi run, Ridge Regression achieved a reported MAE of **38.50 ms**.
- On the reported mobile-hotspot run, EWMA achieved a reported MAE of **92.24 ms**.

These are values reported in the paper. The original raw data and original scripts are not included here, so these figures have **not** been independently reproduced from the files currently available in this repository.

## Research report

The LaTeX source is `sn-article.tex`, with the bibliography in `references.bib` and supporting figures in the `fig_*.pdf` files.

The source uses the Springer Nature `sn-jnl` document class. You may need the appropriate class files from the official Springer Nature LaTeX template to compile the paper locally or in Overleaf.

## Limitations and data availability

- The original Wi-Fi and mobile-hotspot measurement CSV files are not currently included.
- No public URL for the original dataset is currently available.
- The Python scripts are reconstructed reference implementations, not the original scripts used to generate the paper's reported results.
- The collector measures TCP connection completion time using Python sockets; this is not identical to instrumenting the TCP handshake packets directly.
- The collector does not generate background network traffic.
- Results from newly collected data may differ from the results in the report.
- The current reference implementation does not claim to implement every statistical test, ablation, prediction-horizon experiment, or result-generation step described in the paper.

## Responsible use

The collector contacts public HTTPS hosts. Use it at a reasonable rate, follow applicable terms and network policies, and run measurements only from networks and systems you are authorized to use.

## Acknowledgement

AI-assisted tools were used during project preparation and documentation. This README distinguishes the available reconstructed reference scripts from the unavailable original experimental implementation and dataset.
