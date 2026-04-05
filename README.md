# ReID Investigation

Goal: Reproducible research workspace for soccer object detection, tracking, and ReID experiments;

## Overview

This repository contains a modular pipeline for Stage 1 baseline evaluation and later ReID stages;
Core logic is implemented in Python modules, while notebooks are used for orchestration and analysis;

## Project Structure

- `core/`: Abstract contracts for detectors and trackers;
- `models/`: Concrete detector and tracker implementations;
- `utils/`: Data loading, metrics, and video helpers;
- `configs/`: YAML experiment configurations;
- `experiments/`: Research notebooks;

## Requirements

- Python: `3.12`;
- Package manager: `uv`;
- Runtime target: Google Colab with T4 GPU;

## Local Setup

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## Colab Setup

Use the Stage 1 notebook in `experiments/01_stage1_baseline.ipynb`;

For private repositories, provide a GitHub token before running the bootstrap cell;

```python
%env GITHUB_TOKEN=your_read_only_pat
%env REID_REPO_URL=https://github.com/<owner>/<repo>.git
```

Notes: Keep token scopes minimal and never commit tokens or notebook outputs with secrets;

## Running Stage 1

1. Open `experiments/01_stage1_baseline.ipynb`;
2. Run the bootstrap cell to resolve project path and install dependencies;
3. Run the detector and tracker initialization cell;
4. Run dataset loading and metrics cells;

## Repository Hygiene

- `.gitignore` excludes local environments, outputs, caches, and secrets;
- Commit source modules, configs, notebooks, and documentation only;
- Do not commit dataset files, model weights, or personal IDE settings;
