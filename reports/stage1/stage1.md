# Stage 1 Report

Date: 2026-04-05;
Stage: Environment Setup and Baseline;
Notebook: experiments/01_stage1_baseline.ipynb;

## Goal

Determine a stable baseline for soccer tracking using a detector + geometric tracker pair, then measure MOTA, IDF1, ID Swaps, and FPS on SoccerNet tracking sequences;

## What Was Implemented

- Colab-ready project bootstrap with repository resolution and dependency installation;
- Detector initialization in YOLOv26 library mode (`yolo26m`) without local weight files;
- Robust SoccerNet data preparation with fallback extraction for `.zip` archives;
- Annotation support for `.json`, `.csv`, and MOT-style `.txt` files;
- Frame-source fallback for `img1` sequence folders when no video files are present;
- Multi-source evaluation flow with isolated identity spaces between sources;
- Aggregate and per-source metric export to CSV files;

## Evaluation Setup

- Detector: YOLOv26m;
- Tracker: ByteTrack;
- Sources evaluated: 10 sequences;
- Processed frames per source: 120;
- Frame stride: 2;
- IoU threshold for MOT metrics: 0.5;
- Identity isolation across sources: enabled via frame and track ID offsetting;

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| MOTA | 11.331 |
| IDF1 | 16.740 |
| ID Swaps | 1561 |
| FPS (avg across sources) | 19.947 |
| Sources Evaluated | 10 |

## Per-Source Metrics

| Source | MOTA | IDF1 | ID Swaps | FPS | Processed Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| tracking/test/SNMOT-116 | 14.337 | 15.996 | 96 | 16.163 | 120 |
| tracking/test/SNMOT-117 | 13.521 | 20.895 | 45 | 23.941 | 120 |
| tracking/test/SNMOT-118 | 7.244 | 11.943 | 251 | 12.291 | 120 |
| tracking/test/SNMOT-119 | 5.923 | 13.516 | 0 | 36.640 | 120 |
| tracking/test/SNMOT-120 | 6.858 | 14.097 | 243 | 14.487 | 120 |
| tracking/test/SNMOT-121 | 12.836 | 16.215 | 81 | 23.689 | 120 |
| tracking/test/SNMOT-122 | 13.769 | 20.527 | 232 | 20.331 | 120 |
| tracking/test/SNMOT-123 | 8.837 | 9.974 | 377 | 12.243 | 120 |
| tracking/test/SNMOT-124 | 12.667 | 22.828 | 156 | 22.339 | 120 |
| tracking/test/SNMOT-125 | 17.835 | 23.116 | 80 | 17.346 | 120 |

## Main Issues Resolved During Stage 1

- Dataset visibility mismatch between Colab runtime storage and Google Drive mount points;
- Missing video assets in some cases, handled by sequence-frame fallback;
- NumPy 2.x and `motmetrics` compatibility issue (`np.asfarray`) handled with runtime shim;
- Output persistence confusion between `/content` and `/content/drive/MyDrive`;

## Artifacts

- outputs/stage1_metrics.csv;
- outputs/stage1_metrics_by_source.csv;
- reports/stage1/stage1.md;

## Interpretation

- Baseline execution is stable and reproducible across multiple sequences;
- Average runtime is close to real-time on T4 for the tested subset;
- ID consistency remains limited in crowded scenes, as indicated by high total ID Swaps;
- This is an expected baseline outcome before integrating appearance-based ReID in Stage 2;

## Next Steps

- Keep this baseline fixed as a reference for all Stage 2 comparisons;
- Introduce ReID embeddings (for example OSNet) into association logic;
- Re-run on the same 10 sources to compare IDF1 and ID Swaps deltas;
- Expand evaluation to additional splits after Stage 2 stabilization;
