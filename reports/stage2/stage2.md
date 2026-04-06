# Stage 2 Report

Date: 2026-04-06;
Stage: Standard ReID (OSNet) and Association Sensitivity;
Notebook: experiments/02_stage2_standard_reid.ipynb;

## Goal

Find a practical balance between geometric association (IoU) and appearance similarity (ReID) to reduce identity switches in crowded soccer scenes;

## What Was Implemented

- Standard ReID pipeline with OSNet embeddings integrated into tracker association;
- Hyperparameter grid search on a hard validation sequence (`test/SNMOT-123`);
- Multi-source evaluation run on 10 SoccerNet sequences using a fixed Stage 2 configuration;
- Export of aggregate metrics, per-source metrics, and grid-search history;

## Evaluation Setup

- Detector: YOLO family detector used by the Stage 2 notebook pipeline;
- Tracker: ByteTrack with ReID-aware matching;
- ReID model: OSNet;
- Grid-search target source: `test/SNMOT-123`;
- Grid-search parameters: `association_alpha` in [0.70, 0.80, 0.90, 0.95, 0.98] and `match_threshold` in [0.75, 0.80, 0.85];
- Multi-source evaluation: 10 sources, 120 processed frames per source;

## Grid Search Summary (SNMOT-123)

Best run (by IDF1 and MOTA in this sweep):

- `association_alpha`: 0.70;
- `match_threshold`: 0.75;
- MOTA: 18.485;
- IDF1: 12.791;
- ID Swaps: 1117;
- FPS: 16.932;

Worst run:

- `association_alpha`: 0.98;
- `match_threshold`: 0.85;
- MOTA: -1.162;
- IDF1: 3.270;
- ID Swaps: 1502;
- FPS: 13.214;

## Aggregate Metrics (10 Sources)

| Metric | Value |
| --- | ---: |
| MOTA | 6.836 |
| IDF1 | 8.670 |
| ID Swaps | 6546 |
| FPS (avg across sources) | 10.138 |
| Sources Evaluated | 10 |
| Association Alpha | 0.60 |

## Per-Source Metrics

| Source | MOTA | IDF1 | ID Swaps | FPS | Processed Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| test/SNMOT-116 | 11.108 | 12.663 | 433 | 11.698 | 120 |
| test/SNMOT-117 | 4.983 | 2.488 | 854 | 9.416 | 120 |
| test/SNMOT-118 | 0.101 | 2.161 | 1059 | 7.290 | 120 |
| test/SNMOT-119 | 5.806 | 4.380 | 125 | 15.859 | 120 |
| test/SNMOT-120 | 2.916 | 5.554 | 742 | 9.181 | 120 |
| test/SNMOT-121 | 7.769 | 8.501 | 684 | 9.483 | 120 |
| test/SNMOT-122 | 10.791 | 13.976 | 562 | 10.478 | 120 |
| test/SNMOT-123 | 2.227 | 2.530 | 1205 | 7.087 | 120 |
| test/SNMOT-124 | 9.771 | 17.493 | 446 | 10.676 | 120 |
| test/SNMOT-125 | 14.091 | 16.821 | 436 | 10.208 | 120 |

## Interpretation

- ReID contribution is important on difficult crowded scenes, and the sweep shows quality degradation when `association_alpha` shifts too far toward geometry-only matching;
- Match confidence is highly sensitive: increasing `match_threshold` from 0.75 to 0.85 consistently increases ID Swaps and can push MOTA to negative values on SNMOT-123;
- Stage 2 remains a valuable improvement stage, but identity consistency is still limited for same-uniform players and heavy occlusions;

## Artifacts

- reports/stage2/stage2.md;
- reports/stage2/stage2_metrics.csv;
- reports/stage2/stage2_metrics_by_source.csv;
- reports/stage2/stage2_grid_search.csv;

## Next Steps

- Reduce background bias in appearance descriptors by introducing segmentation-aware preprocessing;
- Keep the best-performing sweep region (`association_alpha` near 0.70 and `match_threshold` near 0.75) as a starting point for Stage 3 experiments;
- Compare identity quality and latency trade-offs after segmentation is enabled;
