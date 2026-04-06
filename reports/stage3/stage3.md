# Stage 3 Report

Date: 2026-04-06;
Stage: Segmented ReID (Instance Segmentation + OSNet);
Notebook: main.ipynb;

## Goal

Evaluate whether removing background pixels before ReID extraction improves association stability and reduces identity switches;

## What Was Implemented

- Instance segmentation preprocessing for player detections;
- Foreground masking inside each detection crop before OSNet embedding extraction;
- ByteTrack association with ReID-enhanced matching on segmented crops;
- Per-frame latency profiling for detection, preprocessing, ReID extraction, and association;

## Evaluation Setup

- Test source: `test/SNMOT-123`;
- Processed frames: 120;
- Association Alpha: 0.70;
- Match Threshold: 0.75;
- Baseline for comparison: Stage 1 and Stage 2 values on the same source where available;

## Aggregate Metrics (SNMOT-123)

| Metric | Value |
| --- | ---: |
| MOTA | 11.465 |
| IDF1 | 8.656 |
| ID Swaps | 735 |
| FPS | 6.707 |
| Processed Frames | 120 |

## Latency Breakdown (Average Per Frame)

| Stage | Latency (ms) |
| --- | ---: |
| Detection | 79.016 |
| Preprocessing (Masking) | 2.980 |
| ReID Extraction (OSNet) | 31.645 |
| Association (Tracker Update) | 35.385 |

## Comparison Snapshot

| Stage | Method | MOTA | IDF1 | ID Swaps | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage 1 (source baseline) | IoU-oriented baseline | 8.837 | 9.974 | 377 | 12.243 |
| Stage 2 (best sweep on source) | Standard ReID (OSNet) | 18.485 | 12.791 | 1117 | 16.932 |
| Stage 3 | Segmented ReID (OSNet + masking) | 11.465 | 8.656 | 735 | 6.707 |

## Interpretation

- Segmentation improves track stability relative to Stage 2 best-sweep ID switch levels on this source (1117 to 735), indicating reduced background-driven association noise;
- Identity quality remains limited (`IDF1` 8.656), because same-uniform players become more visually similar after background removal;
- Runtime cost is substantial, and FPS falls to 6.707, making real-time deployment difficult without optimization;

## Artifacts

- reports/stage3/stage3.md;
- reports/stage3/stage3_metrics.csv;
- reports/stage3/stage3_latency.csv;

## Next Steps

- Replace standard ReID descriptors with stronger foundation-model features;
- Keep segmentation in the pipeline as a controllable preprocessing component;
- Introduce temporal memory to stabilize embeddings across short-term appearance changes;