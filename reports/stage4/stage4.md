# Stage 4 Report

Date: 2026-04-06;
Stage: Advanced ReID (DINOv2 + Temporal Memory);
Notebook: main.ipynb;

## Goal

Improve identity consistency for same-uniform players by combining segmentation-cleaned crops, transformer embeddings, and temporal feature memory;

## What Was Implemented

- DINOv2 (`vits14`) feature extraction for ReID;
- Temporal memory with exponential moving average (EMA) per track;
- Segmentation-aware crop preprocessing retained from Stage 3;
- Detailed latency profiling including DINO and OSNet benchmark timings;

## Evaluation Setup

- Test source: `test/SNMOT-123`;
- Processed frames: 120;
- Association Alpha: 0.70;
- Match Threshold: 0.75;
- Temporal Buffer Size: 30;
- Temporal Memory Mode: EMA;
- Temporal EMA Momentum: 0.90;
- DINO embedding dimension: 384;

## Aggregate Metrics (SNMOT-123)

| Metric | Value |
| --- | ---: |
| MOTA | 12.424 |
| IDF1 | 9.364 |
| ID Swaps | 717 |
| FPS | 5.226 |
| Processed Frames | 120 |

## Latency Breakdown (Average Per Frame)

| Stage | Latency (ms) |
| --- | ---: |
| Detection | 75.218 |
| Preprocessing (Masking) | 2.915 |
| DINO ReID Extraction | 50.009 |
| OSNet ReID Extraction (reference) | 29.667 |
| Association (Tracker Update) | 33.434 |
| Embedding Buffer Update | 0.000 |

## Progress vs Stage 3

| Metric | Stage 3 | Stage 4 | Delta |
| --- | ---: | ---: | ---: |
| MOTA | 11.465 | 12.424 | +0.959 |
| IDF1 | 8.656 | 9.364 | +0.708 |
| ID Swaps | 735 | 717 | -18 |
| FPS | 6.707 | 5.226 | -1.481 |

## Interpretation

- DINOv2 plus temporal memory improves identity quality and reduces ID switches relative to Stage 3;
- EMA feature buffering helps smooth short-term appearance noise and occlusion artifacts;
- The quality gain comes with a clear compute trade-off, with throughput dropping to 5.226 FPS;
- Stage 4 is a stronger research baseline for identity stability, but deployment will require acceleration (for example, quantization or TensorRT pipelines);

## Artifacts

- reports/stage4/stage4.md;
- reports/stage4/stage4_metrics.csv;
- reports/stage4/stage4_latency.csv;

## Next Steps

- Profile mixed-precision and inference-optimization paths for ViT-based ReID;
- Evaluate scaling to additional sources with the same parameter set for generalization checks;
- Explore lightweight temporal memory alternatives to recover part of the FPS loss;
