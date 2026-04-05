# Comparative Research on Object Identification Methods (ID Assignment)

## 1. Research Objective

Identify the most robust method for assigning and maintaining object IDs under challenging conditions such as occlusions, similar appearance, and changes in perspective. The study will use incremental testing of detection, tracking, and ReID architectures.

## 2. Data Management Strategy

- **Quantitative assessment:** Use the SoccerNet tracking subset to obtain objective metrics from annotated data.
- **Qualitative assessment:** Test final configurations on videos and photos provided by a colleague to verify in-the-wild performance.
- **Storage:** Integrate Google Drive for caching model weights and storing intermediate results such as detection outputs and embeddings.

## 3. Engineering Approach & Project Structure

To keep the project flexible and clean, it follows a hybrid model combining modules and notebooks.

- **Modular core (`.py` files):** Model logic, interfaces, and mathematical calculations live in separate modules to avoid duplication and simplify debugging.
- **Experiment orchestration (`.ipynb`):** Notebooks are reserved for sequential execution, visualization, plotting, and interactive analysis.
- **VS Code & Colab integration:** Development happens in VS Code with the Google Colab extension for remote computations on a T4 GPU.

### Directory Structure

- `core/`: base abstract classes and model registry.
- `models/`: implementations of detectors, trackers, and ReID modules.
- `utils/`: helpers for data loading and metric calculation.
- `configs/`: YAML files with experiment parameters.
- `experiments/`: notebooks for each stage of the research.

## 4. Reproducibility & Deployment

To let other users run the notebooks in Colab, the first cell of each notebook should include an automation script.

- **Repository cloning:** `!git clone https://github.com/ituvtu/research-reid.git`
- **Path configuration:** Add the project root to `sys.path` with `sys.path.append('/content/research-reid')` so modules can be imported like standard libraries.
- **Dependency installation:** Use `uv` instead of standard `pip` by running `!pip install uv && uv pip install -r requirements.txt --system`.
- **Drive mounting:** Optional, for access to large video files and saved model weights.

## 5. Implementation Stages

### Stage 1: Environment Setup and Baseline (Tournament)

**Objective:** Determine the optimal detector and geometric tracker pair.

- Connect Google Drive and install required libraries such as `ultralytics`, `filterpy`, `lapx`, and `TrackEval`.
- Implement a universal SoccerNet loader for images and ground-truth annotations.
- Compare YOLO26 variants (`N`, `S`, `M`, `X`) and YOLOv11 on SoccerNet.
- Compare ByteTrack and OC-SORT using IoU and Kalman filtering.
- **Metrics:** MOTA, FPS, ID Swaps.
- **Expected result:** Select the best detector, for example YOLO26-M, for later stages.

### Stage 2: Standard ReID (Visual Embeddings)

**Objective:** Improve ID stability using visual descriptors.

- Integrate a ReID model such as OSNet into the association pipeline.
- Implement cropping based on bounding boxes from the Stage 1 detector.
- Configure weighted association between appearance and motion cues.
- **Metrics:** IDF1, Rank-1, mAP.
- **Expected result:** Quantify the impact of visual features on reducing ID swaps.

### Stage 3: Segmented ReID (Background Removal)

**Objective:** Test whether object isolation improves embedding quality.

- Add a segmentation module using YOLO26-seg or SAMv2-tiny.
- Preprocess crops by masking the background around the object with black or grey fill.
- Feed segmented crops through the same ReID model used in Stage 2.
- Compare standard ReID against segmented ReID.
- **Expected result:** Conclude whether segmentation reduces background noise effectively.

### Stage 4: Foundation Models (DINOv2/v3 Integration)

**Objective:** Evaluate Vision Transformer capabilities for ReID.

- Integrate DINOv2 as an alternative frozen feature extractor.
- Test DINO on both segmented crops and original crops.
- Analyze feature distributions with t-SNE to understand discriminative power.
- **Metrics:** HOTA, as the main metric for balancing localization and identification.
- **Expected result:** Determine whether heavy foundation models are practical for this task.

## 6. Final Reporting and Visualization

- **Metric table:** Summarize all configurations, including detector, tracker, ReID, segmentation, MOTA, IDF1, and FPS.
- **Correlation charts:** Show how identification accuracy depends on object resolution and occlusion levels.
- **Side-by-side video:** Generate comparative videos for the baseline and the best SOTA configuration on the colleague's data.
- **Heatmap analysis:** Visualize attention zones for DINO and OSNet on crops with bibs.

## 7. Accents and Risks

- **Low-quality crops:** The provided photos have significant motion blur, so the focus should be on blur-resistant models.
- **FPS monitoring:** Measuring inference time for each module is critical for assessing real-time viability.
- **Dataset bias:** Results on SoccerNet may differ from the colleague's videos, so cross-validation is mandatory.
