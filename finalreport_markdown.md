# YOLO26-Based Driving Scene Detection for Risk-Aware Perception

**MIE1517 - Introduction to Deep Learning, University of Toronto**  
**Team 20:** Yanzhi Deng, Jimmy Jin, Edwin Xu, Zichun Xu

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/blob/main/finalreport.ipynb) [GitHub Repo](https://github.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception)

> Opening this notebook in Colab creates a working copy. The source notebook in GitHub remains unchanged unless a user already has write access to this repository.

This notebook is the **single final report notebook** for our project. It is organized as a concise walkthrough of the completed system: the problem setup, the key preprocessing decision, the final YOLO26n detector, the interpretable risk module, the final qualitative results, and the main lessons learned.

*Project summary: BDD100K preprocessing feeds the final YOLO26n detector, whose outputs are aggregated into an interpretable frame-level risk signal and visualized in both camera-view and top-down 2D form.*

### Project Overview: Result Preview

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/project_overview_demo_1.gif" alt="Result preview demo 1" width="760"/>

*Demo clip 1. Final annotated dashcam output showing bounding-box detections, class counts, and the smoothed risk panel on an urban intersection scene.*

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/project_overview_demo_2.gif" alt="Result preview demo 2" width="760"/>

*Demo clip 2. A second short segment from the final annotated video, illustrating how the risk signal evolves over time in a different traffic configuration.*

## Problem

Road traffic safety remains a major real-world challenge. About **1.19M** road fatalities were estimated in the World in each year.  [WHO:Road Safty](https://www.who.int/health-topics/road-safety#tab=tab_1)

Our project asks a narrower technical question:

> **How effectively can object-level detections from dashcam imagery be converted into a stable, interpretable scene-level driving risk signal?**

Our goal:
> **Detect road objects + Quantify scene danger**

## End-to-End Pipeline

The final system follows four stages:

1. **Data strategy and label redesign**  
   Compress the original long-tailed BDD100K label space into four risk-relevant classes.

2. **YOLO26n object detection**  
   Detect `vehicle`, `person`, `traffic light`, and `traffic sign` from each frame.

3. **Risk Aggregation**
  
    Combine detection outputs into a single risk score by considering:
    - object proximity (distance)
    - object importance (e.g., pedestrian vs vehicle)
    - temporal smoothing across frames

4. **Dual-view visualization**  
   Render the results as both a camera-view overlay and a top-down 2D risk display.

This notebook follows that same order so the final report reads like a walkthrough of the finished system, not a dump of every intermediate experiment.

# 1. Environment Setup and Dataset Overview

We first import the required libraries and define the dataset paths. A separate output directory is used for the converted four-class dataset so that old label versions are not mixed with the current conversion.

```python
from pathlib import Path
import json
import shutil
from collections import Counter, defaultdict
import math
import random
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown
import torch
from ultralytics import YOLO
```

## 1.1 Dataset Overview
We use **BDD100K**, a large-scale dashcam dataset with diverse weather, lighting, and traffic conditions.

- Total images: **100,000**
- Training split: **70,000**
- Validation split: **10,000**
- Test split: **20,000**

BDD100K provides object-level labels, but it does **not** provide frame-level risk annotations. That constraint shaped the project: detection could be trained directly, but risk had to be formulated as an interpretable downstream proxy rather than an end-to-end supervised target.

[BDD100K download page](http://bdd-data.berkeley.edu/download.html)

---

```python
ROOT = Path(".")
IMG_ROOT = ROOT / "bdd100k_images_100k" / "100k"
LBL_ROOT = ROOT / "bdd100k_labels" / "100k"

# A separate output directory is used for the four-class dataset.
# This prevents old label versions from being mixed with the current conversion.
OUT_ROOT = ROOT / "data"

DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'newdata_yolo26_960run_100_32'
TEST_RESULT_DIR = ROOT / 'test_result'
VIZ_OUTPUT_DIR = ROOT / 'yolo_viz_package' / 'outputs'

# Reproducibility
random.seed(42)
np.random.seed(42)

SPLITS = ["train", "val", "test"]

print("ROOT exists:", ROOT.exists())
print("Image root exists:", IMG_ROOT.exists())
print("Label root exists:", LBL_ROOT.exists())
print("Output root:", OUT_ROOT)
```

    ROOT exists: True
    Image root exists: True
    Label root exists: True
    Output root: data_4class\data

## 2. Data Selection Strategy: Why Four Classes

> **The original BDD100K labels are highly imbalanced and semantically uneven**.

A few classes dominate the annotations, while many others appear too rarely to be learned reliably in our setup. We also found large variation in the number of retained boxes per image, which changes scene difficulty substantially across the training split.

To make the detector both more trainable and more aligned with downstream risk estimation, we retained only the four categories that matter most to our final task:

- `vehicle`
- `person`
- `traffic light`
- `traffic sign`

This redesign reduced sparsity, simplified the learning target, and produced detector outputs that map naturally into the later risk module.

## 2.1. BDD100K Dataset Comparison: Original VS. Converted
The images below are the BDD100K class distribution visualization: original  VS converted

#### Original BDD100K distribution

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/download.png" alt="Original BDD100K class distribution" width="780"/>

*Original BDD100K class distribution before label merging. A few categories dominate the dataset, while several others appear only rarely.*

#### Converted four-class distribution

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/dataset_distribution.png" alt="Converted four-class distribution across train, val, and test splits" width="900"/>

*Converted four-class distribution across the train, val, and test splits after remapping BDD100K labels into the four safety-relevant classes used by our detector.*

---
### The two Images below show the BDD100k box geometry: original vs converted

#### Original BDD100K box geometry

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/box_geometry_original.png" alt="Original BDD100K box geometry" width="820"/>

*Original BDD100K box geometry before label conversion.*

#### Converted YOLO-format box geometry

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/box_geometry_converted.png" alt="Converted four-class box geometry" width="820"/>

*Converted four-class YOLO box geometry after remapping and normalization.*

## 3. Final Detector: YOLO26n

We use **Ultralytics YOLO26n** as the final perception backbone. Earlier iterations with denser label spaces and lower-resolution training were noticeably weaker, so the final system standardizes on four-class preprocessing plus a higher-resolution YOLO26n detector.

### Final training setup

| Parameter | Final value |
| --- | --- |
| Model | `yolo26n.pt` |
| Task | `detect` |
| Input size | `960` |
| Epochs | `100` |
| Batch size | `32` |
| Optimizer | `auto` |
| AMP | `True` |
| Device | `8,9` |
| Training time | `67,625.1 s` (`~18.8 hours`) |

### Why this detector worked better

- `960px` input improved visibility of small traffic lights and signs.
- The nano model remained lightweight enough for video-style deployment demos.
- YOLO-style outputs, namely bounding boxes, class IDs, and confidences, connect directly to the downstream risk model.

## 3.1. Training

> [!CAUTION]
> **This final training run was executed on a server with 2 × RTX A6000 GPUs.**

```python
def train(epochs=100, batch_size=32, device='Server GPU'):
    print(f"--> PyTorch Version: {torch.__version__}")

    # 1. Load Standard YOLO Model (Not OBB)
    # Using 'yolo26n.pt' automatically sets the task to 'detect'
    print("--> Loading Model (YOLO26 Standard Detection)...")
    model = YOLO('yolo26n.pt')

    print("--> Starting Training...")

    model.train(
        data='dataset.yaml', # Update this filename if you change your yaml
        epochs=epochs, # change to other number of epochs
        imgsz=960,
        batch=batch_size,
        device=device, # Use GPU 0; change to 'cpu' if you want to train on CPU 'mps' for Apple Silicon
        project= os.path.join(os.getcwd(), 'results'),
        name=f'run_{epochs}_{batch_size}', # Changed run name
        # Performance Settings
        workers=3,
        amp=True,
        exist_ok=True,
        # dropout = 0.2
    )

if __name__ == '__main__':
    train()
```

### 3.2 Training Results

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/TRAIN_VAL_CURVE.png" alt="Training and validation curves" width="860"/>

*Training and validation curves from the final YOLO26n run, showing the evolution of loss and detection metrics over 100 epochs.*

## 4. Quantitative Results

The final model achieved the following headline metrics on the test split:

| Metric | Value |
| --- | ---: |
| mAP@50 | **0.6981** |
| mAP@50-95 | **0.3656** |
| Precision | **0.7495** |
| Recall | **0.6164** |

Overall, the detector performs well for a compact four-class driving-scene model. The strongest category is `vehicle`, while recall is still weaker for smaller or more difficult targets such as pedestrians and some traffic-control objects.

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/SECTION5_CONFUSIONMATRIX.png" alt="Normalized confusion matrix" width="900"/>

*Normalized confusion matrix for the final detector, summarizing class-wise prediction quality on the held-out evaluation split.*

## 5. Risk Scoring

We define the total driving risk ($R$) for each frame as the combined contribution of immediate collision threats and scene-level complexity. This is expressed by the integrated formula:

$$
R = \sum_{i \in \{\text{direct}\}} (w_{c_i} \cdot S_i \cdot P_i) + R_{\text{indirect}}
$$

This score is smoothed over time using an Exponential Moving Average (EMA) to produce a stable perception-to-risk signal.

---

### 5.1 Direct Risk

Direct risk is computed from vehicles and persons. For each detected object $i$, we combine its class importance $w_{c_i}$, a size factor $S_i$ (as a proximity proxy), and a horizontal position factor $P_i$.

$$
R_{\text{direct}} = \sum_{i \in \{\text{vehicle, person}\}} w_{c_i} \cdot S_i \cdot P_i
$$

<br>

> **Note on Direct Risk Components:**
>
> *   **Size Factor ($S_i$):** Uses bounding box area as a proxy for proximity and inherent size: $S_i = \sqrt{\frac{A_i}{A_{\text{img}}}}$.
> *   **Position Factor ($P_i$):** Emphasizes objects near the horizontal center: $P_i = \lambda + (1-\lambda)(1 - 2|x_i - 0.5|)$.
> *   **Class Weights ($w_c$):** **Vehicle** = 1.0, **Person** = 1.5.

---

### 5.2 Indirect Risk

Indirect risk accounts for contextual traffic-control objects. These increase scene complexity and cognitive load without representing an immediate physical threat.

$$
R_{\text{indirect}} = \alpha N_{\text{light}} + \beta \log(1 + N_{\text{sign}})
$$

<br>
<br>

> **Note on Indirect Risk Components:**
>
> *   **Traffic Light Risk ($R_{\text{light}}$):** Scaled by the count of lights: $R_{\text{light}} = \alpha N_{\text{light}}$.
> *   **Traffic Sign Risk ($R_{\text{sign}}$):** Scaled logarithmically by the count of signs: $R_{\text{sign}} = \beta \log(1 + N_{\text{sign}})$.

```python
def compute_frame_risk(results, img_width, img_height):
    """Compute per-frame driving risk from YOLO detection results.
    Returns a dict with total_risk = direct + indirect, per-class counts, and a
    per-object metadata list (object_risks) used downstream for rendering.
    """
    img_area = img_width * img_height

    direct_risk = 0.0
    num_vehicle = 0
    num_person = 0
    num_sign = 0
    num_light = 0
    # Per-object metadata carried through the pipeline so that rendering
    # functions (bounding boxes, risk flash) can access class, confidence,
    # box coordinates, and the individual partial_risk contribution.
    object_risks = []

    if results.boxes is None or len(results.boxes) == 0:
        return {
            "direct_risk": 0.0,
            "indirect_risk": 0.0,
            "total_risk": 0.0,
            "counts": {"vehicle": 0, "person": 0, "traffic light": 0, "traffic sign": 0},
            "object_risks": [],
        }

    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        box_w = max(x2 - x1, 0.0)
        box_h = max(y2 - y1, 0.0)
        box_area = box_w * box_h

        center_x = (x1 + x2) / 2.0
        # Normalise centre x to [0, 1] so that positional risk is resolution-independent.
        center_x_norm = center_x / img_width if img_width > 0 else 0.5

        # pos_factor linearly interpolates between LAMBDA_EDGE (at frame edges)
        # and 1.0 (at frame centre).  |center_x_norm - 0.5| measures horizontal
        # distance from centre; multiplying by 2 maps that to [0, 1].
        pos_factor = LAMBDA_EDGE + (1 - LAMBDA_EDGE) * (1 - 2 * abs(center_x_norm - 0.5))
        pos_factor = max(LAMBDA_EDGE, min(1.0, pos_factor))

        # size_factor: sqrt(box_area / img_area).  Proportional to apparent size.
        # Larger detected objects are physically closer and therefore riskier.
        size_factor = math.sqrt(box_area / img_area) if img_area > 0 else 0.0
        partial_risk = 0.0

        if cls_id == 0:
            num_vehicle += 1
            partial_risk = DIRECT_WEIGHTS[cls_id] * size_factor * pos_factor
            direct_risk += partial_risk
        elif cls_id == 1:
            num_person += 1
            partial_risk = DIRECT_WEIGHTS[cls_id] * size_factor * pos_factor
            direct_risk += partial_risk
        elif cls_id == 2:
            num_light += 1
        elif cls_id == 3:
            num_sign += 1

        object_risks.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "partial_risk": partial_risk,
        })

    # Indirect risk: linear from traffic lights, logarithmic from signs.
    # log1p(n) = log(1 + n) so a single sign still contributes, but 10 signs
    # contribute only ~2.4x what 1 sign does — preventing sign-heavy streets
    # from dominating the risk score.
    indirect_risk = ALPHA_LIGHT * num_light + BETA_SIGN * math.log1p(num_sign)
    total_risk = direct_risk + indirect_risk

    return {
        "direct_risk": float(direct_risk),
        "indirect_risk": float(indirect_risk),
        "total_risk": float(total_risk),
        "counts": {
            "vehicle": num_vehicle,
            "person": num_person,
            "traffic light": num_light,
            "traffic sign": num_sign,
        },
        "object_risks": object_risks,
    }
```

---

### 5.3. Temporal Smoothing for Video

To reduce frame-to-frame fluctuation in video, we apply an exponential moving average ([EMA](https://www.investopedia.com/terms/e/ema.asp)):

$$
R_t^{\text{EMA}} = \gamma R_{t-1}^{\text{EMA}} + (1-\gamma)R_t
$$

where:

- $R_t$ is the raw risk at frame $t$
- $R_t^{\text{EMA}}$ is the smoothed risk
- $\gamma = 0.8$ in our implementation

---

### 5.4. Risk Level Classification

Finally, the smoothed risk is mapped to three qualitative levels:

$$
\text{Risk Level} =
\begin{cases}
\text{Low}, & R < T_1 \\
\text{Medium}, & T_1 \le R < T_2 \\
\text{High}, & R \ge T_2
\end{cases}
$$

where typical thresholds are:

- $T_1 = 1.0$
- $T_2 = 2.5$

```python
def update_ema_risk(current_risk, previous_ema=None, momentum=0.8):
    """Exponential moving average (EMA) for temporal smoothing.

    With momentum = 0.8, 80% of the smoothed value comes from the previous
    EMA and only 20% from the new observation.  This provides temporal
    stability — the displayed risk level won't flicker wildly between frames
    when detections appear and disappear due to occlusion or model noise.
    The effective half-life is ~3 frames (ln(2)/ln(1/0.8) ≈ 3.1).
    """
    if previous_ema is None:
        return current_risk
    return momentum * previous_ema + (1 - momentum) * current_risk

def risk_level(score, low_th=1.0, high_th=2.5):
    """Map a continuous risk score to a discrete qualitative level.
    """
    if score < low_th:
        return "LOW"
    elif score < high_th:
        return "MEDIUM"
    return "HIGH"

```

## 6. Final Qualitative Results

Before testing the full visualization pipeline on unseen Toronto dashcam footage, we also inspected representative predictions on the held-out **BDD100K test split**. The side-by-side comparisons below show that the final detector generally captures the main vehicles, pedestrians, and traffic-control objects with good spatial alignment relative to ground truth. The remaining differences are concentrated in smaller, darker, or more cluttered regions rather than in the dominant scene structure.

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/testset_result_1.png" alt="Held-out BDD100K test example 1" width="980"/>

*Held-out BDD100K test example (nighttime scene): the predicted boxes recover the main vehicles and traffic-control objects with scene layout that remains close to the ground-truth annotations.*

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/testset_result_2.png" alt="Held-out BDD100K test example 2" width="980"/>

*Held-out BDD100K test example (urban intersection): the detector preserves the major scene structure and object categories, especially for vehicles and larger traffic-control objects, while minor discrepancies remain in denser regions.*

### Unseen Toronto Dashcam Demo

The completed system was then tested on unseen Toronto dashcam footage. Each row below shows one matched sequence progressing from the original frame to the camera-view detection/risk overlay and then the top-down 2D risk visualization.

![3x3 qualitative grid](https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/grid_3x3_sequences.png)

*Each row is one sequence: original frame, camera-view detection result, and matched top-down 2D risk view. The camera view preserves raw evidence, while the top-down view makes spatial layout and overall risk level easier to read.*

### Representative Interesting Failure Case

This failure is best understood as a **semantic compression issue**. The stroller is classified as `vehicle`, not because the detector completely fails, but because [BDD100K](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf) does not provide stroller as a dedicated detection category, and our additional four-class remapping further compresses the label space. As a result, the model is forced to map an uncommon object to the nearest frequent category.

This interpretation is consistent with long-tailed detection literature, which shows that scarce or ambiguous classes tend to cluster with analogous frequent ones when their representations are not sufficiently discriminative ([Li et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Adaptive_Hierarchical_Representation_Learning_for_Long-Tailed_Object_Detection_CVPR_2022_paper.html)).

In our case, the stroller is not completely ignored, but its semantics are distorted. Although the category assignment is imperfect, the scene still produces a nonzero risk signal. A stroller implies a more vulnerable and context-sensitive situation than a generic vehicle, yet under our current formulation it inherits the geometric and semantic treatment of the `vehicle` class. This makes the risk output directionally reasonable, but semantically less precise.

<img src="https://raw.githubusercontent.com/DDaswE/YOLO26-Based-Driving-Scene-Detection-for-Risk-Aware-Perception/main/assets/section7_STROLLER.png" alt="Representative stroller failure case" width="900"/>

*Semantic compression failure: the stroller is mapped to the merged `vehicle` category, so the scene still produces risk, but its vulnerability semantics are only partially preserved.*

## 7. Discussion

### Overall Results

The final model performs well as a compact perception-to-risk prototype, achieving a strong balance between real-time inference and interpretable output. While the detection backbone is standard, the **modular risk layer** is what makes the results interesting: it successfully translates raw bounding boxes into a human-readable safety signal.

A surprising observation was that the **Position Factor ($P_i$)** was just as critical as object size; a small object (far away) directly in the center often represented a more persistent 'threat' in the EMA smoothing than a large object briefly appearing at the periphery. This suggests that spatial context is a powerful, low-cost proxy for intent.

### What worked well

- **Four-class consolidation:** Reduced class confusion and aligned the detector more closely with the downstream risk-estimation objective.
- **Temporal Smoothing (EMA):** Stabilized frame-level risk outputs and reduced flicker, which is especially important for video-based perception systems.
- **Higher input resolution:** Improved visibility for small and distant objects, which are often the most important for early safety signaling.

### Main limitations

- **Recall remains weaker than precision:** This is an important safety limitation, since missed detections can propagate directly into underestimated scene-level risk.
- **No explicit distance or motion reasoning:** Bounding-box geometry is only a proxy for proximity, and the system does not explicitly model depth, speed, or full 3D scene structure.
- **Hand-tuned risk parameters::** Thresholds and class weights are manually calibrated rather than learned from risk-labeled supervision.

> ### Practical Challenge
> - **High memory demand at imgsz=960:** Higher-resolution training improved small-object visibility, but it also increased computational cost and constrained local experimentation.
---

### Comparison with Related Work

- **Compared with standard driving-perception systems, our project is more modular and focused on interpretability.**  
  Works such as [YOLOP](https://arxiv.org/pdf/2108.11250) and the modular framework of [Kazerouni et al. (2023)](https://arxiv.org/pdf/2303.16710) focus primarily on perception quality and richer scene understanding, while our project emphasizes whether object detections can be translated into a stable and human-readable safety signal.

- **Compared with richer prior systems, our current framework is still limited in geometric and temporal reasoning.**  
  Prior work has incorporated stronger cues such as monocular depth estimation, explicit distance measurement, and temporal modeling. For example, [Kazerouni et al. (2023)](https://arxiv.org/pdf/2303.16710) include depth-aware perception modules, while [AAT-DA (Kumamoto et al., 2025)](https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Kumamoto_AAT-DA_Accident_Anticipation_Transformer_with_Driver_Attention_WACVW_2025_paper.pdf) uses temporal and relational reasoning for accident anticipation. By contrast, our method relies on single-frame detections plus heuristic aggregation, so it remains weaker in depth awareness, motion reasoning, and complex-scene robustness.

- **Compared with end-to-end or accident-anticipation models, our main advantage is transparency.**  
  Rather than mapping pixels directly to control or accident predictions, our system preserves explicit intermediate reasoning through detected objects, risk factors, and visualization. This differs from end-to-end driving approaches such as [Bojarski et al. (2016)](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), and makes our framework lighter, easier to debug, and easier to explain, even if it is less powerful than fully supervised temporal models.

---

### What We Learned & Key Insights

The biggest insight from this project is that **data design and objective framing mattered more than model complexity**.

We learned that:

- **Data preprocessing was a major part of the project effort.**  
  Converting BDD100K JSON annotations into YOLO format and remapping 10 categories into 4 safety-relevant classes was not a trivial setup step. In practice, preprocessing and label design shaped the final performance almost as much as the detector itself.

- **Resolution choice was more important than we initially expected.**  
  Increasing the input size from `640` to `960` produced a larger improvement than many smaller hyperparameter changes. This showed that small-object visibility was a critical bottleneck for our task.

- **Processed data and labels must always be visualized and manually checked.**  
  Several issues only became obvious after inspecting converted labels and detection outputs directly. This reinforced the importance of validating the actual model input, not just trusting the conversion pipeline.

- **Simplification is a feature, not just a compromise.**  
  Merging labels did not only make training easier; it also made the downstream risk logic more stable by reducing semantic noise and focusing the detector on safety-relevant classes.

- **Heuristic risk models can still be effective when designed carefully.**  
  By using an interpretable formula rather than an end-to-end neural risk predictor, we were able to build a practical scene-level risk signal without requiring additional risk-labeled data.

- **Interpretability is a practical advantage.**  
  Because the risk computation is transparent, we can explain exactly why a frame is flagged as higher risk, for example due to a large central vehicle, multiple traffic-control objects, or sustained risk accumulation over time.

- **Temporal consistency is non-negotiable in video-based systems.**  
  Without EMA smoothing, the risk signal flickered heavily from frame to frame and the qualitative output felt unreliable, even when detection metrics were reasonable. This emphasized an important gap between standard computer-vision metrics and real user-facing system behavior.

## 8. Future Work

The most natural next steps follow directly from the limitations of the current system.

- **Add depth, speed, and motion cues:**  
  The current risk model relies on bounding-box geometry as a proxy for proximity. Incorporating monocular depth, optical flow, or object motion would make the risk estimate more physically grounded.

- **Incorporate object tracking:**  
  Tracking across frames would provide stronger temporal consistency than frame-wise detection plus EMA smoothing alone, and would allow risk to reflect persistent object behavior over time.

- **Refine traffic-control semantics:**  
  Future versions could distinguish more informative states or subtypes for `traffic light` and `traffic sign`, rather than treating them as coarse categories only. This would make indirect risk estimation more context-sensitive.

- **Learn the risk function more systematically:**  
  The current risk weights and thresholds are hand-tuned for interpretability. A future direction is to replace them with a learned or weakly supervised formulation that preserves transparency while improving calibration.

## 9. Reproducibility and Key Artifacts

This notebook is intentionally concise. The main supporting files are included directly in this submission package:

- Data preprocessing notebook: [`bdd100k_data_loading_visualization_4class.ipynb`](bdd100k_data_loading_visualization_4class.ipynb)
- Dataset-distribution plotting utility for the converted four-class split summary: [`generate_dataset_distribution.py`](generate_dataset_distribution.py)
- Training configuration: [`dataset.yaml`](dataset.yaml)
- Training script: [`train_yolo.py`](train_yolo.py)
- Final training artifacts and weights: [`newdata_yolo26_960run_100_32`](newdata_yolo26_960run_100_32)
- Evaluation script and saved evaluation outputs: [`evaluate_yolo.py`](evaluate_yolo.py) and [`test_result`](test_result)
- Ground-truth versus prediction comparison utility for selected test images: [`compare_gt_pred.py`](compare_gt_pred.py)
- Video-generation code bundle (run `create_video/yolo_viz.py` to generate video): [`create_video.zip`](create_video.zip)
- Final unseen-data demo videos: [Google Drive folder](https://drive.google.com/drive/folders/1_odBVfQlX-BqP8tH-Jy6-LwjgyKkdW8l?usp=sharing)

Large converted-data folders and full demo videos are kept separate from the lightweight submission package, but the files above contain the code, configuration, trained weights, and result figures needed to inspect the final system.

## 10. References

- Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Jackel, L. D., Monfort, M., Muller, U., Zhang, J., Zhang, X., Zhao, J., & Zieba, K. (2016). *End to end learning for self-driving cars*. NVIDIA Technical Blog. https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
- Investopedia. (n.d.). *Exponential moving average (EMA)*. https://www.investopedia.com/terms/e/ema.asp
- Kazerouni, A., Heydarian, A., Soltany, M., Mohammadshahi, A., Omidi, A., & Ebadollahi, S. (2023). *An intelligent modular real-time vision-based system for environment perception*. arXiv. https://arxiv.org/abs/2303.16710
- Kumamoto, Y., Ohtani, K., Suzuki, D., Yamataka, M., & Takeda, K. (2025). *AAT-DA: Accident anticipation transformer with driver attention*. In *Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops* (pp. 1142-1151). Computer Vision Foundation. https://openaccess.thecvf.com/content/WACV2025W/HAVI/html/Kumamoto_AAT-DA_Accident_Anticipation_Transformer_with_Driver_Attention_WACVW_2025_paper.html
- Li, B. (2022). *Adaptive hierarchical representation learning for long-tailed object detection*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2313-2322). Computer Vision Foundation. https://openaccess.thecvf.com/content/CVPR2022/html/Li_Adaptive_Hierarchical_Representation_Learning_for_Long-Tailed_Object_Detection_CVPR_2022_paper.html
- World Health Organization. (n.d.). *Road safety*. https://www.who.int/health-topics/road-safety#tab=tab_1
- Wu, D., Liao, M.-W., Zhang, W.-T., Wang, X.-G., Bai, X., Cheng, W.-Q., & Liu, W.-Y. (2022). *YOLOP: You only look once for panoptic driving perception*. *Machine Intelligence Research, 19*, 550-562. https://doi.org/10.1007/s11633-022-1339-y
- Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2020). *BDD100K: A diverse driving dataset for heterogeneous multitask learning*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2636-2645). Computer Vision Foundation. https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html

## Final Takeaway

This project shows that object-level detections from dashcam imagery can be transformed into a stable and interpretable driving-risk signal even without explicit risk labels. The main contribution is therefore not only the detector itself, but the **modular perception-to-risk framework** that connects detection, aggregation, and visualization into a coherent end-to-end system. More broadly, the project demonstrates that meaningful scene-level safety reasoning can be built from standard object-detection supervision when the pipeline is carefully designed for interpretability and robustness.
