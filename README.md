# Advanced Deep Learning for Multi-View Structural Reasoning in Mammographic Analysis

**Intern:** [Imade Bouftini](https://github.com/ibouftini) &nbsp;&nbsp;&nbsp;&nbsp; **Supervision:** [Youssef ALJ](youssef.alj@um6p.ma)

**Institutions:** [AI Movement - UM6P](https://aim.um6p.ma/) & [École Centrale de Nantes](https://www.ec-nantes.fr/)

---

## 📋 Table of Contents

- [📖 Introduction](#-introduction)
- [🎯 Objectives](#-objectives)
- [🧹 Data Preprocessing & Cleaning](#-data-preprocessing--cleaning)
- [⚙️ Methodology](#️-methodology)
- [🕸️ Graph Construction & Anatomical Reasoning](#️-graph-construction--anatomical-reasoning)
- [🛠️ Implementation](#️-implementation)
- [📊 Results](#-results)
- [💬 Discussion](#-discussion)
- [🔗 References](#-references)

---

## 📖 Introduction

Breast cancer affects **2.3 million women annually** worldwide, with early detection through mammography reducing mortality by 20-40%. However, current Computer-Aided Detection (CAD) systems face significant challenges due to:

- **Extremely low prevalence** (~0.5% in screening populations)
- **High inter-reader variability** among radiologists
- **Complex anatomical structures** with varying tissue density
- **Subtle presentation** of early-stage malignancies

Radiologists naturally employ **multi-view reasoning**, examining both ipsilateral views (CC and MLO of the same breast) and bilateral symmetry (left vs. right breast) to cross-validate findings. This project implements an **Anatomy-aware Graph Neural Network (AGN)** to replicate this clinical reasoning process in automated breast cancer detection.

<div align="center">
  <img src="figures/mammography_views.png" alt="Mammography views and reasoning" width="70%">
  <p><em>Multi-view mammographic analysis: CC and MLO views with anatomical landmarks</em></p>
</div>

---

## 🎯 Objectives

1. **Establish robust preprocessing pipelines** for anatomical landmark detection and graph construction
2. **Implement and evaluate** a state-of-the-art Mask R-CNN architecture for single-view mammographic mass detection
3. **Develop an Anatomy-aware Graph Neural Network** that integrates multi-view reasoning through:
   - **Bipartite Graph Network (BGN)** for ipsilateral correspondence 
   - **Inception Graph Network (IGN)** for bilateral symmetry analysis
4. **Quantitatively assess** the impact of multi-view reasoning on detection performance using clinical metrics

---

## 🧹 Data Preprocessing & Cleaning

### Dataset Overview: CBIS-DDSM

The CBIS-DDSM (Curated Breast Imaging Subset of Digital Dataset for Screening Mammography) serves as our primary dataset, containing digitized film mammograms converted to DICOM format with segmented and labeled regions.

**Dataset Statistics:**
- **1,566 unique patients**
- **3,069 mammographic images** 
- **3,568 annotated findings**
- **Distribution:** 1,457 malignant, 2,111 benign cases

### Data Quality Challenges

The CBIS-DDSM dataset presented several critical inconsistencies requiring systematic correction:

#### 1. Mirrored Images
- **Issue:** Some images mirrored with incorrect laterality (breast positioning in wrong orientation)
- **Prevalence:** 26.7% of the dataset
- **Solution:** Laterality-based orientation correction using column intensity comparison

#### 2. Resolution Discrepancies
- **Issue:** Mask files with different resolutions than original images
- **Impact:** Misaligned annotations affecting training quality
- **Solution:** Bilinear interpolation for resolution matching

#### 3. Filename Inconsistencies
- **Issue:** Filenames not corresponding with CSV descriptions
- **Solution:** Image intensity statistics-based filename correction

#### 4. Artifacts in Mammograms
- **Issue:** Printed annotations, film edges, and non-anatomical information
- **Impact:** Potential model confusion from irrelevant features
- **Solution:** Adaptive thresholding-based artifact removal

#### 5. Corrupted Files
- **Issue:** ROI files replaced by binary masks
- **Solution:** Automated detection and bounding box extraction from largest connected components

### Systematic Data Cleaning Pipeline

#### Artifact Removal and Cropping

Non-anatomical information introduced during digitalization requires removal through coordinate-based cropping and removing artefacts outside the contour:

<div align="center">
  <img src="figures/preprocessing_crop.png" alt="Artifact removal example" width="60%">
  <p><em>Example of cropping results: (a) Original with artifacts (b) Cleaned mammogram</em></p>
</div>

#### Orientation Standardization

Anatomical orientation standardization ensures consistent laterality across the dataset:

<div align="center">
  <img src="figures/preprocessing_flip.png" alt="Orientation correction example" width="60%">
  <p><em>Flipping logic illustration: (a) Incorrect orientation (b) Corrected orientation</em></p>
</div>

#### Corrupted File Recovery

Some ROI files were mistakenly replaced with binary masks. Our solution extracts bounding boxes from the largest connected components:

### Structural Elements Calculation

#### Basic Operations

**Image Acquisition and Normalization:**

Intensity normalization standardizes the dynamic range:

$$I_{norm}(x,y) = \frac{I(x,y) - \min(I)}{\max(I) - \min(I)}$$

**Breast Segmentation using Otsu's Method:**

The optimal threshold $t^*$ maximizes between-class variance:

$$t^* = \underset{t}{\arg\max}\{\omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2\}$$

where $\omega_0(t)$, $\omega_1(t)$ are class probabilities and $\mu_0(t)$, $\mu_1(t)$ are respective mean intensities.

#### Contour Extraction and Smoothing

To address pixel-level noise in breast boundaries, we apply B-spline smoothing with adaptive parameters:

$$S(u) = \sum_{i=0}^{n-1} P_i B_i(u)$$

The smoothing optimization balances fidelity and regularity:

$$\min_S \left\{ \sum_{i=1}^n |C_i - S(u_i)|^2 + s \int_0^1 |S''(u)|^2 du\right\}$$

**Adaptive smoothing parameter:**

$$s = \begin{cases}
10^7 & \text{if view = MLO} \\
100 & \text{if view = CC}
\end{cases}$$

<div align="center">
  <img src="figures/contour_smoothing.png" alt="Contour extraction and smoothing" width="50%">
  <p><em>Contour extraction: (a) Raw extracted contour (b) B-spline smoothed contour</em></p>
</div>

#### Pectoral Muscle Detection

**CC View Approximation:**
For CC views, the pectoral muscle is approximated as a vertical line:

$$x_{pectoral} = \begin{cases}
\min_i x_i & \text{if side = Left} \\
\max_i x_i & \text{if side = Right}
\end{cases}$$

**MLO View Multi-Stage Detection:**

*1. ROI Definition:*

$$\text{ROI} = \begin{cases}
[0, 0.4w] \times [0, 0.6h] & \text{if side = Left} \\
[0.6w, w] \times [0, 0.6h] & \text{if side = Right}
\end{cases}$$

*2. CLAHE Enhancement:*

$$I_{CLAHE} = \text{CLAHE}(I_{ROI}, \text{clipLimit}=3.0, \text{tileGridSize}=(8,8))$$

*3. Combined Thresholding:*

$$T_{Combined} = T_{Otsu} \land T_{Adaptive}$$

*4. Hough Line Detection:*

$$L = \text{HoughLinesP}(E, \rho=1, \theta=\pi/180, \text{threshold}=20)$$

*5. Line Filtering and Scoring:*

$$\text{valid}(L_i) = \begin{cases}
\text{slope}(L_i) < 0 & \text{if side = Left} \\
\text{slope}(L_i) > 0 & \text{if side = Right}
\end{cases}$$

$$\text{score}(L_i) = \text{length}(L_i) \cdot (w_{pos} \cdot \text{pos\_score} + w_{angle} \cdot \text{angle\_score})$$

<div align="center">
  <img src="figures/pectoral_detection.png" alt="Pectoral muscle detection pipeline" width="80%">
  <p><em>Pectoral muscle detection: (a) Original MLO (b) ROI+CLAHE (c) Thresholding (d) Morphological ops (e) Edge detection (f) Final line detection</em></p>
</div>

---

## ⚙️ Methodology

### Multi-View Detection Framework

Our approach extends Mask R-CNN by replacing the standard backbone with an Anatomy-aware Graph Neural Network that processes three mammographic views simultaneously:

$$Y = f(F_{e}, F_{a}, F_{c}; \mathcal{G}_B, \mathcal{G}_I)$$

where:
- $F_{e}, F_{a}, F_{c} \in \mathbb{R}^{HW \times C}$: Feature maps from examined, auxiliary, and contralateral views
- $\mathcal{G}_B$: Bipartite graph for ipsilateral reasoning
- $\mathcal{G}_I$: Inception graph for bilateral analysis

<div align="center">
  <img src="figures/agn_architecture.png" alt="AGN Architecture" width="80%">
  <p><em>Anatomy-aware Graph Neural Network architecture with dual-graph reasoning</em></p>
</div>

### Anchor Optimization for Mammographic Masses

#### K-means Based Aspect Ratio Selection

Rather than using fixed anchor configurations, we employ data-driven optimization:

**Size Selection (Percentile-based):**
To handle the wide range of mass sizes while maintaining robustness to outliers, we select anchor scales at specific percentiles:
- 10th, 32.5th, 55th, 77.5th, and 90th percentiles of the size distribution

**Aspect Ratio Clustering:**
K-means clustering with k=3 applied to the aspect ratio distribution identifies the most representative ratios:
```python
