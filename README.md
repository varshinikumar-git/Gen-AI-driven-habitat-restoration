# GenAI-Based Habitat Restoration 

## Overview

This project presents a Generative AI-based approach for **habitat restoration and environmental analysis** using multi-modal geospatial data. The system models relationships between environmental factors such as land cover, temperature, and soil carbon, and predicts restoration zones using **Conditional Diffusion Models**.

The pipeline integrates:

* Environmental feature mapping
* Deep learning-based translation (Pix2Pix)
* Conditional diffusion modeling
* Final restoration zone prediction

---

## Key Features

*  Multi-modal environmental analysis
*  Conditional image-to-image translation using Pix2Pix
*  Diffusion-based pattern generation
*  Restoration zone prediction
*  Modular and extensible pipeline

---

## Project Pipeline

1. **Dataset Preparation**

   * Environmental variables are paired:

     * Land Cover → Forest Loss
     * Mean Temperature → Forest Loss
     * Soil Carbon → Water Occurrence

2. **Model Training**

   * Pix2Pix models are trained for each environmental pair

3. **Conditional Diffusion**

   * Diffusion models refine and enhance generated outputs

4. **Final Restoration Mapping**

   * Combined outputs produce final restoration zones

---

## Project Structure

```
genai-habitat-restoration/
│
├── scripts/
│   ├── training/
│   ├── inference/
│   ├── datasets/
│   ├── diffusion/
│   └── restoration/
│
├── outputs/
│   ├── mappings/
│   ├── diffusion/
│   └── final/
│
├── presentation.pptx
│
├── requirements.txt
├── README.md

```

---

## Results

### Environmental Mapping

* Land Cover → Forest Loss
* Temperature → Forest Loss
* Soil Carbon → Water Occurrence

### Conditional Diffusion Outputs

* Enhanced spatial pattern generation

### Final Output

* Restoration Zone Map highlighting potential recovery regions

---

## Installation

```bash
git clone https://github.com/your-username/genai-habitat-restoration.git
cd genai-habitat-restoration
pip install -r requirements.txt
```

---

## Usage

run individual modules:

```bash
python src/training/training_pair1.py
python src/diffusion/conditional_diffusion.py
python src/restoration/final_restoration.py
```

---

## Future Work

* Integration with **Deep Reinforcement Learning (DRL)**
* Species migration and reintroduction modeling
* Real-time environmental monitoring
* Scaling to larger geospatial datasets

---

## Author

Logavarshini K <br>
B.Tech Robotics and Artificial Intelligence
---

## Acknowledgment

This project explores the intersection of **Generative AI and environmental sustainability**, contributing toward intelligent ecosystem restoration.
