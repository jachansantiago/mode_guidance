# MGD³: Mode-Guided Dataset Distillation using Diffusion Models

[![Project Page](https://img.shields.io/badge/Project%20Page-Mode--Guided%20Distillation-blue)](https://jachansantiago.com/mode-guided-distillation/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**📌 ICML 2025 Spotlight (Top 2.6%)**

---

## 🧠 Introduction

**MGD³** presents a novel approach to dataset distillation by leveraging pre-trained diffusion models without the need for fine-tuning. The method enhances diversity and representativeness in synthetic datasets through a three-stage process:

1. **Mode Discovery**: Identifies distinct data modes within each class.
2. **Mode Guidance**: Steers the diffusion process toward the discovered modes.
3. **Stop Guidance**: Transitions to unguided diffusion to prevent artifacts.

This approach ensures representative and diverse synthetic datasets suitable for training models.

For more details, visualizations, and supplementary materials, visit the [Project Page](https://jachansantiago.com/mode-guided-distillation/).

## 🚀 Highlights
- **No Fine-Tuning Required**: Utilizes pre-trained diffusion models directly.
- **Enhanced Diversity**: Achieves superior intra-class diversity compared to existing methods.
- **Scalability**: Demonstrates effectiveness on large-scale datasets like ImageNet-1K.

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jachansantiago/mode_guidance.git
   cd mode_guidance
   ```
2. Set up the environment:

   ```bash
   conda create -n modeguidance python=3.8
   conda activate modeguidance
   pip install -r requirements.txt
   ```
### 📊 Usage
To run the code on the ImageNette dataset:
  ```bash
  ./scripts/nette.sh
   ```

