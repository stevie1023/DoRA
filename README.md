# DoRA
This repository contains a minimal, implementable code base for the NeurIPS 2025 paper: **"[Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](https://arxiv.org/abs/2504.05831)"**.

<img width="526" height="295" alt="image" src="https://github.com/user-attachments/assets/ba983227-52e7-4434-b565-87ed7f115171" />


## 🏗️ Code Base & Environment

This code base is built upon the **RRHF (Rank Responses to Align Human Feedback)** framework.

Please refer to the original [RRHF paper/repository](https://github.com/GanjinZero/RRHF) for instructions on:
- Setting up the Python environment.
- Generating the initial training data.

## 📂 Data Preparation

To support the robust optimization methods proposed in the paper, the training data requires a specific field.

We include a **`"probs"`** key in the data file. This value is provided by learned classifiers to indicate the likelihood that the response belongs to the target distribution.

**Example JSON Structure:**
```json
{
  "prompt": "Example prompt...",
  "responses": ["Example response 1...", "Example response 2...", "Example response 3...", "Example response 4..."]
  "scores": [0.85,0.6,0.2,1.5],
  "probs": [0.95,0.5,0.1,0.8]
  "...": "other fields",
}
```

## 🚀 Supported Alignment Objectives

We include implementations w.r.t. DoRA on the following alignment objectives for quick and easy use:

- **RRHF**
- **LIRE**
- **DPO**: Direct Preference Optimization, supporting:
  - **Bradley-Terry (BT)** model
  - **Plackett-Luce (PL)** model

## 📄 Citation

If you find this code or paper useful in your research, please cite:

```bibtex
@article{zhu2025leveraging,
  title={Leveraging Robust Optimization for LLM Alignment under Distribution Shifts},
  author={Zhu, Mingye and Liu, Yi and Fu, Zheren and Zhang, Yongdong and Mao, Zhendong},
  journal={arXiv preprint arXiv:2504.05831},
  year={2025}
}
