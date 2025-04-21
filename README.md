# DictaLM2.0-Instruct Encoder Training and Analysis

This guide explains how to set up the environment, install dependencies, and train various models using DictaLM2.0-Instruct as a strong sentence encoder.

---

## Prerequisites

- **Operating System**: Linux
- **GPU**: At least 24 GB of VRAM (RTX 3090 or equivalent recommended)
- **Conda**: Installed and available in your `PATH`

---

## 1. Environment Setup

1. **Create Conda environment** (skip if already created):
   ```bash
   conda env create -f nlp_env.yml
   ```
2. **Activate environment**:
   ```bash
   conda activate papez_env
   ```

---

## 2. Install Additional Dependencies

```bash
pip install torch torchvision torchaudio
pip install flash-attn --no-build-isolation
pip install googletrans
```

---

## 3. Training MNTP and SimCSE on SLURM

> **Note:** Ensure your GPU has sufficient RAM. A Titan RTX may be insufficient; RTX 3090 or higher is recommended.

- **MNTP**:
  ```bash
  sbatch mntp_job.slurm
  ```

- **SimCSE**:
  ```bash
  sbatch simcse_job.slurm
  ```

---

## 4. Training Linear Classifiers

1. **Train classifiers**:
   ```bash
   cd Dicta_LM_experiments/linear_classifier
   python baseline.py                # Decoder-only baseline
   python llm2vec_linear_classifier.py  # LLM2Vec classifier
   ```

---

## 5. Calculating Similarity Scores

1. **Translate data**:
   ```bash
   cd Dicta_LM_experiments/similarity
   sbatch translate.slurm
   ```

2. **Compute similarity** for all models:
   ```bash
   sbatch similarity.slurm
   ```

---

## 6. Directory Structure (Example)

```
├── nlp_env.yml
├── mntp_job.slurm
├── simcse_job.slurm
└── Dicta_LM_experiments
    ├── linear_classifier
    │   ├── data
    │   ├── baseline.py
    │   └── llm2vec_linear_classifier.py
    └── similarity
        ├── translate.slurm
        └── similarity.slurm
```

---

## License

This project is licensed under the MIT License. Feel free to adapt and extend for your own research.

