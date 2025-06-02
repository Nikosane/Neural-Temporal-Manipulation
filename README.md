# 🧠 Neural Temporal Manipulation

Train deep learning models that can **predict, reconstruct, and manipulate time series in non-linear and chaotic ways** — essentially enabling "time bending" simulations for complex systems.

## 🚀 Project Overview

This project focuses on modeling chaotic or highly non-linear time series systems using deep learning. Inspired by phenomena like the butterfly effect and bifurcation theory, we aim to generate multiple plausible "what-if" futures for systems like:

- 🌦️ Weather patterns
- 📉 Financial markets
- 🧬 Evolutionary processes
- ⚙️ Chaotic mechanical systems (e.g., Lorenz Attractor)

## 🎯 Objectives

- Build transformer-based or TCN-based architectures for time series.
- Train with **chaos-aware loss functions** (e.g., divergence penalties, temporal perturbations).
- Predict **alternate futures**, reconstruct **missed paths**, and even **simulate counterfactual timelines**.
- Visualize time-bending predictions and explore stability, divergence, and convergence zones.

## 📊 Datasets

You can use either real-world or synthetic chaotic data:
- ✅ Lorenz system (synthetic)
- ✅ Double pendulum (synthetic)
- ✅ NASDAQ / Bitcoin price data (real)
- ✅ ERA5 weather data (real)

## 🧠 Model Concepts

- **Temporal Transformers**: Attention-based sequence modeling.
- **Time Warping Layers**: Allow skipping, rewinding, or accelerating time steps.
- **Latent Trajectory Divergence**: Enforce chaotic divergence between small input changes.
- **Counterfactual Training**: Predict how systems behave under altered conditions.

## 🧪 Example Use-Case

```python
from models.transformer import TimeBenderTransformer
from data.lorenz import generate_lorenz_data

model = TimeBenderTransformer(...)
X, y = generate_lorenz_data()

model.train(X, y)
model.predict_what_if(X_variant)
```

## 📦 Requirements

- Python 3.10+
- PyTorch or TensorFlow
- NumPy, SciPy, Matplotlib
- tqdm, YAML

Install with:

```bash
pip install -r requirements.txt
```

---

This project is a research prototype and experimental exploration. Use at your own risk.



emg commit 1
emg commit 2
emg commit 3