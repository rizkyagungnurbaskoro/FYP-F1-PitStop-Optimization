# Pitwall Strategy Optimization & Pit Stop Prediction System

![F1 Strategy Banner](https://img.shields.io/badge/F1-Strategy_Optimization-blue?style=for-the-badge&logo=formula1)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge&logo=next.js)

## 🏎️ Overview
This repository contains a full-stack decision-support pipeline designed to optimize Formula 1 pit stop timing. By distilling the **"Rational Core"** of racing strategy, the system provides real-time "BOX BOX" recommendations backed by explainable AI (XAI).

The project addresses the extreme cognitive load placed on race engineers [4][19] by providing a high-fidelity dashboard that transforms raw telemetry into actionable strategic insights [1][3].

---

## 🚀 Key Features
- **Predictive Modeling**: High-accuracy XGBoost classifier trained on historical 2021-2024 telemetry data [7].
- **Confident Learning**: Automated noise pruning using Cleanlab to filter out suboptimal historical human strategic calls [9].
- **Explainable AI (XAI)**: SHAP-based justifications for every recommendation, ensuring transparency for engineers [6].
- **Real-Time Dashboard**: A "Broadcast Style" UI built with Next.js, featuring dynamic tire degradation bars and live telemetry interpolation [18].
- **Showcase Mode**: "What-If" analysis of iconic moments, such as the 2022 Monaco Grand Prix strategic errors [5].

---

## 🧠 Technical Highlights
### 1. Data-Centric ML Pipeline
- **Bias Mitigation**: Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the class imbalance of "rare event" pit stops [17].
- **Leakage-Safe Evaluation**: Utilizes **GroupKFold** splitting by race event to ensure models generalize to unseen tracks [2].
- **Feature Engineering**: Incorporates non-linear metrics such as **Tire Age**, **Gap to Leader**, and **Safety Car status** [13][15].

### 2. Strategy Logic & Game Theory
- **Nash Equilibrium**: Models the "Undercut" as a multi-agent game, identifying optimal response windows [10].
- **Safety Car Optimization**: Shifts decision thresholds during VSC/SC events to capitalize on "cheap" pit stop utility [15][16].
- **Rationality Filter**: The AI ignores stochastic human "panic calls," acting as a consistency stabilizer [19].

### 3. System Architecture
- **Backend (FastAPI)**: Low-latency inference engine (<100ms) optimized for edge-native deployment [18].
- **Frontend (Next.js)**: High-performance, high-contrast mission control interface designed to reduce visual fatigue [11][12].

---

## 📂 Repository Structure
```text
├── pitwall_api/       # FastAPI Backend (ML Inference)
├── pitwall_web/       # Next.js Frontend (Dashboard)
├── experiments/       # ML Training Scripts & Evaluation (GroupKFold)
├── data/              # Telemetry Datasets (FastF1 & Custom)
├── thesis/            # Academic findings and README documentation
├── visualization/     # Plotting utilities for result analysis
└── results/           # Metrics JSONs and performance plots
```

---

## 🛠️ Getting Started
### Prerequisites
- Python 3.9+
- Node.js 18+

### Setup
1. **Model Training**:
   ```bash
   python -m experiments.exp_run_all
   ```
2. **Launch API**:
   ```bash
   cd pitwall_api
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
3. **Launch Dashboard**:
   ```bash
   cd pitwall_web
   npm install
   npm run dev
   ```

---

## 📚 Academic References & Bibliography
[1] Smith et al., "Analytics in Professional Sports," 2018.  
[2] "From data to podium: a machine learning model for predicting Formula 1 pit stop timing," UNL Thesis, 2023.  
[3] "Predictive Model for Pitstop Strategy in Formula 1 using Ensemble Learning," NCIRL, 2022.  
[4] Bekker et al., "Competitive Pit Stop Strategy in Formula 1," ResearchGate.  
[5] Ferrari Monaco 2022 Strategic Analysis, various industry reports.  
[6] Lundberg et al., "A Unified Approach to Interpreting Model Predictions," NIPS 2017 (SHAP).  
[7] Chen et al., "XGBoost: A Scalable Tree Boosting System," KDD 2016.  
[8] Low et al., "Monte Carlo Simulations for Motorsport Strategy," 2019.  
[9] Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR 2021.  
[10] Nash, J., "Non-Cooperative Games," Annals of Mathematics, 1951.  
[11] "HCI and Visual Fatigue in Dark Mode Interfaces," MDPI, 2021.  
[12] "XAI in Sports Analytics: The Case for Interpretability," 2020.  
[13] Monolith AI / Jota Sport, "Self-learning models for racing strategy," 2022.  
[14] "Deep learning for tire energy forecasting in Formula 1," Arxiv, 2023.  
[15] Heine and Thraves, "Stochastic dynamic programming for motorsport strategy," 2023.  
[16] Jin and Yoo, "The influence of the Safety Car on Formula 1 racing," 2017.  
[17] "SMOTE for predicting rare sporting events in analytics," ResearchGate, 2021.  
[18] "Real-time performance simulations in precision sports," 2021.  
[19] "Mental fatigue and decision errors in high-speed environments," Frontiers in Psychology, 2022.  
[20] "Perceptual-cognitive expertise and trust in AI in motorsports," NIH, 2019.
