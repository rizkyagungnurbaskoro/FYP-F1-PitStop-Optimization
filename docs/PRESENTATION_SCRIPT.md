# Dashboard Presentation Script: Pitwall Strategy Optimization

*This script is designed for a 3-5 minute demonstration of the FYP Pit Stop Prediction project.*

---

## 🏁 1. Introduction (30 seconds)
"Welcome to the **Pitwall Strategy Dashboard**. This project is a decision-support pipeline designed to optimize F1 pit stop timing using a combination of machine learning and human-in-the-loop policy logic. 

We utilize leakage-safe models—specifically XGBoost with GroupKFold splitting by race—to ensure our predictions are based on genuine strategic patterns rather than data memorization."

---

## 🔍 2. The 'Explore' Mode (1 minute)
"Let's move to the **Explore** page. This is where we analyze real-time or historical session data. 

Notice the **Timing Tower** on the left and the **Driver Cards**. The dashboard provides a complete telemetry overview for each driver, including:
- **Tire Health / Degradation Bar**: This isn't just a lap counter; it uses a trained model to estimate real wear percentage.
- **AI Recommendation**: You'll see the system issues a **'BOX BOX'** command when the probability of a pit stop's success crosses our optimized threshold.

The AI now provides **Narrative Explanations**. Instead of showing raw data, it explains *why* it wants to pit—for example, if it detects critical wear approaching the race pace limit or an opening in the pit window."

---

## 🏎️ 3. Iconic Moments Showcase (1 minute)
"One of the most powerful features is our **Showcase** section. We've highlighted high-stakes scenarios like the **2022 Monaco Grand Prix**. 

Here, we specifically analyze Ferrari’s infamous double-stack strategy. 
- **Historical View**: We see what actually happened (History).
- **What-If Analysis**: Our AI re-evaluates the moment. In this case, the model demonstrates how staying out for a few more laps or optimizing the crossover would have gained several seconds in net race time.

This section proves that the model can identify strategic errors in real-world scenarios and suggest better alternatives."

---

## 🛠️ 4. AI Logic & Reliability (30 seconds)
"Behind the scenes, the system is governed by a robust **Policy Engine**. 
- It respects the **Pit Window**, but it’s also smart enough to trigger an **Emergency Call** if it detects tyre wear exceeding 90%. 
- We also have a **Strategy Strength** indicator that shows the confidence level of the AI, allowing engineers to gauge how aggressive the move should be."

---

## 💻 5. Technical Stack & Conclusion (30 seconds)
"Technically, the project is a full-stack solution:
- **Frontend**: Next.js with a high-fidelity 'Broadcast' style UI.
- **Backend**: FastAPI providing low-latency predictions.
- **ML**: XGBoost models calibrated for strategic reliability.

By combining deep data analysis with a professional-grade interface, we’ve created a tool that bridges the gap between raw data science and split-second pitwall decisions. Thank you!"

---
