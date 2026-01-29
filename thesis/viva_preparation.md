# Viva Preparation & Learning Guide
**Project:** F1 Pit Stop Optimization & Strategy System
**Goal:** Minimizing performance anxiety by mastering the "Why" and "How" of your codebase.

---

## 1. The "Elevator Pitch" (Start here)
*If they ask: "Tell us about your project in 1 minute."*

> "This project is an **AI-driven Strategy Engineer** for Formula 1.
>
> In F1, races are won or lost on the pit wall. Traditional strategy relies on human intuition or slow Monte Carlo simulations. My system uses **Machine Learning (XGBoost)** to predict the optimal pit stop moment in real-time (<100ms).
>
> It processes telemetry data (Tyre Age, Gaps, Safety Cars), predicts a 'BOX' probability, and filters it through a **Safety Policy Layer** to ensure rational decisions. The result is a dashboard that gives race engineers instant, data-backed recommendations, handling complex scenarios like 'Undercuts' and weather crossovers better than static rule-based systems."

---

## 2. The Architecture (How it works)

Think of your system in **3 Layers**. Visualize data flowing from left to right.

### Layer 1: Data Ingestion (The Senses)
*   **Source:** FastF1 (Python library).
*   **What it does:** Fetches raw telemetry (Lap times, Sector times, Tyre compounds) from the official F1 API.
*   **Key File:** `pitwall_api/app/data.py`
*   **Process:** We don't just take raw numbers; we calculate *deltas* (Gap to leader) and *context* (Is the Pit Window open?).

### Layer 2: The Brain (API & ML Core)
*   **Tech:** Python, FastAPI, XGBoost.
*   **Location:** `pitwall_api/app/model.py`
*   **The Logic:**
    1.  **Training:** The model looks at historical race data (2020-2024).
    2.  **Inference:** For every new lap, it calculates the probability of a pit stop.
    3.  **Policy Layer:** A set of logical rules acts as a "safety net" (e.g., "Don't box if the pit window is closed, even if the model is excited").

### Layer 3: The Interface (The Pit Wall)
*   **Tech:** Next.js (React), TypeScript.
*   **Location:** `pitwall_web/`
*   **Visuals:** A "Dark Mode" dashboard designed for low cognitive load. It includes a **3D Monaco Track** (`Monaco3DTrack.tsx`) to visualize car positions spatially.

---

## 3. Deep Dive: The Machine Learning Core
*This is the most likely area for technical questions.*

### **The Model: XGBoost Classifier**
*   **Why XGBoost?** (Question: "Why didn't you use Deep Learning / LSTM?")
    *   **Answer:** "F1 data is **tabular** and structured (Lap number, Tyre Age, Gaps). Deep Learning (LSTMs) is great for unstructured data like voice or video, but for tabular data, **Gradient Boosted Trees (XGBoost)** consistently outperform Neural Networks.
    *   **Reason 2:** **Non-linearity**. Tyre degradation isn't a straight line; it falls off a "cliff". Trees handle these sudden "cutoff points" naturally.
    *   **Reason 3:** **Interpretability**. We need to know *why* the model is saying BOX. XGBoost allows us to calculate Feature Importance."

### **The "Safety Policy" (Your Secret Weapon)**
*   **Concept:** Pure ML models can be "jittery" (flicking between 49% and 51% probability). You added a deterministic layer on top.
*   **Code Location:** `pitwall_api/app/model.py` -> `demo_policy_decision()` function.
*   **How it works:**
    *   It takes the **Raw Probability** from XGBoost.
    *   It checks **Hard Constraints** (e.g., "Is the Pit Window Open?").
    *   It adjusts the threshold based on context (e.g., "If there is a Safety Car (SC), lower the threshold to Box because a pit stop is cheaper").
    *   **Viva Gold:** This proves you aren't just blindly trusting AI; you are building a **Robust System**.

### **Validation Strategy**
*   **Method:** **5-Fold Group Cross-Validation**.
*   **Crucial Detail:** You grouped by `Race_ID`.
*   **Why?** "To prevent **Data Leakage**. If I train on Laps 1-10 of the Monaco GP and test on Laps 11-20 of the *same* Monaco GP, the model is cheating (it knows the weather/conditions of that specific day). By grouping by Race, I force the model to predict on a *brand new race* it has never seen, which mimics real life."

---

## 4. Key Files to Know (For code walkthroughs)

If they ask you to "Show us the code"...

1.  **`pitwall_api/app/model.py`**:
    *   **Lines ~303 (`demo_policy_decision`)**: The decision logic. Show this to prove you understand "Hybrid AI" (Model + Rules).
    *   **Lines ~546 (`_make_pipeline`)**: Shows the XGBoost classifier.

2.  **`pitwall_web/components/Monaco3DTrack.tsx`**:
    *   Shows the 3D visualization. Mention this uses **WebGL/Three.js** (via React Three Fiber) to render the track geometry.

3.  **`thesis/README.md`**:
    *   Your own documentation. It contains the "Findings" list (e.g., "The Outlap Anomaly").

---

## 5. Viva Q&A "Cheat Sheet"
*Anticipating the stuttering points.*

**Q: Is this "Artificial Intelligence" or just "Machine Learning"?**
> **A:** "It uses **Machine Learning** (XGBoost) as the prediction engine, but the overall system acts as an **AI Agent**. The 'Agent' includes the **Safety Policy** logic that reasons about the environment (Safety Cars, Pit Windows) to make the final decision. So, it's an AI system powered by an ML core."

**Q: How do you handle the fact that pit stops are rare? (Class Imbalance)**
> **A:** "Good question. Pit stops only happen ~2% of the time. I used **scale_pos_weight** in XGBoost to heavily penalize missing a 'BOX' call. This forces the model to pay more attention to the minority class (the pit stops)."

**Q: Why separate the Backend (FastAPI) and Frontend (Next.js)?**
> **A:** "**Decoupling**. The heavy number-crunching (Data Science) happens in Python, which is best for ML. The UI rendering happens in React, which is best for dashboards. If the UI freezes, the strategy engine keeps running. If the engine is slow, the UI stays responsive."

**Q: What is the "latency" of your system?**
> **A:** "The inference time is **<100ms** per lap. In F1, you have about 90 seconds per lap to make a decision. My system provides a recommendation in a fraction of a second, leaving plenty of time for human review."

**Q: How did you validate that your model works?**
> **A:** "I used historical data from 2020-2024. I tested the model on 'unseen' races (Group Cross-Validation). I specifically looked at whether it could predict known strategic masterstrokes, like potential undercuts, and compared its accuracy against simple rule-based baselines."

---

## 7. Feature Engineering Details
*If they ask: "What features did you engineer?" or "How did you process the data?"*

> "I didn't just dump raw data into XGBoost. I created specific **Derived Features** to capture race dynamics:"
>
> **1. Lag Features (`_prev`)**
> *   **What:** I used the *previous* lap's data to predict the *current* decision (e.g., `gap_to_leader_prev`).
> *   **Why:** **Causality & Realism**. In real-time, I don't know the result of Lap 50 until it finishes. To predict the strategy for Lap 50, I must strictly use data available at the end of Lap 49.
>
> **2. Rate of Change (Deltas)**
> *   **What:** Calculated `GapDelta` (Is the gap shrinking or growing?) and `WearPerLap` (Degradation Rate).
> *   **Why:** The *absolute* gap (2 seconds) matters less than the *trend* (was it 3s last lap? Then I'm gaining).
>
> **3. Interaction Features**
> *   **What:** `PitWindowProgress` (Pit Window Status × Race Progress).
> *   **Why:** A "Pit Window Open" signal means something very different on Lap 10 vs Lap 50. Combining them helps the tree find these context-specific rules.
>
> **4. Cyclical Encoding**
> *   **What:** Converted Wind Direction (0-360°) into `WindDirSin` and `WindDirCos`.
> *   **Why:** 0° and 360° are the same. A standard model sees them as far apart. Cyclical encoding fixes this math.

---

## 8. Dataset Comparison (Ref vs. My Data)
*If they ask: "What is the difference between the Reference Data and Your Data?"*

> "The **Reference Dataset** was a legacy baseline (2014-2019) used early in development. I built **My Own Dataset** for three reasons:"
>
> 1.  **Recency (2018-2024)**: F1 cars changed completely in 2022 (Ground Effect era). The old data was obsolete. My dataset includes the modern era.
> 2.  **Granularity**: My dataset is **4x larger (35MB vs 8MB)**. I extracted richer telemetry (Tyre Wear %, Track Temp) directly from the FastF1 API, whereas the reference data was just basic lap times.
> 3.  **Feature Alignment**: My dataset is pre-processed with the `_prev` (Lag) structure native to my pipeline, ensuring zero causal leakage during training."

---

## 9. Data Leakage Prevention (The "Anti-Cheating" Mechanism)
*If they ask: "How did you ensure your model isn't cheating?" or "What is Data Leakage?"*

> "Data Leakage happens when the model accidentally sees 'future data' during training. I prevented this in **Two Ways**:"
>
> **1. Chronological Leakage (`_prev` Features)**
> *   **The Problem:** Using 'Lap Time' from the current lap (Lap 50) to predict if I should pit on Lap 50 is cheating, because I don't know the lap time until the lap is *finished*.
> *   **The Fix:** I strictly use **Lag Features** (e.g., `gap_to_leader_prev`). I force the model to look only at data available at the moment of decision (End of Lap 49).
>
> **2. Group Leakage (Race-Wise Split)**
> *   **The Problem:** If I randomly split laps, I might train on Lap 1 of Monaco 2024 and test on Lap 20 of Monaco 2024. The model would memorize the weather/conditions of that specific day.
> *   **The Fix:** I used **GroupKFold Validation**, grouping by **Race ID**.
> *   **Result:** If 'Monaco 2024' is in the Test Set, the model has *zero* access to any lap from that race during training. It must generalize completely."

---

## 10. Meaning of Key Terms (Vocabulary List)
*Use this script for the "Results & Validation" screen (the dark dashboard with S1/S2 scores).*

> "This screen summarizes the quantitative proof that the system works.
>
> **1. The Metric (F1 Score vs Accuracy):**
> You'll see I prioritized the **F1 Score (0.812)** as my primary metric. In Formula 1, pit stops are 'rare events' (occurring <2% of the time). If I used simple Accuracy, the model could just say 'Stay Out' every single lap and still get 98% accuracy, but it would be useless. The F1 score forces the model to balance Precision (Trust) and Recall (Actual pit stops).
>
> **2. The 'Leakage-Safe' Validation:**
> The card labels mention 'Leakage-Safe Split' or '80/20 Group Split'. This is crucial. I didn't just split the data randomly. I split it by **Race ID**. This means the model never saw the 'future' of the race it was currently predicting. This simulates a real-life scenario where we go into a race weekend completely blind.
>
> **3. The Comparison (REFTECH vs MYMETHOD):**
> On the left, you see the 'Timing Tower'.
> *   **RefTech (Gray)**: This is the baseline—essentially a standard rule-based approach.
> *   **MyMethod (Orange)**: This is my XGBoost Hybrid Agent.
> The Green delta **(+0.030)** proves that my Machine Learning approach outperforms the static rules.
>
> **4. Dual Validation (Status Card):**
> Finally, the 'System Status' shows **Dual Validation**. I ran the tests twice: once on an 80/20 split and again on a separate 70/30 Holdout set. The fact that the improvement is consistent across both tests proves the model is robust and not just 'lucky'."
