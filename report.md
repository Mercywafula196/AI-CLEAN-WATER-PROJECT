# AI for Clean Water (SDG 6)

### 1. Project Title
**AI-Powered Water Quality Prediction System**

---

### 2. SDG Focus
**Goal:** SDG 6 — Clean Water and Sanitation  
**Problem:**  
Access to safe drinking water remains a challenge in many regions. Traditional laboratory testing is expensive, slow, and unavailable in remote areas. Communities need affordable, AI-driven tools to predict water safety using easily measurable chemical properties.

---

### 3. Technical Solution & AI Approach
The solution uses a **machine learning classification model** trained on public datasets (e.g., Kaggle’s *Water Potability* dataset).  

- **Automation:** Data cleaning, feature scaling, and model training are fully automated using Python pipelines.  
- **Testing:** Automated unit tests (`test_model.py`) verify the model loads correctly, produces valid predictions, and outputs probabilities within logical bounds.  
- **Scalability:** Code is modular and containerized via Docker, enabling deployment on local machines or cloud environments.  
- **Model:** A `RandomForestClassifier` predicts whether water is *potable (safe)* or *not potable* based on attributes like pH, hardness, solids, and chloramines.  
- **Interface:** A Streamlit web app (`app.py`) allows users to manually input values or upload a CSV for batch prediction.

---

### 4. Tools & Frameworks
| Category | Tools Used |
|-----------|-------------|
| Programming Language | Python 3.11 |
| AI/ML Libraries | scikit-learn, pandas, numpy, joblib |
| App Framework | Streamlit |
| Testing | pytest |
| Deployment | Docker |
| Version Control | Git/GitHub |

---

### 5. Deliverables
| Deliverable | Description |
|--------------|--------------|
| `train.py` | Trains and exports the AI model |
| `app.py` | Streamlit web interface |
| `test_model.py` | Unit tests for reliability |
| `requirements.txt` | Dependency list for reproducibility |
| `Dockerfile` | Enables containerized deployment |
| `report.md` | Project documentation and ethical reflection |

---

### 6. Ethical & Sustainability Considerations
- **Bias Mitigation:** The dataset was reviewed to ensure balanced representation of safe and unsafe samples.  
- **Transparency:** The model uses interpretable metrics (e.g., feature importance) to explain predictions.  
- **Environmental Impact:** The Random Forest model is lightweight and optimized to run on low-resource devices.  
- **Accessibility:** The app can run on free platforms like Google Colab or any laptop, making it practical for developing regions.  
- **Data Privacy:** No personal or user-sensitive data is collected.

---

### 7. Expected Impact
- Enables communities to test water quality instantly.  
- Reduces dependency on expensive lab facilities.  
- Promotes environmental health and sustainable water resource management.  
- Supports SDG 6 targets for **universal access to safe and affordable drinking water**.

---

### 8. Software Engineering Concepts Applied
| Concept | Application |
|----------|--------------|
| **Automation** | Data preprocessing and training via pipelines |
| **Testing** | Unit tests using pytest ensure model reliability |
| **CI/CD Readiness** | Docker + requirements.txt enable easy redeployment |
| **Version Control** | Git ensures collaborative and trackable changes |
| **Ethical AI Design** | Focus on fairness, transparency, and sustainability |

---

### 9. Reflection
This project demonstrates how AI and software engineering can address real-world sustainability challenges.  
By combining automation, ethical design, and testing, the **AI for Clean Water** system shows that technology can contribute directly to achieving the **UN Sustainable Development Goals**.  

Future improvements include integrating IoT sensors for real-time water data and using deep learning for anomaly detection.

---

### 🧠 *Prepared by:*  
**Mercy Wafula**  
AI for Software Engineering — SDG 6 Project
