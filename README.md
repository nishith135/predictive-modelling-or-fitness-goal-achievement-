# Predictive Modelling of Fitness Goal Achievement using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)

A sophisticated machine learning application designed to predict and recommend personalized fitness and nutrition plans based on user demographics, health metrics, and lifestyle habits. This project implements a **Two-Stage Voting Classifier** architecture and integrates **Explainable AI (XAI)** for transparent decision-making.

---

## 🌟 Key Features

- **Dual-Stage Prediction Engine**:
  - **Stage 1**: Predicts the optimal **Workout Plan Level** (High, Medium, Low, None).
  - **Stage 2**: Uses Stage 1 output alongside user data to predict the ideal **Nutrition Plan Level**.
- **Ensemble Learning**: Utilizes a robust `VotingClassifier` combining Random Forest, Gradient Boosting, and XGBoost.
- **Explainable AI (XAI)**: Integrated **LIME** and **SHAP** explanations to help users understand why specific plans were recommended.
- **Secure Web Portal**: Full-stack Flask application with User Authentication (Signup/Signin) and a responsive dashboard.
- **Interactive Analytics**: Visual representation of data distributions and model performance metrics.

## 🏗️ Project Architecture

The system follows a sequential prediction pipeline:
1. **Input Data**: User enters age, BMI, gender, fitness goals, lifestyle habits, etc.
2. **Preprocessing**: Data cleaning, outlier removal (IQR), and Label Encoding.
3. **Workout Prediction (Stage 1)**: Voting Classifier predicts the workout intensity.
4. **Nutrition Prediction (Stage 2)**: The predicted workout level is fed back into the second model to refine the nutrition plan.
5. **Output**: Personalized recommendation displayed on a premium dashboard.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nishith135/predictive-modelling-or-fitness-goal-achievement-.git
   cd predictive-modelling-or-fitness-goal-achievement-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the Web UI**:
   Open your browser and navigate to `http://127.0.0.1:5000`

---

## 📊 Models & Performance

We evaluated multiple algorithms before settling on the Voting Classifier ensemble:
- **Random Forest**
- **Decision Trees**
- **CatBoost**
- **XGBoost**
- **KNN**
- **Voting Classifier (Final Model)**

*Performance metrics (Accuracy, Precision, Recall, and F1-Score) are detailed in the `Notebook.ipynb` and accessible via the `/graphs` route in the app.*

---

## 🧪 Explainable AI (XAI)

To ensure the models are not just "black boxes", we implemented:
- **LIME**: Local Interpretable Model-agnostic Explanations for individual prediction transparency.
- **SHAP**: SHapley Additive exPlanations for global feature importance and contribution analysis.

---

## 📂 Directory Structure

```text
├── Dataset/             # Raw and processed datasets
├── Models/              # Saved .sav and .pkl model files
├── static/              # CSS, JS, and Image assets
├── templates/           # HTML templates (Flask)
├── app.py               # Main Flask application entry point
├── Notebook.ipynb       # Research, EDA, and Model Training
├── requirements.txt     # Project dependencies
└── signup.db            # SQLite database for user accounts
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model accuracy or add new features:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ by [Nishith](https://github.com/nishith135)*
