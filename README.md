AI Cyber Radar (AI-Threat-Classifier).

In this repository, there is my final year practical project of UA92: a small and local web system that categorizes cyber/security log events in Low / Medium / High severity and justifies why the model made this choice.

This is aimed at modeling a situation in which a SOC analyst would understand how to triage alerts more quickly, with explainable artificial intelligence (XAI) instead of a black-box response.

What it does
Adds a log / event (copy text or insert a file)
Breaks it down into a format.
Reverts an educated ML model to forecast severity (Low / Medium / High).
Gives a description of the choice in SHAP (feature contribution style reasoning) style.
Optional: a sample/live agent script is provided to produce some sample/live events to do some demos.

Why I built it
SOC teams frequently have large volumes of alerts and frequent false positives. It is a prototype demonstrating that machine learning + explainability can assist in the process of alert triage and decision-making (human-in-the-loop) and not with the analyst.

Tech stack
Python
Flask (web app)
pandas / numpy (data processing)
scikit-learn (model training)
joblib (saving/loading the model)
SHAP (explainability)
matplotlib (visual outputs)

Note: LIME has been investigated whilst developing, although it adds heavy build dependencies to Windows through scikit-image. The last project is dedicated to SHAP in terms of stability and reproducibility.

Run (Windows / PowerShell) How to operate it.

1) Assess and establish a virtual environment.
powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If activation is blocked:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Train the model
python .\train_unsw_model.py
Run the web app
python .\app.py


Project files (overview)
app.py — Flask app entry point
risk_engine.py — parsing + scoring/classification logic
train_unsw_model.py — training pipeline
models/ — saved model output (.joblib)
templates/ — HTML template(s)
static/ — CSS and static assets
outputs/ — saved explanation images / reports

Data
This project uses a public intrusion detection dataset (UNSW-NB15) for training/evaluation.
If the dataset is not included in this repo (file size limits), it should be downloaded separately and placed in the data/ folder as described in the report/appendix.

Future improvements
Add user authentication + audit logging
Improve feature engineering and threshold calibration
Add more datasets and stronger evaluation (cross-validation, drift checks)
Containerise the app (Docker) for easier deployment

Author
Muhammad Shayan
