# MedPRO: Medical AI Diagnostic Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.x-cyan.svg)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)

A full-stack, AI-powered diagnostic platform designed to bridge the gap between clinical data and patient understanding. MedPRO provides predictive screening for **Diabetes**, **Heart Disease**, and **Kidney Disease**, complete with explainable AI (XAI) visualizations and an intelligent timeline dashboard for patient monitoring.

## 🌟 Key Features

* **Multi-Disease Prediction API**: Neural Networks rigorously trained for accurate diagnostics across three critical endpoints: Heart, Kidney, and Diabetes.
* **Explainable AI (SHAP Waterfall)**: Provides instant visual insights into *why* the AI made its decision, mapping precise risk factors (e.g., Blood Pressure, Glucose) against a clinical baseline.
* **Doctor & Patient Portals**: 
  * **Doctor View**: Secure workspace for generating diagnostic insight, interacting with "What-If" clinical feature simulations, and automated treatment roadmap structuring. 
  * **Patient View**: A beautifully animated patient timeline ("Colorful Clinical" theme) equipped with follow-up reminders.
* **Advanced RAG (Retrieval-Augmented Generation)**: In-house vector database lookup built on `faiss` and `SentenceTransformers` to pull medically sound diet/treatment advice directly related to the inferred risk level.
* **Voice-Activated Accessibility**: Integrated Web Speech API letting patients securely fetch their health dashboard hands-free.

## 📁 System Architecture

| Directory/File | Description |
| --- | --- |
| `server.py` | Primary Flask backend orchestrating inference pipelines and JWT Auth endpoints. |
| `database.py` | SQLAlchemy definitions for `Patient_Profiles`, `Visits`, and strict `Audit_Log` tracking. |
| `ai_env/` | Sandboxed Python environment constructed by the robust bat script for dependency safety. |
| `Datasets/` | Preprocessed clinical training datasets and their accompanying `.keras` neural networks. |
| `celery_worker.py` | Asynchronous task queue routing for heavy background workloads (Requires Redis). |
| `medical-frontend/`| React codebase featuring dynamic routing, Framer Motion animations, and Tailwind styling. |

## 🚀 Quickstart Guide

### 1. Booting the Backend (Windows)
The repository contains a custom Failsafe Clean Sandbox script that automatically sets up your virtual environment, sanitizes caches, and wires the required dependencies.

Simply double-click or run:
```bat
run_backend.bat
```
This deploys the `Flask` application locally on port `5000`.

### 2. Launching the Frontend
In a new terminal, navigate to the React directory to spin up the UI.
```bash
cd medical-frontend
npm install
npm start
```

### 3. Demo Access
You can experience both interfaces without modifying the authentication system manually.
* Click **Demo Login** on the login screen to instantiate local doctor and patient profiles. 

## 🩺 Tech Stack
* **Machine Learning**: TensorFlow / Keras, Scikit-Learn (MinMaxScaler pipelines), SHAP (Internal fast-permutation logic)
* **Backend**: Python, Flask, SQLAlchemy, Celery 
* **Database**: SQLite (built-in relational schema), FAISS (Knowledge Index)
* **Frontend**: React, Tailwind CSS, Framer Motion, Axios, Recharts/Chart.js

## 🔒 Security
- **Strict Role-based Authorization**: JSON Web Tokens ensure zero unauthorized access overlap between patient timelines and doctor execution pipelines.
- **Audit Logging**: Every single `ACCESS_PATIENT_RECORD` or `ACCESS_RISK_TREND` hit is uniquely time-stamped and mapped structurally to the interrogating Doctor ID.

## 📝 License
This project is made open-source under the MIT License.
