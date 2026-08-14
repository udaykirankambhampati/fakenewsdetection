# 📰 Fake News Detection — Week 3

## 🌐 Flask Web Application & Machine Learning Integration

> **EDP Project — Week 3**

This week focuses on integrating the trained Machine Learning model with a **Flask web application** to create an interactive Fake News Detection system.

The application allows users to enter a news headline and article content and receive an ML-based prediction indicating whether the news is **FAKE NEWS** or **REAL NEWS**, along with the model confidence score.

---

## 🎯 Week 3 Objectives

- Integrate the trained ML model with Flask
- Load the trained model and TF-IDF vectorizer
- Accept news input through a web interface
- Preprocess user-provided news text
- Generate predictions using the trained model
- Display prediction confidence
- Build a clean and responsive user interface
- Connect frontend, backend and ML components

---

## 🧠 System Overview

```text
                 👤 USER
                    │
                    ▼
        ┌─────────────────────────┐
        │     Flask Web UI        │
        │                         │
        │  News Title             │
        │  News Content           │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Text Preprocessing    │
        │                         │
        │   Cleaning & Combining  │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │    TF-IDF Vectorizer    │
        │                         │
        │   Text → Numerical      │
        │       Features          │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Trained ML Model      │
        │                         │
        │ Passive Aggressive /    │
        │ Logistic Regression     │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │    Prediction Result    │
        │                         │
        │   🔴 FAKE NEWS          │
        │   🟢 REAL NEWS          │
        │                         │
        │   Confidence Score      │
        └─────────────────────────┘