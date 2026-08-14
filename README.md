# Fake News Detection 

# 📰 Fake News Detection

> **EDP Project — Week 2**
>
> A Machine Learning and NLP based system for detecting whether a news article is **Fake** or **Real**.



## Week 2 Objective

The main objective of Week 2 was to build the **core Machine Learning pipeline** for the Fake News Detection project.

During this week, the project progressed from raw news datasets to a trained and tested Machine Learning model capable of making predictions on new news articles.


## What I Built This Week

```text       NEWS DATASET
                   │
                   ▼
        ┌─────────────────────┐
        │   Data Loading      │
        │   Fake.csv          │
        │   True.csv          │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Text Preprocessing  │
        │ Cleaning & Combining│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   TF-IDF Vectorizer │
        │ Text → Numerical    │
        │ Features            │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Train / Test Split│
        └──────────┬──────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
 ┌─────────────────┐  ┌────────────────────┐
 │ Logistic        │  │ Passive Aggressive │
 │ Regression      │  │ Classifier         │
 └────────┬────────┘  └─────────┬──────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
            MODEL EVALUATION
                     │
                     ▼
             BEST MODEL SAVED
                     │
                     ▼
              NEWS PREDICTION
```


# Skills Learned

## 1. Python & Data Handling

* Python Programming Basics
* Pandas Library
* Loading CSV Datasets Using Pandas
* Combining Multiple Datasets
* Dataset Shuffling
* Text Data Preprocessing

## 2. Natural Language Processing (NLP)

* Text Cleaning
* Combining News Titles and Article Content
* TF-IDF Vectorization
* Converting Text into Numerical Features

## 3. Machine Learning

* Train-Test Split
* Logistic Regression
* Passive-Aggressive Classifier
* Model Training
* Model Prediction
* Model Comparison

## 4. Model Evaluation

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

## 5. Model Persistence

* Joblib
* Saving Trained Machine Learning Models
* Saving the TF-IDF Vectorizer
* Loading Saved Models for Prediction

## 6. Development Tools & Environment

* Python Virtual Environment
* Git
* GitHub
* Managing Python Dependencies


# Fake News Detection Using Machine Learning

## 1. Dataset

The project uses the **Fake and Real News Dataset** containing **44,898 articles**.

| News Type | Label |   Articles |
| --------- | ----  | ---------  |
| Fake News |     0 |     23,481 |
| Real News |     1 |     21,417 |
| **Total** |     — | **44,898** |


The dataset files are:

dataset/
├── Fake.csv
└── True.csv

## 2. Data Processing

The datasets were loaded, labelled, combined, and cleaned.

python
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

The **title and article content** were cleaned and combined into one text feature.

## 3. TF-IDF

**TF-IDF** was used to convert the cleaned text into numerical features that machine learning models can understand.

text
News Text → Cleaning → TF-IDF → Numerical Features


## 4. Models Used

Two classification models were trained:

| Model                         | Purpose                   |
| ----------------------------- | ------------------------- |
| Logistic Regression           | Main classification model |
| Passive Aggressive Classifier | Comparison model          |

Labels used:

| Label | Meaning   |
| ----  | --------- |
|     0 | Fake News |
|     1 | Real News |

## 5. Results

Logistic Regression achieved **98.57% accuracy** on **8,980 test articles**.

### Confusion Matrix

| Actual / Predicted |  Fake |  Real |
| ------------------ | ----  | ----  |
| **Fake**           | 4,634 |    76 |
| **Real**           |    52 | 4,218 |

### Classification Report

| Class     | Precision | Recall | F1-Score |
| --------- | --------  | -----  | -------  |
| Fake News |      0.99 |   0.98 |     0.99 |
| Real News |      0.98 |   0.99 |     0.99 |

## 6. Model Saving

The trained model and TF-IDF vectorizer were saved using Joblib:

text
model/
├── fake_news_model.pkl
└── tfidf_vectorizer.pkl


These files allow predictions without retraining the model.

## 7. Prediction System

The `predict.py` script loads the saved model and vectorizer and accepts a **news title and content**.

Title + Content
      ↓
Text Cleaning
      ↓
TF-IDF
      ↓
Trained Model
      ↓
Prediction + Confidence


### Output Example

text
Prediction : FAKE NEWS
Confidence : XX.XX%

or

text
Prediction : REAL NEWS
Confidence : XX.XX%

**Result:** The system successfully classifies news articles as Fake or Real with **98.57% accuracy**.

## 📁 Project Structure

```text
Fake_News_Detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│
├── screenshots/
│
├── static/
│
├── templates/
│
├── utils/
│   └── preprocessing.py
│
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── edp_week2_skills_learned.txt
```


---

# 📅 Week 3 — Flask Web Application

## 🎯 Week 3 Objective

The main objective of Week 3 was to integrate the trained Machine Learning model with a **Flask web application** and create an interactive interface for Fake News Detection.

The Machine Learning model developed during Week 2 was connected to a web application so that users can enter news articles and receive a **FAKE NEWS / REAL NEWS** prediction with a confidence score.

---

## 🚀 What I Built This Week

During Week 3, I developed a Flask-based Fake News Detection web application.

The application allows users to:

- Enter a news title
- Enter news content
- Submit the news for prediction
- Receive a Fake/Real prediction
- View the model confidence score
- Clear the entered information

---

## 🔄 Week 3 Application Workflow

```text
              USER
                │
                ▼
       ┌───────────────────┐
       │   News Title      │
       │   News Content    │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │   Flask Web App   │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Text Preprocessing│
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │  TF-IDF Vectorizer│
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │  ML Model         │
       │  Passive          │
       │  Aggressive       │
       │  Classifier       │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │    Prediction     │
       └─────────┬─────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
     FAKE NEWS      REAL NEWS
          │             │
          └──────┬──────┘
                 ▼
       ┌───────────────────┐
       │ Confidence Score  │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │   Web Result      │
       └───────────────────┘
       