# News Document Classifier

A machine learning-powered web application that classifies news articles and documents into predefined categories using multiple classification algorithms.

---

## Overview

News Document Classifier is a Python-based application built with Streamlit that uses machine learning models to automatically categorize news articles into categories such as Business, Sports, Technology, Education, and Entertainment. The application supports multiple classification algorithms and allows users to compare their performance.

---

## Features

### Multiple Input Methods

* Upload text files (`.txt`)
* Upload PDF documents (`.pdf`)
* Paste text directly into the application

### Multiple ML Models

* **Logistic Regression** - Fast and efficient linear classifier
* **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
* **Support Vector Machine (SVM)** - Powerful non-linear classifier

### Model Comparison

* Compare predictions from all three models on the same input
* View accuracy metrics for each model on the test set
* Interactive visualization of model performance

### News Categories

* Business
* Sports
* Technology
* Education
* Entertainment

---

## Website

The application is deployed on Streamlit Cloud:

[News Document Classifier App](https://anushi2511-news-document-classifier-app-ktyq6v.streamlit.app/)

---

## Installation

### Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

### Setup

#### 1. Clone the repository

```bash
git clone https://github.com/anushi2511/News-Document-Classifier.git
cd News-Document-Classifier
```

#### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Training Models

Before using the application, train the models on your dataset:

```bash
python train_model.py
```

This script will:

* Load data from CSV files in the `data/` directory
* Clean and preprocess the text
* Train all three models with TF-IDF vectorization
* Save trained models to the `saved_models/` directory

### Expected Data Structure

```text
data/
├── business_data.csv
├── sports_data.csv
├── technology_data.csv
├── education_data.csv
└── entertainment_data.csv
```

Each CSV should have the following columns:

* `headlines`
* `description`
* `content`

---

### Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501/
```

---

### Compare Models

Run the comparison script to generate accuracy metrics:

```bash
python compareModels.py
```

This generates `model_results.json` with accuracy scores.

Use the **"Compare All Models"** button in the app to:

* Get predictions from all three models
* View side-by-side comparison
* See accuracy metrics visualization

---

## Project Structure

```text
News-Document-Classifier/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── compareModels.py                # Model comparison script
├── requirements.txt                # Python dependencies
├── model_results.json              # Model accuracy metrics
├── data/                           # Training data directory
├── models/                         # Model modules
├── saved_models/                   # Trained model files (.pkl)
└── README.md                       # Project documentation
```

---

## Dependencies

| Package      | Version | Purpose                                       |
| ------------ | ------- | --------------------------------------------- |
| scikit-learn | ≥1.3.0  | Machine learning algorithms and preprocessing |
| pandas       | ≥2.0.0  | Data manipulation and analysis                |
| numpy        | ≥1.24.0 | Numerical computing                           |
| streamlit    | ≥1.52.0 | Web application framework                     |
| PyPDF2       | ≥3.0.0  | PDF file processing                           |
| torch        | ≥2.0.0  | Deep learning (optional)                      |
| matplotlib   | ≥3.7.0  | Data visualization                            |
| seaborn      | ≥0.12.0 | Statistical visualization                     |

---

## How It Works

### Text Preprocessing

The `clean_text()` function in `train_model.py` performs:

* Lowercase conversion
* HTML tag removal
* URL removal
* Special character removal
* Whitespace normalization

### Feature Extraction

* **TF-IDF Vectorization** converts text into numerical features
* Uses **unigrams and bigrams** (`1-2` word combinations)
* Limits features to **20,000** for efficiency

### Classification Pipeline

Each model uses a Scikit-learn pipeline:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf', <Classifier>)
])
```

---

## Model Performance

The application displays:

* **Test Set Accuracy** on holdout test data (20% dataset split)
* **Predictions** from each model
* **Comparison Charts** for model accuracies

