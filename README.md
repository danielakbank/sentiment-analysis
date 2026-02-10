
# Customer Review Sentiment Analysis – AI & NLP Project

## 📌 Project Overview

This project demonstrates a **complete workflow for sentiment analysis** of IMDB movie reviews using **machine learning and deep learning techniques**.  
The objective is to **classify customer reviews as positive or negative** and generate **actionable business insights** to support data-driven decision-making and improve customer experience.

### Key Highlights

- Uses **50,000 IMDB reviews** for binary sentiment classification.
- Implements **two modelling approaches**:
  1. **Baseline machine learning model** using TF-IDF and Logistic Regression.
  2. **Neural network model** for advanced deep learning classification.
- Evaluates models using **accuracy, precision, recall, F1-score, and confusion matrices**.
- Produces **business-focused insights** from sentiment predictions.
- Models and vectorisers are saved using `joblib` for future deployment.

---

## 🗂️ Project Structure

```
sentiment-analysis-ai-project/
│── data/
│     └── imdb_reviews.csv       # IMDB 50k reviews dataset
│── notebooks/
│     ├── 01_sentiment_analysis_baseline.ipynb   # Baseline ML workflow
│     └── 02_sentiment_analysis_nn.ipynb         # Neural network workflow
│── models/
│     ├── logistic_model.pkl
│     └── tfidf_vectorizer.pkl
│── README.md
└── requirements.txt            # Project dependencies
```

---

## 📝 Notebook Summary

### 1️⃣ Baseline Machine Learning  
**`01_sentiment_analysis_baseline.ipynb`**

- Loads, cleans, and preprocesses the IMDB dataset.
- Converts text data into **TF-IDF features** using unigrams and bigrams.
- Trains a **Logistic Regression** classifier.
- Evaluates performance using standard classification metrics.
- Generates business insights from sentiment predictions.
- Saves trained models and vectorisers for reuse.

**Result:**  
Accuracy ≈ **0.89**, F1-score ≈ **0.89** — a strong and reliable baseline model.

---

### 2️⃣ Neural Network Model  
**`02_sentiment_analysis_nn.ipynb`**

- Reuses preprocessed TF-IDF features.
- Builds a **feedforward neural network** with regularisation and dropout.
- Trains the model using **binary cross-entropy loss** and the **Adam optimiser**.
- Applies early stopping to reduce overfitting.
- Evaluates performance on unseen test data.
- Visualises training and validation accuracy and loss.
- Extracts business insights to support decision-making.

**Result:**  
Accuracy ≈ **0.86**, F1-score ≈ **0.86**, with balanced performance across both classes.

---

## 📊 Model Evaluation Summary

The neural network achieved an overall accuracy of **86.3%** on the test set, with consistent precision and recall across positive and negative reviews.

- Negative class (0): Precision = 0.88, Recall = 0.84, F1 = 0.86  
- Positive class (1): Precision = 0.85, Recall = 0.89, F1 = 0.87  

Training and validation curves indicate mild overfitting after early epochs, highlighting the importance of early stopping and regularisation for improved generalisation.

---

## 💡 Business Insights

Based on the sentiment analysis results, the following business insights were identified:

- **Negative reviews** frequently highlight issues related to **delivery delays, product quality, and customer service**, indicating areas requiring operational improvement.
- **Positive reviews** commonly emphasise **ease of use, value for money, and overall satisfaction**, reflecting key business strengths.
- The neural network enables **automatic identification of negative reviews**, allowing customer support teams to respond more quickly.
- Continuous sentiment monitoring supports **data-driven decision-making** and customer experience optimisation.
- Combining neural networks with traditional machine learning models enables an **ensemble approach**, improving prediction reliability.
- These insights help organisations **prioritise service improvements**, optimise resources, and strengthen long-term customer relationships.

---

## ⚡ Skills & Tools Demonstrated

- **Programming & Data Processing:** Python, Pandas, NumPy  
- **Natural Language Processing:** TF-IDF, n-grams  
- **Machine Learning:** Logistic Regression  
- **Deep Learning:** TensorFlow / Keras  
- **Visualisation:** Matplotlib, Seaborn  
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- **Deployment Preparation:** Model and vectoriser persistence with Joblib  

---

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebooks in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially to reproduce results and insights.
