# 🎬 AI Movie Review Sentiment Analysis

An AI-powered web application that predicts whether a movie review expresses **Positive 😊** or **Negative 😞** sentiment using **Natural Language Processing (NLP)** and **Machine Learning**.

Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

This project analyzes movie reviews and predicts their sentiment using a Machine Learning model.

The application converts text into numerical features using the **TF-IDF Vectorizer** and classifies reviews using a **Logistic Regression** model.

---

## 🚀 Features

- 🎬 Predicts Positive or Negative movie reviews
- 📝 User-friendly Streamlit interface
- 🤖 Machine Learning based prediction
- 📊 Confidence score display
- ⚡ Instant prediction
- 📱 Responsive web application

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📂 Project Structure

```
Sentiment-Analysis-App/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── sentiment_dataset.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│   └── Sentiment_Analysis.ipynb
│
└── images/
    ├── homepage.png
    └── prediction.png
```

---

## 🧠 Machine Learning Workflow

```
Movie Review
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression Model
      │
      ▼
Sentiment Prediction
```

---

## 📸 Application Preview

### 🏠 Home Page

![Home Page](images/homepage.png)

---

### 🎯 Prediction Result

![Prediction Result](images/prediction.png)

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Sentiment-Analysis-App.git
```

### Navigate to the project

```bash
cd Sentiment-Analysis-App
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Information

| Model | Logistic Regression |
|-------|----------------------|
| NLP Technique | TF-IDF Vectorizer |
| Problem Type | Binary Classification |

---

## 🎯 Future Improvements

- Multi-language sentiment analysis
- Deep Learning (LSTM/BERT)
- Emoji sentiment detection
- Voice review analysis
- Review history
- Model comparison

---

## 👩‍💻 Author

**Sonali **

B.Tech Computer Science (Data Science)

--