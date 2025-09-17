# fake-news-detector

A simple **machine learning app** that classifies news articles as **FAKE** or **REAL** using a TF-IDF vectorizer and Logistic Regression.  
The model was trained on a dataset of labeled articles and deployed with **Streamlit** for interactive use.  

🔗 **Live Demo:** [Click here to try it](https://fake-news-detector-jwapifjbnzu4faa4mezhnq.streamlit.app/)  

---

## Features
- Paste any article text and instantly get a **FAKE / REAL** prediction.
- Displays **prediction probabilities** for each class.
- End-to-end ML pipeline:
  - Data preprocessing  
  - TF-IDF vectorization  
  - Logistic Regression model  
  - Evaluation with accuracy, precision, recall, F1  
  - Confusion matrix visualization
- Model is saved and loaded with `joblib` for reuse.

## Tools & Libraries
-Python 3.9+
-scikit-learn
-pandas
-joblib
-matplotlib & seaborn (for visualization)
-Streamlit (for deployment)
