# Model-Performance-for-News-Article-Classification

This repository contains a complete machine learning pipeline to classify news articles from the AG News dataset into four categories: World, Sports, Business, and Sci/Tech. Using classic NLP techniques like TF-IDF and Word2Vec, combined with classifiers such as Logistic Regression and SVM, we achieved an accuracy of up to 89%. Ideal for those learning practical NLP or exploring classical text classification approaches.

📌 <b>Project Objective</b>

To build a scalable and efficient ML-based system that automatically categorizes news articles, enabling automated content tagging for news platforms.

📚 <b>Dataset Overview</b>

- **Source**: [AG News Corpus](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- **Classes**: World, Sports, Business, Sci/Tech (evenly balanced)
- **Content**: News **Title** and **Description**
- **Size**: ~7,600 test articles

🧹 <b>Text Preprocessing</b>

We implemented two text vectorization strategies:

- **TF-IDF** (1–2 n-grams, 10,000 features)
- **Word2Vec** (100-dimensional vectors averaged per sentence)

Example:
- **Before**: `"He’s playing at 5pm. Visit https://..."`
- **After**: `"play visit"`

⚙️ <b>Model Building</b>

We experimented with the following ML pipelines:

| Feature Extraction | Classifier           |
|--------------------|----------------------|
| TF-IDF             | Logistic Regression  |
| TF-IDF             | SVM                  |
| Word2Vec           | Logistic Regression  |
| Word2Vec           | SVM                  |

- **Train/Test Split**: 80/20
- **Libraries**: `scikit-learn`, `gensim`

> Note: Used default hyperparameters due to time constraints.

📊 <b>Evaluation Metrics</b>

- **Accuracy**: Up to **89%**
- **Precision, Recall, F1-Score**: Reported for each class
- **Best Model**: TF-IDF + Logistic Regression

<b>Confusion Matrix & Accuracy Comparison</b>

- Logistic Regression slightly outperformed SVM
- TF-IDF outperformed Word2Vec for this dataset

🧠 <b>Key Learnings</b>

- Simpler preprocessing sometimes yields better results
- TF-IDF is effective for short text classification
- Model performance is significantly influenced by preprocessing choices

🚀 <b>Future Work</b>

- Integrate deep learning models and pre-trained embeddings (e.g., BERT)
- Perform hyperparameter tuning
- Use sentence embeddings for richer representation

📎 <b>How to Run</b>

1. Clone the repository
2. Install required libraries
3. Run the notebook: Project3_Group9.ipynb
