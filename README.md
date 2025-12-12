# 📈 Stock Price Prediction using Twitter Sentiment (Hybrid RoBERTa-GRU)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow?style=for-the-badge&logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/Status-Research_Complete-green?style=for-the-badge)

A comparative study and implementation of **Hybrid Neural Architectures** for stock market trend prediction. This project proposes a novel **RoBERTa + GRU** hybrid model that analyzes Twitter sentiment to forecast stock price movements, achieving state-of-the-art performance compared to baseline BERT models.

> **Research Highlight:** The proposed Hybrid GRU model achieved a classification accuracy of **60.8%**, outperforming standard BERT (56.1%) and RoBERTa (57.3%) baselines on the test dataset.

## 🚀 Key Features
* **Hybrid Architecture:** Combines **RoBERTa** (for contextual embedding) with a **Bi-Directional GRU** (for sequential modeling) to capture long-range dependencies in financial text.
* **Large-Scale Analysis:** Trained on a dataset of **300,000+ tweets** (2015–2020) covering major tech stocks (Apple, Google, Microsoft, Tesla).
* **Robust Preprocessing:** Implements custom regex pipelines to handle noisy social media text (removing tickers, URLs, and bot spam).
* **Comprehensive Evaluation:** Benchmarked against 4 different architectures:
    1.  BERT Base
    2.  RoBERTa Base
    3.  Modified RoBERTa (Custom Classification Head)
    4.  **Hybrid RoBERTa-GRU (Winner)**

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Deep Learning:** PyTorch
* **NLP:** Hugging Face Transformers, NLTK
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Data Source:** Kaggle Tweet Dataset & Yahoo Finance API (`yfinance`)

## 📊 Performance Metrics
The models were evaluated on the held-out test set using Accuracy, Precision, Recall, and F1-Score. The **Hybrid RoBERTa-GRU** demonstrated superior performance across all metrics, indicating better handling of sequential context in financial tweets.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid RoBERTa-GRU (Proposed)** | **0.608** | **0.596** | **0.589** | **0.595** |
| Modified RoBERTa | 0.585 | 0.569 | 0.573 | 0.577 |
| RoBERTa Base | 0.573 | 0.567 | 0.576 | 0.566 |
| BERT Base | 0.561 | 0.568 | 0.555 | 0.547 |

## 📂 Project Structure
```bash
├── stock_classifier_tweet.py     # Main model training & evaluation loop (PyTorch)
├── tweet_data_preprocessing.py   # Text cleaning, tokenization, and dataset splitting
├── news_data_collection.py       # Scripts for fetching financial news/data
├── Msc_Project_2792443s.pdf      # Full Research Thesis Report
└── README.md                     # Documentation
