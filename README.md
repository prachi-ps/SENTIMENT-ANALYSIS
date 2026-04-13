# SENTIMENT ANALYSIS

COMPANY: CODTECH IT SOLUTIONS

NAME: PRACHI SONONE

INTERN ID: CTIS2104

DOMAIN: DATA ANALYTICS

DURATION: 16 WEEKS

MENTOR: NEELA SANTHOSH KUMAR


A model that reads Amazon Alexa customer reviews and predicts whether the review is positive or negative.

---

## About the project

Built an NLP pipeline on Amazon Alexa reviews that goes from raw messy text all the way to a working classifier. Helped understand how text classification actually works.

---

## Dataset

- **Source:** Amazon Alexa Reviews (`amazon_alexa.tsv`)
- **Size:** 3,150 reviews
- **Input column:** `verified_reviews` - what customers actually wrote
- **Target column:** `feedback` - 1 = Positive, 0 = Negative

One thing to note: 92% of reviews are positive and only 8% are negative. That's a big imbalance and it does affect the results.

---

## Process

**1. EDA (Exploratory Data Analysis)**  
Data Exploration - checked for missing values, looked at the class distribution, star ratings, and how long reviews usually are. Turns out negative reviews tend to be longer.

**2. Text Preprocessing**  
Cleaned raw text using:
- Lowercasing everything
- Removing URLs, HTML tags, punctuation, and numbers (with regex)
- Removing stopwords (common words like "the", "is", "a")
- Lemmatization — converting words to their root form ("loving" -> "love")

**3. Word Clouds + Frequency Charts**  
Made word clouds separately for positive and negative reviews. Positive reviews are full of words like love, great, easy, music. Negative ones kept showing problem, connect, return, issue, mostly connectivity and hardware complaints.

**4. TF-IDF Vectorization**  
ML models can't read text directly, converted the cleaned reviews into numbers using TF-IDF. It gives higher weight to words that are important in a specific review and lower weight to words that appear everywhere.

**5. Model Training**  
Trained 3 models and compared them:

| Model | Quick note |
|---|---|
| Logistic Regression | Good baseline, easy to interpret |
| Naive Bayes | Fast and simple, but struggled a bit with the imbalanced data |
| Linear SVM | Best performer overall |

Used an 80/20 train-test split with stratification so both splits had the same class ratio.

**6. Evaluation**  
Compared all 3 models using accuracy, precision, recall, F1-score, and confusion matrices.

**7. Custom Prediction**  
Built a small function where you can type any review text and it tells you if it's positive or negative.

---

## Results

Logistic Regression did the best, followed closely by Linear SVM. Naive Bayes was decent but fell behind on the minority (negative) class. 

---

## What I used

- Python 3.10
- pandas, numpy
- matplotlib, seaborn, wordcloud
- NLTK
- scikit-learn

---

## How to run it

1. Open `Sentiment_Analysis.ipynb` in Google Colab or Jupyter
2. Upload `amazon_alexa.tsv` to the same working directory
3. Run all cells from top to bottom

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

