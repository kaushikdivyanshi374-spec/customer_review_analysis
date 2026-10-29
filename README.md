# Flipkart Customer Review Sentiment Analysis

A beginner-friendly Python project that explores product reviews and predicts whether review text is **positive**, **neutral**, or **negative**.

## Project Files

- `data/flipkart_product.csv`: source dataset
- `sentiment_project.py`: data analysis, model training, and prediction commands
- `requirements.txt`: Python packages used by the project

## How Sentiment Labels Are Created

The dataset includes a 1-to-5 star `Rate` but does not include sentiment labels. This project derives labels from that rating:

| Star rating | Sentiment |
| --- | --- |
| 1 or 2 | Negative |
| 3 | Neutral |
| 4 or 5 | Positive |

The model reads the `Summary` and `Review` columns together. Because the labels come from star ratings, the classifier learns to predict this rating-based rule; it is not being evaluated against independently human-labeled sentiment. A review's wording may not always agree with its star rating.

## Setup

Install Python 3.10 or newer. From this project folder, create and activate a virtual environment, then install the packages:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell does not recognize `py`, install Python from [python.org](https://www.python.org/downloads/) and make sure its launcher is enabled. In VS Code, select the `.venv` Python interpreter after setup.

## Analyze the Dataset

```powershell
python sentiment_project.py analyze
```

This prints the usable row count, star-rating counts, sentiment counts, and the ten most-reviewed products with their average rating. It also saves `outputs/sentiment_distribution.png`.

## Train and Evaluate

```powershell
python sentiment_project.py train
```

The script combines each review summary and review body, creates a stratified 80/20 train/test split, and trains a TF-IDF text vectorizer with a linear support vector classifier. It prints precision, recall, F1-score, and a confusion matrix for the held-out test data. The trained model is saved to `models/sentiment_model.joblib`.

## Predict a New Review

Train the model first, then pass review text in quotation marks:

```powershell
python sentiment_project.py predict "The product works great and the quality is excellent"
```

To use another CSV or change output locations, run `python sentiment_project.py --help` for the available options.

## Beginner Notes

- **Precision**: of the reviews predicted as a sentiment, how many were correct?
- **Recall**: of the reviews that belong to a sentiment, how many did the model find?
- **F1-score**: a balance of precision and recall.
- **Confusion matrix**: rows are actual labels and columns are predicted labels; off-diagonal counts show which sentiments were mixed up.

This is a baseline model. Results depend on the rating-derived labels and the dataset's balance and quality; they should not be treated as a perfect measure of customer opinion.