"""Beginner-friendly Flipkart review sentiment analysis workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "flipkart_product.csv"
DEFAULT_MODEL = PROJECT_DIR / "models" / "sentiment_model.joblib"
DEFAULT_RATING_MODEL = PROJECT_DIR / "models" / "rating_regression_model.joblib"
DEFAULT_CHART = PROJECT_DIR / "outputs" / "sentiment_distribution.png"
DEFAULT_REGRESSION_CHART = PROJECT_DIR / "outputs" / "rating_regression.png"
LABEL_ORDER = ["negative", "neutral", "positive"]


def load_reviews(csv_path: Path) -> pd.DataFrame:
    """Load usable review text and create sentiment labels from star ratings."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig", encoding_errors="replace")
    required_columns = {"Rate", "Review", "Summary"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    ratings = pd.to_numeric(data["Rate"], errors="coerce")
    data = data.loc[ratings.isin([1, 2, 3, 4, 5])].copy()
    data["Rate"] = ratings.loc[data.index].astype(int)
    data["text"] = (
        data["Summary"].fillna("").astype(str).str.strip()
        + " "
        + data["Review"].fillna("").astype(str).str.strip()
    ).str.strip()
    data = data.loc[data["text"].ne("")].copy()
    data["sentiment"] = data["Rate"].map(
        {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    )
    return data


def analyze(data: pd.DataFrame, chart_path: Path) -> None:
    """Print dataset summaries and save a chart of the derived labels."""
    print(f"Usable reviews: {len(data):,}")
    print("\nRating counts:")
    print(data["Rate"].value_counts().sort_index().to_string())
    print("\nSentiment counts:")
    sentiment_counts = data["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    print(sentiment_counts.to_string())
    print("\nTop 10 products by review count:")
    if "ProductName" in data.columns:
        product_summary = (
            data.groupby("ProductName")
            .agg(review_count=("Rate", "size"), average_rating=("Rate", "mean"))
            .sort_values("review_count", ascending=False)
            .head(10)
        )
        print(product_summary.to_string(float_format=lambda value: f"{value:.2f}"))

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    counts = sentiment_counts
    axes = counts.plot(kind="bar", color=["#d46a5f", "#d8aa4d", "#4e9a78"])
    axes.set_title("Review sentiment from star ratings")
    axes.set_xlabel("Sentiment")
    axes.set_ylabel("Number of reviews")
    axes.tick_params(axis="x", rotation=0)
    axes.figure.tight_layout()
    axes.figure.savefig(chart_path, dpi=150)
    plt.close(axes.figure)
    print(f"\nChart saved to: {chart_path}")

    rating_chart = chart_path.parent / "rating_distribution.png"
    rating_axes = data["Rate"].value_counts().sort_index().plot(
        kind="bar", color="#4b8296"
    )
    rating_axes.set_title("Customer star-rating distribution")
    rating_axes.set_xlabel("Star rating")
    rating_axes.set_ylabel("Number of reviews")
    rating_axes.tick_params(axis="x", rotation=0)
    rating_axes.figure.tight_layout()
    rating_axes.figure.savefig(rating_chart, dpi=150)
    plt.close(rating_axes.figure)
    print(f"Rating chart saved to: {rating_chart}")

    if "ProductName" in data.columns:
        product_ratings = (
            data.groupby("ProductName")
            .agg(review_count=("Rate", "size"), average_rating=("Rate", "mean"))
            .query("review_count >= 30")
            .sort_values("average_rating", ascending=False)
            .head(10)
            .sort_values("average_rating")
        )
        if not product_ratings.empty:
            product_chart = chart_path.parent / "top_product_average_ratings.png"
            product_axes = product_ratings["average_rating"].plot(
                kind="barh", color="#4e9a78"
            )
            product_axes.set_title("Top-rated products (at least 30 reviews)")
            product_axes.set_xlabel("Average star rating")
            product_axes.set_ylabel("")
            product_axes.figure.tight_layout()
            product_axes.figure.savefig(product_chart, dpi=150)
            plt.close(product_axes.figure)
            print(f"Product chart saved to: {product_chart}")


def train(data: pd.DataFrame, model_path: Path) -> None:
    """Train a text classifier and report performance on held-out reviews."""
    label_counts = data["sentiment"].value_counts()
    if len(label_counts) != len(LABEL_ORDER) or label_counts.min() < 2:
        raise ValueError("At least two reviews are required for each sentiment class.")

    train_text, test_text, train_labels, test_labels = train_test_split(
        data["text"],
        data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["sentiment"],
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)

    print(f"Training reviews: {len(train_text):,}")
    print(f"Test reviews: {len(test_text):,}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
            digits=3,
        )
    )
    matrix = confusion_matrix(test_labels, predictions, labels=LABEL_ORDER)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_string())

    chart_dir = DEFAULT_CHART.parent
    chart_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 5))
    image = axes.imshow(matrix, cmap="Greens")
    axes.set(
        xticks=range(len(LABEL_ORDER)),
        yticks=range(len(LABEL_ORDER)),
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        xlabel="Predicted sentiment",
        ylabel="Actual sentiment",
        title="Sentiment model confusion matrix",
    )
    for row in range(len(LABEL_ORDER)):
        for column in range(len(LABEL_ORDER)):
            axes.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Number of reviews")
    figure.tight_layout()
    confusion_chart = chart_dir / "sentiment_confusion_matrix.png"
    figure.savefig(confusion_chart, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix chart saved to: {confusion_chart}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


def regress(data: pd.DataFrame, model_path: Path, chart_path: Path) -> None:
    """Predict star ratings from review text using a simple Ridge model."""
    train_text, test_text, train_ratings, test_ratings = train_test_split(
        data["text"],
        data["Rate"],
        test_size=0.2,
        random_state=42,
        stratify=data["Rate"],
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_text, train_ratings)
    predictions = model.predict(test_text)
    mae = mean_absolute_error(test_ratings, predictions)
    rmse = mean_squared_error(test_ratings, predictions) ** 0.5
    r_squared = r2_score(test_ratings, predictions)

    print(f"Training reviews: {len(train_text):,}")
    print(f"Test reviews: {len(test_text):,}")
    print(f"Mean absolute error (MAE): {mae:.3f} stars")
    print(f"Root mean squared error (RMSE): {rmse:.3f} stars")
    print(f"R-squared: {r_squared:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(test_ratings, predictions, alpha=0.15, color="#4b8296")
    axes[0].plot([1, 5], [1, 5], linestyle="--", color="#d46a5f")
    axes[0].set(
        xlabel="Actual rating",
        ylabel="Predicted rating",
        title="Actual vs. predicted ratings",
        xlim=(1, 5),
        ylim=(1, 5),
    )
    axes[1].hist(test_ratings - predictions, bins=30, color="#d8aa4d")
    axes[1].set(
        xlabel="Rating error (actual - predicted)",
        ylabel="Number of reviews",
        title="Prediction error distribution",
    )
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    print(f"\nModel saved to: {model_path}")
    print(f"Regression charts saved to: {chart_path}")


def predict(model_path: Path, text: str) -> None:
    """Load a trained model and predict the sentiment of one review."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run the train command first."
        )
    model = joblib.load(model_path)
    print(f"Predicted sentiment: {model.predict([text])[0]}")


def predict_rating(model_path: Path, text: str) -> None:
    """Load a trained regression model and predict a star rating."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run the regress command first."
        )
    model = joblib.load(model_path)
    rating = model.predict([text])[0]
    print(f"Predicted star rating: {rating:.1f} out of 5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Flipkart product reviews.")
    parser.add_argument(
        "command",
        choices=["analyze", "train", "predict", "regress", "predict-rating"],
        help="Workflow step to run",
    )
    parser.add_argument(
        "text", nargs="*", help="Review text to classify or rate"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input CSV path")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help="Saved model path"
    )
    parser.add_argument(
        "--rating-model",
        type=Path,
        default=DEFAULT_RATING_MODEL,
        help="Saved rating-regression model path",
    )
    parser.add_argument(
        "--chart", type=Path, default=DEFAULT_CHART, help="Output chart path"
    )
    parser.add_argument(
        "--regression-chart",
        type=Path,
        default=DEFAULT_REGRESSION_CHART,
        help="Output rating-regression chart path",
    )
    args = parser.parse_args()

    if args.command in {"predict", "predict-rating"}:
        if not args.text:
            parser.error(f"{args.command} requires review text in quotation marks")
        if args.command == "predict":
            predict(args.model, " ".join(args.text))
        else:
            predict_rating(args.rating_model, " ".join(args.text))
        return

    data = load_reviews(args.data)
    if args.command == "analyze":
        analyze(data, args.chart)
    elif args.command == "regress":
        regress(data, args.rating_model, args.regression_chart)
    else:
        train(data, args.model)


if __name__ == "__main__":
    main()
