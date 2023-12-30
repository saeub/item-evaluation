import argparse
import csv
import re
import string
import sys
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Literal, Sequence

import numpy as np
import pandas as pd
import sklearn.metrics

from data import Article
from models import MODEL_CLASSES, QAModel


def predict(
    articles: Sequence[Article],
    model_class: type[QAModel],
    kwargs: dict[str, Any],
    guess: bool,
):
    model = model_class(**kwargs)
    for article in articles:
        text = article.text
        if guess:
            text = None
        else:
            assert article.text is not None, f"Text of {article} is None."
        for item in article.items:
            predictions = model.answer(text, item)
            for answer, prediction in zip(item.answers, predictions):
                answer.prediction = prediction
        print(article.to_json())


def evaluate(
    articles: Collection[Article],
):
    trues = []
    probs = []
    preds = []
    for article in articles:
        for item in article.items:
            for answer in item.answers:
                if answer.prediction is not None:
                    trues.append(answer.correct)
                    probs.append(answer.prediction.prob_correct)
                    preds.append(answer.prediction.pred_correct)

    accuracy = sklearn.metrics.accuracy_score(trues, preds)
    print(f"Accuracy: {accuracy:.2%}")

    f1 = sklearn.metrics.f1_score(trues, preds)
    print(f"F1: {f1:.2%}")

    if None not in probs:
        auc = sklearn.metrics.roc_auc_score(trues, probs)
        print(f"ROC AUC: {auc:.2%}")


def optimize(
    articles: Sequence[Article],
    train_articles: Collection[Article],
):
    # Find optimal threshold in train_articles
    probs = []
    trues = []
    for article in train_articles:
        for item in article.items:
            for answer in item.answers:
                probs.append(answer.prediction.prob_correct)
                trues.append(answer.correct)
    best_accuracy = 0
    best_threshold = 0
    for threshold in probs:
        preds = [prob >= threshold for prob in probs]
        accuracy = sklearn.metrics.accuracy_score(trues, preds)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    print(f"Best threshold: {best_threshold:.2%} (train accuracy: {best_accuracy:.2%})", file=sys.stderr)

    # Apply threshold to articles
    for article in articles:
        for item in article.items:
            for answer in item.answers:
                if answer.prediction is not None:
                    answer.prediction.pred_correct = answer.prediction.prob_correct >= best_threshold
        print(article.to_json())


def scores(
    subject_articles: dict[str, Sequence[Article]],
):
    def iter_scores(articles):
        for article in articles:
            for item in article.items:
                for answer in item.answers:
                    if answer.prediction is None:
                        yield None
                    else:
                        yield int(answer.prediction.pred_correct == answer.correct)

    writer = csv.writer(sys.stdout)
    subjects = list(subject_articles.keys())
    article_lists = [subject_articles[subject] for subject in subjects]
    writer.writerow(["item", *subjects])
    for i, scores in enumerate(zip(*map(iter_scores, article_lists))):
        writer.writerow([f"item-{i}", *scores])


def confusion(
    articles: Collection[Article], format: Literal["csv", "latex"] | None = None
):
    pred_true_counts = defaultdict(int)
    for article in articles:
        for item in article.items:
            for answer in item.answers:
                pred = answer.prediction.pred_correct
                true = answer.correct
                pred_true_counts[pred, true] += 1
    data = {"pred": [], "true": [], "count": []}
    for (pred, true), count in pred_true_counts.items():
        data["pred"].append(pred)
        data["true"].append(true)
        data["count"].append(count)
    df = pd.DataFrame(data).pivot(columns="pred", index="true", values="count")
    if format == "csv":
        print(df.to_csv())
    elif format == "latex":
        print(df.to_latex())
    else:
        print(df)


def errors(articles: Collection[Article], short: bool):
    for article in articles:
        for item in article.items:
            if short:
                text = re.sub(r"\n+", " ", article.text)
                summary = f"{text[:200]}...\n"
            else:
                text = re.sub(r"\n+", "\n", article.text)
                summary = f"{text}\n"
            summary += f"❔ {item.question}\n"
            has_error = False
            for letter, answer in zip(string.ascii_uppercase, item.answers):
                summary += f"{letter}) {answer.text}\n"
                summary += f"   → {answer.prediction.pred_correct}"
                if answer.prediction.prob_correct is not None:
                    summary += f" ({answer.prediction.prob_correct:.0%})"
                if answer.prediction.pred_correct == answer.correct:
                    summary += f" ✅️"
                else:
                    summary += f" ❌ ({answer.correct})"
                    has_error = True
                summary += "\n"
            if has_error:
                print(summary)


def agreement(
    subject_articles: dict[str, Sequence[Article]],
):
    subjects = list(subject_articles.keys())
    article_lists = [subject_articles[subject] for subject in subjects]

    def iter_predictions(articles, prob=False):
        for article in articles:
            for item in article.items:
                for answer in item.answers:
                    if answer.prediction is None:
                        yield None
                    elif prob:
                        yield answer.prediction.prob_correct
                    else:
                        yield answer.prediction.pred_correct

    iter_pred = lambda articles: iter_predictions(articles, prob=False)
    iter_prob = lambda articles: iter_predictions(articles, prob=True)
    df_pred = pd.DataFrame(
        zip(*map(iter_pred, article_lists)),
        columns=subjects,
    )
    df_prob = pd.DataFrame(
        zip(*map(iter_prob, article_lists)),
        columns=subjects,
    )
    print("Pearson correlation coefficients:")
    print(df_prob.corr(method="pearson"))
    print()
    print("Spearman rank correlation coefficients:")
    print(df_prob.corr(method="spearman"))
    print()
    print("Kendall rank correlation coefficients:")
    print(df_prob.corr(method="kendall"))
    print()

    df_pairwise_cohen = pd.DataFrame(
        index=subjects,
        columns=subjects,
    )
    for subject1 in subjects:
        for subject2 in subjects:
            if subject1 == subject2:
                df_pairwise_cohen.loc[subject1, subject2] = 1
            else:
                pred1 = df_pred[subject1]
                pred2 = df_pred[subject2]
                mask = pred1.notnull() & pred2.notnull()
                pred1 = pred1[mask].astype(bool)
                pred2 = pred2[mask].astype(bool)
                df_pairwise_cohen.loc[
                    subject1, subject2
                ] = sklearn.metrics.cohen_kappa_score(pred1, pred2)
    print("Cohen's kappa:")
    print(df_pairwise_cohen)
    df_pairwise_cohen_mean = df_pairwise_cohen.mask(
        np.eye(len(df_pairwise_cohen), dtype=bool)
    ).mean()
    print("Means:")
    print(df_pairwise_cohen_mean)


def kwarg_eval(string: str) -> tuple[str, Any]:
    key, value = string.split("=")
    try:
        value = literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    return key, value


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser(
        "predict", help="predict answers using a model and print output"
    )
    predict_parser.add_argument(
        "file",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="?",
        default=sys.stdin,
    )
    predict_parser.add_argument("--model", choices=MODEL_CLASSES.keys(), required=True)
    predict_parser.add_argument("--kwargs", type=kwarg_eval, nargs="*", default=[])
    predict_parser.add_argument("--guess", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="print accuracy")
    evaluate_parser.add_argument(
        "files",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="*",
        default=[sys.stdin],
        help="prediction output files",
    )

    optimize_parser = subparsers.add_parser("optimize", help="optimize classification threshold")
    optimize_parser.add_argument(
        "file",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="?",
        default=sys.stdin,
        help="prediction output files to apply the optimized threshold on",
    )
    optimize_parser.add_argument(
        "--train",
        type=argparse.FileType("r", encoding="utf-8"),
        required=True,
        help="prediction output files to train threshold on",
    )

    scores_parser = subparsers.add_parser(
        "scores",
        help="print item scores (1 = correct, 0 = incorrect) in CSV format (e.g., for IRT analysis)",
    )
    scores_parser.add_argument(
        "files",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="+",
        help="prediction output files",
    )

    confusion_parser = subparsers.add_parser(
        "confusion", help="print confusion matrix of predicted correctness labels"
    )
    confusion_parser.add_argument(
        "file",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="?",
        default=sys.stdin,
        help="prediction output file",
    )
    confusion_parser.add_argument(
        "--format",
        choices=["csv", "latex"],
        help="output format",
    )

    errors_parser = subparsers.add_parser(
        "errors", help="print incorrectly predicted answers"
    )
    errors_parser.add_argument(
        "file",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="?",
        default=sys.stdin,
        help="prediction output file",
    )
    errors_parser.add_argument(
        "--short",
        action="store_true",
        help="print only the beginning of each article text",
    )

    agreement_parser = subparsers.add_parser(
        "agreement", help="print agreement between annotators/models"
    )
    agreement_parser.add_argument(
        "files",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="+",
        help="prediction output files",
    )

    args = parser.parse_args()

    if args.command == "predict":
        articles = [Article.from_json(line) for line in args.file]
        model_class = MODEL_CLASSES[args.model]
        predict(articles, model_class, dict(args.kwargs), guess=args.guess)

    if args.command == "evaluate":
        articles = [Article.from_json(line) for file in args.files for line in file]
        evaluate(articles)

    if args.command == "optimize":
        articles = [Article.from_json(line) for line in args.file]
        train_articles = [Article.from_json(line) for line in args.train]
        optimize(articles, train_articles)

    if args.command == "scores":
        subject_articles = {}
        num_articles = None
        for file in args.files:
            articles = [Article.from_json(line) for line in file]
            if num_articles is None:
                num_articles = len(articles)
            else:
                assert num_articles == len(
                    articles
                ), f"{file.name} has {len(articles)} articles, should be {num_articles}."
            subject = Path(file.name).stem
            subject_articles[subject] = articles
        scores(subject_articles)

    if args.command == "confusion":
        articles = [Article.from_json(line) for line in args.file]
        confusion(articles, args.format)

    if args.command == "errors":
        articles = [Article.from_json(line) for line in args.file]
        errors(articles, args.short)

    if args.command == "agreement":
        subject_articles = {
            Path(file.name).stem: [Article.from_json(line) for line in file]
            for file in args.files
        }
        agreement(subject_articles)


if __name__ == "__main__":
    main()
