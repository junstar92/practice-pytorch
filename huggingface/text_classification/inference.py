from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_dir = "./pretrained_model"

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=6)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    emotions = load_dataset("emotion")
    labels = emotions["train"].features["label"].names

    custom_tweet = "I saw a movie today and it was really good."
    preds = classifier(custom_tweet, return_all_scores=True)

    preds_df = pd.DataFrame(preds[0])
    plt.bar(labels, 100 * preds_df["score"], color='C0')
    plt.title(f'"{custom_tweet}"')
    plt.ylabel("Class probability (%)")
    plt.show()