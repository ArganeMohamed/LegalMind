import re
import json
import os
from datasets import load_dataset


def filter_none(data):
    return data.filter(lambda x: x["label"] != "NONE")


def extract_quotes(rationale_text):
    return re.findall(r'"([^"]+)"', rationale_text)


def clean_quote(quote_text):
    return quote_text.replace("**", "").strip()


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_self_referential(quote_text, label):
    q = normalize(quote_text)
    lbl = normalize(label)
    if q == lbl:
        return True
    q_words = set(q.split())
    lbl_words = set(lbl.split())
    return len(q_words) <= 4 and q_words and q_words.issubset(lbl_words)


def is_grounded(source_text, quote):
    return quote.lower() in source_text.lower()


def grounded_quotes_for_row(row):
    quotes = extract_quotes(row["rationale"])
    survivors = []
    for raw_q in quotes:
        q = clean_quote(raw_q)
        if is_self_referential(q, row["label"]):
            continue
        if is_grounded(row["clause"], q):
            survivors.append(q)
    return survivors


def pick_best_quote(survivors):
    return max(survivors, key=len) if survivors else None


def explanation_sentence_for_quote(rationale_text, quote):
    lines = rationale_text.split("\n")
    for i, line in enumerate(lines):
        quoted_spans = extract_quotes(line)
        for span in quoted_spans:
            if clean_quote(span).lower() == quote.lower():
                sentence = re.sub(r"^[\-\*\d\.\s]+", "", line)
                sentence = re.sub(r"\*\*", "", sentence).strip()

                next_index = i + 1
                max_extra_lines = 3
                while sentence.endswith(":") and max_extra_lines > 0 and next_index < len(lines):
                    next_clean = re.sub(r"^[\-\*\d\.\s]+", "", lines[next_index])
                    next_clean = re.sub(r"\*\*", "", next_clean).strip()
                    if next_clean:
                        sentence = sentence + " " + next_clean
                        max_extra_lines -= 1
                    next_index += 1

                return sentence
    return None


def build_clean_example(row):
    survivors = grounded_quotes_for_row(row)
    if not survivors:
        return None

    best_quote = pick_best_quote(survivors)
    explanation = explanation_sentence_for_quote(row["rationale"], best_quote)
    if not explanation:
        return None

    return {
        "clause": row["clause"],
        "label": row["label"],
        "excerpt": best_quote,
        "explanation": explanation,
    }


def build_dataset(data):
    cleaned_rows = []
    for row in data:
        result = build_clean_example(row)
        if result is not None:
            cleaned_rows.append(result)
    return cleaned_rows


def save_dataset(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    dataset = load_dataset("zenml/cuad-deepseek")

    train_data = filter_none(dataset["train"])
    train_rows = build_dataset(train_data)
    save_dataset(train_rows, "data/distilled_train.jsonl")
    train_clauses = set(r["clause"].strip() for r in train_rows)

    print("train")
    print(f"clean rows saved : {len(train_rows)}")

    val_data = filter_none(dataset["validation"])
    val_rows = build_dataset(val_data)
    val_rows = [r for r in val_rows if r["clause"].strip() not in train_clauses]
    save_dataset(val_rows, "data/distilled_validation.jsonl")
    val_clauses = set(r["clause"].strip() for r in val_rows)

    print("\nvalidation")
    print(f"clean rows saved : {len(val_rows)}")

    test_data = filter_none(dataset["test"])
    test_rows = build_dataset(test_data)
    test_rows = [
        r for r in test_rows
        if r["clause"].strip() not in train_clauses
        and r["clause"].strip() not in val_clauses
    ]
    save_dataset(test_rows, "data/distilled_test.jsonl")

    print("\ntest")
    print(f"clean rows saved : {len(test_rows)}")