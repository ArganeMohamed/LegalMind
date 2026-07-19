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


OFF_TOPIC_QUESTIONS = [
    "What time is it?",
    "Which team won the Champions League in 2015?",
    "Give me Python code for a CNN using TensorFlow.",
    "What's 245 times 18?",
    "What's the weather like today?",
    "Tell me a joke.",
    "How far is the moon from Earth?",
    "What's the capital of Japan?",
    "Can you recommend a good recipe for pasta?",
    "Who painted the Mona Lisa?",
    "Translate 'good morning' to French.",
    "What's the best programming language to learn in 2026?",
    "How do I fix a flat tire?",
    "What's the meaning of life?",
    "Who is the current president of France?",
    "Write me a short poem about the ocean.",
    "What's a healthy breakfast option?",
    "How does photosynthesis work?",
    "What's trending on social media right now?",
    "Can you help me plan a birthday party?",
    "What's the square root of 144?",
    "Who wrote Romeo and Juliet?",
    "How do I lose weight fast?",
    "What's the best smartphone to buy right now?",
    "Can you write me a resignation letter for my job?",
    "What's the difference between a virus and bacteria?",
    "How do airplanes stay in the air?",
    "What's a good name for my new puppy?",
    "Explain quantum computing in simple terms.",
    "What's the tallest mountain in the world?",
    "How do I invest in stocks?",
    "Write a birthday message for my friend.",
    "What's the population of Brazil?",
    "How long does it take to boil an egg?",
    "What's your favorite color?",
    "Can you help me debug this JavaScript error?",
    "What's the best workout routine for beginners?",
    "How do I make my WiFi faster?",
    "Who is Elon Musk?",
    "What's the plot of the movie Inception?",
    "Give me some tips for public speaking.",
    "What's 15% of 200?",
    "How do vaccines work?",
    "Can you recommend a good book to read?",
    "What's the exchange rate between USD and EUR?",
    "How do I set up a Gmail account?",
    "What's the difference between machine learning and AI?",
    "Write me a haiku about autumn.",
    "How many calories are in a banana?",
    "What's the best way to learn a new language?",
    "Can you summarize the plot of Harry Potter?",
    "What's the fastest animal on Earth?",
    "How do I change a car battery?",
    "What's your opinion on cryptocurrency?",
    "Give me a workout playlist recommendation.",
    "How does the stock market work?",
    "What's the history of the Roman Empire?",
    "Can you help me write a wedding speech?",
    "What's the boiling point of water in Fahrenheit?",
    "How do I improve my sleep quality?",
    "I just moved to a new city and I don't know anyone here yet.",
    "I've been feeling really stressed at work lately because of tight deadlines.",
    "My laptop keeps overheating whenever I run multiple programs at once.",
    "I'm planning a trip to Japan next month and I've never traveled abroad before.",
    "My dog has been acting differently the past few days and I'm a bit worried.",
    "I recently switched careers from marketing to software development.",
    "I have a big presentation coming up and I'm feeling pretty nervous about it.",
    "Our team just finished a major project and everyone is exhausted.",
    "I'm trying to save money for a house but it's harder than I expected.",
    "My internet connection has been really unstable for the past week.",
    "I started going to the gym three months ago and I've noticed some progress.",
    "My sister is getting married next summer and I'm one of the bridesmaids.",
    "I've been learning to cook and today I tried making sushi for the first time.",
    "Our office is relocating to a new building next quarter.",
    "I'm a college student majoring in biology and I have finals next week.",
    "My car has been making a strange noise when I brake.",
    "I adopted a kitten last weekend and she's still getting used to the house.",
    "I've been trying to wake up earlier but I keep hitting snooze.",
    "My phone battery drains really fast even when I'm not using many apps.",
    "I just started a new job and I'm still learning where everything is.",
]

REFUSAL_TEMPLATES = [
    "I'm LegalMind, a contract clause classification and explanation assistant. I can't help with that — please provide a contract clause you'd like analyzed.",
    "That's outside what I do. I'm LegalMind — I classify and explain contract clauses. Send me a clause and I'll help.",
    "I'm specialized in contract clause analysis, not general questions. If you have a clause you'd like classified and explained, I'm ready.",
]


def build_refusal_examples():
    rows = []
    for i, question in enumerate(OFF_TOPIC_QUESTIONS):
        refusal = REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)]
        rows.append({
            "clause": question,
            "label": "OUT_OF_SCOPE",
            "excerpt": "",
            "explanation": refusal,
        })
    return rows


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

    refusal_rows = build_refusal_examples()
    train_rows = train_rows + refusal_rows
    save_dataset(train_rows, "data/distilled_train.jsonl")

    print(f"\nrefusal examples added: {len(refusal_rows)}")
    print(f"final train size: {len(train_rows)}")