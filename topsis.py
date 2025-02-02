import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import textstat
import language_tool_python
import torch

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**encodings, labels=encodings["input_ids"]).loss
    return torch.exp(loss).item()

def generate_text():
    models = [
        ("gpt2", "gpt2"),
        ("distilgpt2", "distilgpt2"),
        ("EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-125M"),
        ("facebook/opt-125m", "facebook/opt-125m"),
        ("bigscience/bloom-560m", "bigscience/bloom-560m"),
        ("microsoft/DialoGPT-small", "microsoft/DialoGPT-small")
    ]
    
    results = []
    model_names = []
    tool = language_tool_python.LanguageTool('en-US')
    
    for name, model_id in models:
        text_generator = pipeline("text-generation", model=model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = text_generator("The future of AI is", max_length=30, num_return_sequences=1)[0]['generated_text']
        word_count = len(text.split())
        readability = textstat.flesch_reading_ease(text)
        lex_diversity = len(set(text.split())) / word_count
        grammar_errors = len(tool.check(text))
        perplexity = calculate_perplexity(model, tokenizer, text)
        results.append([word_count, readability, lex_diversity, grammar_errors, perplexity])
        model_names.append(name)
    
    return np.array(results), model_names

def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def topsis(data, weights, impacts):
    norm_data = data / np.sqrt((data ** 2).sum(axis=0))
    weighted_data = norm_data * weights
    ideal_best = np.max(weighted_data, axis=0) * (np.array(impacts) == 1) + np.min(weighted_data, axis=0) * (np.array(impacts) == -1)
    ideal_worst = np.min(weighted_data, axis=0) * (np.array(impacts) == 1) + np.max(weighted_data, axis=0) * (np.array(impacts) == -1)
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    scores = dist_worst / (dist_best + dist_worst)
    return scores

data, model_names = generate_text()
norm_data = normalize(data)
weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
impacts = [1, 1, 1, -1, -1]
scores = topsis(norm_data, weights, impacts)
best_model_index = np.argmax(scores)

for i, (name, score) in enumerate(sorted(zip(model_names, scores), key=lambda x: x[1], reverse=True)):
    print(f"{i+1}. {name}: {score:.4f}")

plt.figure(figsize=(10, 5))
sns.barplot(x=model_names, y=scores, palette="viridis")
plt.xlabel("Models")
plt.ylabel("TOPSIS Score")
plt.title("Model Comparison Based on TOPSIS")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(norm_data, annot=True, xticklabels=["Word Count", "Readability", "Lexical Diversity", "Grammar Errors", "Perplexity"], yticklabels=model_names, cmap="coolwarm")
plt.title("Normalized Feature Values")
plt.show()
