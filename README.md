# 🧠 From Bi-gram to BabyGPT: A Language Modeling Journey

This project demonstrates a step-by-step evolution from a simple **character-level Bi-gram model** to a **miniature GPT-style Transformer**, built entirely from scratch using PyTorch.

## 📁 Project Structure

- `bigram.py` — A basic character-level Bi-gram language model.  
- `babygpt.py` — A custom Transformer model (BabyGPT) with self-attention and multi-head layers.  
- `input.txt` — The text corpus used to train both models.  
- `babygpt_report.pdf` — A detailed report explaining model design, training, evaluation, and insights.

## 🔍 Highlights

- Character-level tokenization and data preparation  
- Implementation of self-attention and multi-head attention from scratch  
- Transformer block with residual connections, layer normalization, and feed-forward network  
- Model evaluation using **loss** and **perplexity**  
- Comparison between Bi-gram and Transformer performance  
- Sample text generation using trained models

## 🚀 Getting Started

Run either model using:

```bash
python bigram.py
```
or
```bash
python babygpt.py
```

## Results

- Bi-gram model: Perplexity ≈ 12
- Transformer (BabyGPT): Perplexity ≈ 4.4
- GPT-style model produces significantly more coherent and meaningful text.

## Report

Read the full documentation in[babygpt_report.pdf]{https://github.com/sayan-dewanjee23/Babygpt/blob/main/babygpt_report.pdf}

## Inspiration

This project began with curiosity about how language models learn from data. Inspired by open-source efforts and foundational tutorials, the aim was to explore—from the ground up—the journey from simple frequency-based models to neural network-based Transformers.
