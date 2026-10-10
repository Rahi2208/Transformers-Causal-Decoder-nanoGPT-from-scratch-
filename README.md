# Transformers Causal Decoder — nanoGPT from Scratch

A character-level, decoder-only Transformer (GPT-style) built from scratch in **PyTorch**, following Andrej Karpathy's *"Let's build GPT: from scratch, in code, spelled out."*

The model learns to generate text one character at a time by predicting the next character from the previous ones, using **causal (masked) self-attention**.

---

## 🧠 Architecture

```
Input characters
      │
      ▼
Token Embedding + Positional Embedding
      │
      ▼
┌──────────────────────────────┐
│  TransformersBlock  × N      │
│  ┌────────────────────────┐  │
│  │ LayerNorm              │  │
│  │ Masked Multi-Head Attn │──┼─ + residual
│  │ LayerNorm              │  │
│  │ FeedForward            │──┼─ + residual
│  └────────────────────────┘  │
└──────────────────────────────┘
      │
      ▼
Final LayerNorm → Linear (vocab_size)
      │
      ▼
Softmax → next-character probabilities
```

### Causal masking

Each position may only attend to itself and earlier positions. This is done with a lower-triangular matrix registered as a buffer:

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
wei = q @ k.transpose(-2, -1) * head_size ** -0.5   # scaled dot-product
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `batch_size` | 64 | Sequences processed in parallel |
| `block_size` | 128 | Maximum context length (characters) |
| `n_embd` | 256 | Embedding dimension |
| `n_head` | 16 | Attention heads per block (head size = 64 / 4 = 16) |
| `n_layer` | 16 | Number of Transformer blocks |
| `dropout` | 0.0 | Dropout rate |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `max_iters` | 8000 | Training steps |
| `eval_interval` | 100 | Steps between loss evaluations |
| `eval_iters` | 200 | Steps in evaluations |

---

## 📚 Dataset

Trained on **[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)** (~1.1M characters of Shakespeare's plays).

- Character-level tokenizer (vocabulary of 65 unique characters)
- 90% train / 10% validation split

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch

---

**Sample output:**

```
(paste generated text here)
```

The output won't be real Shakespeare, but it picks up the structure: speaker names, line breaks, and word-like spellings.

---


## 🙏 Acknowledgements

- [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
