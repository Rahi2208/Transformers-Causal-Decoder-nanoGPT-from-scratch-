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
│  Block  × n_layer            │
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

The LayerNorm is applied **before** each sub-layer (pre-norm), which keeps training stable as the residual stream grows.

### Causal masking

Each position may only attend to itself and earlier positions. This is done with a lower-triangular matrix registered as a buffer:

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

att = Q @ K.transpose(-2, -1) * head_size ** -0.5      # scaled dot-product
att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
att = softmax(att, dim=-1)
```

### Tensor shapes

| Tensor | Shape | Meaning |
|---|---|---|
| `x`, `y` | `(B, T)` | token ids; `y` is `x` shifted one step to the right |
| after embedding | `(B, T, C)` | `C = n_embd` |
| attention weights | `(B, T, T)` | who attends to whom |
| logits | `(B, T, vocab_size)` | scores for the next character |

---

## ⚙️ Hyperparameters

All settings live in a single dataclass, `TCDEconfig`:

| Parameter | Value | Meaning |
|---|---|---|
| `batch_size` | 16 | Sequences processed in parallel |
| `block_size` | 32 | Maximum context length (characters) |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 4 | Attention heads per block (head size = 64 / 4 = 16) |
| `n_layer` | 4 | Number of Transformer blocks |
| `dropout` | 0.0 | Dropout rate |
| `lr` | 1e-3 | AdamW learning rate |
| `epochs` | 5000 | Training steps |
| `test_interval` | 100 | Steps between loss/accuracy evaluations |
| `test_epochs` | 200 | Batches averaged per evaluation |

Override any of them without editing the class:

```python
config = TCDEconfig(n_embd=128, dropout=0.3)
```

---

## 📚 Dataset

Trained on **[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)** (~1.1M characters of Shakespeare's plays).

- Character-level tokenizer (vocabulary of 65 unique characters)
- 90% train / 10% test split

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- Jupyter (if you run the notebook)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Rahi2208/Transformers-Causal-Decoder-nanoGPT-from-scratch-.git
cd Transformers-Causal-Decoder-nanoGPT-from-scratch-

# 2. Install dependencies
pip install torch

# 3. Download the dataset (saved as input.txt)
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Run

Open `TCDE.ipynb` and run all cells. The notebook will:

1. Read `input.txt` and build the character vocabulary
2. Train the model and print train/test loss and accuracy every `test_interval` steps
3. Generate text from the trained model

A GPU is used automatically when available (`cuda`), otherwise it falls back to CPU.

### Generate text

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)   # start from token id 0
print(decoder(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

---

## 📁 Project Structure

```
.
├── TCDE.ipynb      # model, training loop, and text generation
├── input.txt       # Tiny Shakespeare dataset (download it first)
└── README.md
```

---

## ✨ Sample output

```
(paste generated text here)
```

The output won't be real Shakespeare, but it picks up the structure: speaker names, line breaks, and word-like spellings.

---

## 🙏 Acknowledgements

- [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
