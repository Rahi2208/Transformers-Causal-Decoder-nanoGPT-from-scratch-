# Transformers Causal Decoder — nanoGPT from Scratch

A character-level, decoder-only Transformer (GPT-style) built from scratch in **PyTorch**, following Andrej Karpathy's *"Let's build GPT: from scratch, in code, spelled out."*

The model learns to generate text one character at a time by predicting the next character from the previous ones, using **causal (masked) self-attention**.

---

## ✨ What's Inside

Every core piece of the Transformer decoder is implemented by hand — no `nn.Transformer`, no pretrained weights:

| Component | Description |
|---|---|
| `Head` | A single head of causal self-attention (key, query, value + lower-triangular mask) |
| `MultiHeadAttention` | Several attention heads running in parallel, concatenated and projected back |
| `FeedForward` | Position-wise MLP (Linear → ReLU → Linear) with a 4× hidden expansion |
| `TransformersBlock` | Multi-head attention + feed-forward, with residual connections and pre-LayerNorm |
| `BigramLanguageModel` | Token + positional embeddings → stack of blocks → LayerNorm → linear head to vocab |

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

```bash
pip install torch
```

### Download the data

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Train and generate

```bash
python gpt.py
```

Or open the notebook in **Google Colab** and run all cells (a GPU runtime is recommended).

The script prints train/validation loss during training and then generates new text from the trained model:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

---

## 📈 Results

| Model | Validation loss |
|---|---|
| Bigram baseline | ~2.5 |
| This Transformer | ~2.0 *(update with your own result)* |

**Sample output:**

```
(paste generated text here)
```

The output won't be real Shakespeare, but it picks up the structure: speaker names, line breaks, and word-like spellings.

---

## 🗺️ Roadmap

- [ ] Scale up the model (larger `n_embd`, more layers, longer context)
- [ ] Switch to a subword tokenizer (BPE / `tiktoken`)
- [ ] Reproduce GPT-2 (124M)
- [ ] Fine-tune a pretrained LLM

---

## 🙏 Acknowledgements

- [Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
