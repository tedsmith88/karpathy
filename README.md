# Andrej Karpathy's Neural Networks Zero to Hero — Hands-on Practice

This repository contains my personal implementations, experiments, and notes from Andrej Karpathy's acclaimed **"Neural Networks: Zero to Hero"** lecture series. Every notebook here was written from scratch while following the lectures to master the underlying math and mechanics of deep learning—moving from raw scalar gradients to large language model (LLM) architectures.

---

## 🛠️ Repository Roadmap & Contents

### 1. The Foundations & Autograd (`micrograd`)
* `micrograd_karpathy.ipynb` – Implementation of a tiny scalar-valued autograd engine. Covers building computation graphs, forward passes, and backpropagation from scratch.

### 2. Character-Level Language Modeling (`makemore`)
A series of notebooks dedicated to generating new names (using the `names.txt` dataset), progressively upgrading the architecture from simple statistics to deep neural networks:
* `makemore_karpathy.ipynb` – **Bigram Model:** Introduction to character-level language modeling using both a frequency lookup table and a basic single-layer neural network.
* `makemore MLP.ipynb` – **Multilayer Perceptron:** Implementing a Bengio et al. (2003) architecture. Covers word/character embeddings and stacking linear layers.
* `makemore pt.3.ipynb` – **Activation & Gradients:** A deep dive into internal network dynamics. Focuses on fixing broken gradients, proper weight initialization (Kaiming init), and implementing Batch Normalization.
* `pt4_backprop.ipynb` – **Manual Backpropagation:** The ultimate "hardcore" exercise. Writing out the exact mathematical derivatives for all layers (Linear, BatchNorm, Tanh, CrossEntropy) on raw PyTorch tensors without ever calling `.backward()`.
* `makemore pt5.ipynb` – **WaveNet:** Upgrading the MLP to a hierarchical dilated convolutional network (WaveNet layout) to efficiently handle longer context lengths.

### 3. The Transformer Era (`NanoGPT`)
* 📂 `NanoGPT/` – Transitioning from character-level models to full-scale generative pre-trained transformers (GPT). Building Self-Attention, Multi-Head Attention, Residual connections, and training a mini-GPT on the TinyShakespear dataset.

---

## 🚀 Key Concepts & Skills Mastered

* **Under-the-Hood Backprop:** Deep understanding of the Chain Rule, calculating matrix/tensor derivatives manually.
* **Network Diagnostics:** Spotting dead neurons, analyzing activation distributions, and understanding why proper weight scaling ($1/\sqrt{fan\_in}$) matters.
* **Architectural Shifts:** Progressing from simple N-grams to MLPs, WaveNets, and ultimately the modern Decoder-only Transformer architecture.
* **Generative Mechanics:** Tokenization, managing context windows, handling logits, and temperature-based sampling.

---

## 📁 File Structure Reference

As seen in the project layout (e.g., `image_203520.png`):
* `names.txt` – The core dataset used for training the `makemore` models.
* `_karpathy` and `ptX` prefixes mark the exact chronological progress through the course lectures.
