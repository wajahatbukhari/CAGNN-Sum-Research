# CAGNN-Sum: Context-Aware Heterogeneous Graph Neural Network for Extractive Summarization

This repository contains the implementation of **CAGNN-Sum**, a Context-Aware Heterogeneous Graph Neural Network designed for the extractive summarization of long scientific papers. This project was developed as part of Master's research in Computer Technology at the University of Science and Technology of China (USTC) by Syed Wajahat Ali Bukhari.

## 📌 Project Overview
Standard extractive summarization models often treat documents as simple sequences of sentences, which can lead to a loss of global context, especially in complex, domain-specific scientific literature. CAGNN-Sum addresses this by modeling the document as a rich **Heterogeneous Graph**. 

By incorporating Named Entities and Latent Topics as distinct nodes alongside sentences, the model learns not just structural importance, but the thematic and domain-specific weight of the text. The model is trained and evaluated on the [arXiv Summarization Dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization).

## 🏗️ Architecture Breakdown

The pipeline consists of three major components:

1. **Graph Construction**
   - **Sentence Nodes:** Individual sentences extracted from the document.
   - **Entity Nodes:** Named entities (ORG, PERSON, WORK_OF_ART) extracted using `spaCy`.
   - **Topic Nodes:** Global scientific themes identified across the corpus using `BERTopic` and `SentenceTransformers`.
   - **Edges:** Sentences are connected to other sentences via TF-IDF cosine similarity (`similar_to`), to entities they contain (`mentions`), and to global themes (`belongs_to`).

2. **Neural Encoding**
   - Implemented using the Deep Graph Library (`DGL`).
   - A multi-layer `HeteroGraphConv` network passes messages across the different edge types, enriching sentence representations with entity and topic context.

3. **Multi-Level Attention Pooling**
   - A custom attention mechanism calculates a final importance score for each sentence by weighing three streams: the sentence's internal embedding, the aggregated entity context, and the aggregated topic context.

## 🚀 Installation & Requirements

To run the Jupyter Notebook, you will need a Python environment (Python 3.8+ recommended) with GPU support for PyTorch and DGL.

```bash
# Core Machine Learning & Graph Libraries
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install dgl -f [https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html](https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html)

# NLP & Topic Modeling
pip install spacy datasets transformers sentence-transformers bertopic
python -m spacy download en_core_web_sm

# Visualization
pip install matplotlib networkx seaborn
