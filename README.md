# NumPy Word2Vec (Skip-Gram with Negative Sampling)

This repository contains a from-scratch implementation of the Word2Vec algorithm (Skip-Gram with Negative Sampling variant) built entirely in NumPy

## Key Engineering Features

To ensure the model trains efficiently on local hardware, several techniques were implemented:

* **Memory-Efficient Data Pipeline:** `DataProcessor` utilizes a Python generator (`yield`) to stream mini-batches on the fly
* **Highly Vectorized Optimization:** Gradients are calculated using advanced NumPy broadcasting (`np.newaxis`) and safely accumulated using `np.add.at` to handle duplicate word IDs in a single batch
* **Frequent Word Subsampling:** Implements Mikolov's heuristic formula to drop overly frequent, low-information words (like "the" and "a"), forcing the model to focus on semantic relationships
* **$O(1)$ Negative Sampling:** Utilizes a pre-computed sampling table based on the unigram distribution raised to the $3/4$ power, allowing for lightning-fast noise word generation

## Repository Structure

* `src/data_processor.py`: Handles text parsing, vocabulary building, subsampling, and memory-efficient batch generation
* `src/model.py`: The core mathematical engine. Contains the `Word2Vec` class, initialization, vectorized forward pass, loss calculation, and gradient updates
* `notebooks/word2vec_training.ipynb`: Interactive Jupyter Notebook that ties the pipeline together, trains the model, and evaluates cosine similarities

## ⚙️ Quick Start

1. Clone the repository
``` bash
git clone ...
```
2. Install requirments
``` bash
pip install -r requirments.txt
```
3. Launch and run the notebook
``` bash
jupyter notebook notebooks/word2vec_training.ipynb
```