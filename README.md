![Logo](https://github.com/AKGanesh/Basic_Text_Classification_StackOVQs/blob/main/scd.jpg)

# Stack Overflow Text Classification with LSTMs
This project aims to classify Stack Overflow questions into one of four programming languages: Python, CSharp, JavaScript, or Java. We'll leverage the power of Long Short-Term Memory (LSTM) networks to achieve this task.

### Learning Objectives:

- Text Data Loading: Understand techniques for loading and pre-processing text data from datasets like Stack Overflow.
- LSTM Networks: Gain practical experience building and training LSTM models for sequence classification tasks.

- Dataset: The Stack Overflow dataset can be downloaded from: https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz

## Task:

The goal is to create a model that takes a question asked on Stack Overflow as input and predicts the programming language it's related to (Python, CSharp, JavaScript, or Java).

### Steps:
**Data Loading and Preprocessing:**

- Download and extract the Stack Overflow dataset. Check the data quality to decide on appropriate pre-processing steps.
- Load the data, separating questions and their corresponding programming language labels.
- Preprocess the text data by cleaning (e.g., removing punctuation, stop words) and tokenizing it into sequences.
- Convert the text tokens and labels into numerical representations (e.g., one-hot encoding).
- Split the data into training, validation, and test sets.

**Model Building:**

- Construct an LSTM network with an embedding layer to represent text tokens numerically.
- Implement a dense output layer with four units (one for each programming language) and a softmax activation function to predict the language probability distribution.

**Model Training:**
- Compile the model with an appropriate loss function (e.g., Sparse categorical cross-entropy) and optimizer (e.g., Adam).
- Train the model on the training data, monitoring its performance on the validation set to prevent overfitting.   

**Evaluation:**

Evaluate the model's performance on the test set using metrics like accuracy and plot the graphs for traning and testing accuracy and loss.

**Expected Outcomes:**

A trained LSTM model capable of classifying Stack Overflow questions by programming language.
Deeper understanding of text data loading, pre-processing, and LSTM network application for sequence classification.

**Roadmap:**

- Experiment with different hyperparameter settings (e.g., LSTM layer size, number of epochs) to optimize model performance.
- Explore advanced pre-processing techniques like stemming or lemmatization for potential improvements.
- Consider using pre-trained word embeddings (e.g., Word2Vec or GloVe) to enrich the text representation and potentially boost model accuracy.
- To explore tf.data.AUTOTUNE


## Evaluation and Results
![Logo](https://github.com/AKGanesh/Basic_Text_Classification_StackOVQs/blob/main/trvl.png)
![Logo](https://github.com/AKGanesh/Basic_Text_Classification_StackOVQs/blob/main/trvla.png)

## Libraries

**Language:** Python

**Packages:** Pandas, Numpy, Matplotlib, Tensorflow, Keras


## FAQ

#### What is Embedding?
Embedding in the context of natural language processing (NLP) is a technique used to represent words or phrases as dense vectors in a continuous space. These vectors, often referred to as embeddings, capture the semantic meaning and relationships between words.

Key concepts:

- Dense vectors: Embeddings are represented as fixed-length numerical vectors, unlike one-hot encoding which results in sparse vectors.
- Continuous space: The vectors exist in a continuous space, allowing for smooth transitions between similar words.
- Semantic relationships: Embeddings capture the semantic relationships between words, meaning similar words will have similar embeddings.

#### What are common embedding techniques?
- Word2Vec: A popular technique that learns word embeddings by predicting surrounding words in a text corpus.
- GloVe: A technique that learns word embeddings by factoring a co-occurrence matrix.
- FastText: An extension of Word2Vec that learns embeddings for subwords, making it more effective for handling out-of-vocabulary words.
- BERT: BERT (Bidirectional Encoder Representations from Transformers) is a more recent technique that learns contextualized word embeddings. It uses a transformer architecture to model the relationship between words in a sentence.

## Acknowledgements
- https://www.tensorflow.org/tutorials/keras/text_classification
- https://www.sciencedirect.com/science/article/abs/pii/S0164121219302791 (Readme header image)

## Contact

For any queries, please send an email (id on my github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
