# Twitter Sentiment Analysis using DistilBERT

This project performs sentiment analysis on Twitter data using the DistilBERT pre-trained transformer model from Hugging Face's Transformers library. The goal is to classify the sentiment of tweets into three categories: Positive, Neutral, and Negative.

## Overview

The project includes the following steps:

1.  **Data Loading and Preprocessing:** Loading Twitter training (`twitter_training.csv`) and validation (`twitter_validation.csv`) datasets, cleaning the text (removing URLs, mentions, hashtags, punctuation, stopwords, and performing lemmatization).
2.  **Model Loading and Tokenization:** Utilizing the DistilBERT tokenizer (`distilbert-base-uncased`) to prepare the text data for the model with a maximum length of 64 tokens.
3.  **Dataset Preparation:** Creating TensorFlow datasets for efficient training and evaluation, including shuffling of the training data and batching.
4.  **Model Training/Fine-tuning:** Fine-tuning the DistilBERT model on the training data using a custom training loop with explicit label handling and tracking of loss and accuracy. Trained for 3 epochs.
5.  **Model Evaluation:** Evaluating the performance of the trained model on the validation dataset, including metrics like loss, accuracy, precision, recall, and F1-score.
6.  **Reporting:** Generating a classification report and a confusion matrix to provide a detailed analysis of the model's predictions.
7.  **Visualization:** Plotting the training and validation loss and accuracy over the epochs.

## Files

* `README.md`: This file, providing an overview of the project.
* `.gitignore`: Specifies intentionally untracked files that Git should ignore.
* `LICENSE`: Contains the license under which the project is distributed.
* `main.py`: The main Python script containing the project code.
* `requirements.txt`: Lists the Python libraries required to run the project.
* `twitter_training.csv`: The training dataset. 
* `twitter_validation.csv`: The validation dataset. 

## Dependencies

The project relies on the following Python libraries:

* transformers
* tensorflow
* pandas
* scikit-learn
* nltk
* matplotlib
* seaborn
* numpy

You can install these dependencies using pip:

```bash
pip install -r requirements.txt

Update README with project details
