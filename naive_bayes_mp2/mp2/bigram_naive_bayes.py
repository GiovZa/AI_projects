# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

def extract_bigrams(doc):
    bigrams = []
    for i in range(len(doc) - 1):
        bigrams.append((doc[i], doc[i + 1]))
    return bigrams

def compute_likelihood(counts=None, total_words=None, vocab_size=None, laplace=1.0):
    likelihoods = {}

    # Calculate the likelihood for each known word prior to nested loop dev phase
    for w in counts:
        likelihoods[w] = math.log((counts[w] + laplace) / (total_words + laplace * vocab_size))
    
    # Set default value for unknown words prior to nested loop dev phase
    default_likelihood = math.log(laplace / (total_words + laplace * vocab_size))
    likelihoods['<UNKNOWN>'] = default_likelihood

    return likelihoods


def assign_label(log_prob_pos, log_prob_neg, pos_prior):
    if log_prob_pos > log_prob_neg:
        return 1
    elif log_prob_pos < log_prob_neg:
        return 0
    else:
        # If the probs are equal, go with whichever label has higher prior
        return 1 if pos_prior > 0.5 else 0

"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    # Training phase for unigrams (same as in naive_bayes)
    positive_counts = Counter()
    negative_counts = Counter()
    bigram_pos_counts = Counter()
    bigram_neg_counts = Counter()
    total_positive_words = 0
    total_negative_words = 0
    total_positive_bigrams = 0
    total_negative_bigrams = 0
    vocab = set()
    bigram_vocab = set()

    POSITIVE_LABEL = 1

    for doc, label in zip(train_set, train_labels):
        # Unigram counting
        if label == POSITIVE_LABEL:
            total_positive_words += len(doc)
            positive_counts.update(doc)
        else:
            total_negative_words += len(doc)
            negative_counts.update(doc)
        vocab.update(doc)
        
        # Bigram counting
        bigrams = extract_bigrams(doc)
        if label == POSITIVE_LABEL:
            total_positive_bigrams += len(bigrams)
            bigram_pos_counts.update(bigrams)
        else:
            total_negative_bigrams += len(bigrams)
            bigram_neg_counts.update(bigrams)
        bigram_vocab.update(bigrams)

    vocab_size = len(vocab)
    bigram_vocab_size = len(bigram_vocab)

    # Compute unigram likelihoods
    positive_likelihoods = compute_likelihood(positive_counts, total_positive_words, vocab_size, unigram_laplace)
    negative_likelihoods = compute_likelihood(negative_counts, total_negative_words, vocab_size, unigram_laplace)

    # Compute bigram likelihoods
    bigram_positive_likelihoods = compute_likelihood(bigram_pos_counts, total_positive_bigrams, bigram_vocab_size, bigram_laplace)
    bigram_negative_likelihoods = compute_likelihood(bigram_neg_counts, total_negative_bigrams, bigram_vocab_size, bigram_laplace)

    # Log priors
    log_prior_pos = math.log(pos_prior)
    log_prior_neg = math.log(1 - pos_prior)

    results = []

    # Development phase (predicting on the dev_set)
    for doc in tqdm(dev_set, disable=silently):
        log_prob_pos = log_prior_pos
        log_prob_neg = log_prior_neg

        # Unigram contribution
        for word in doc:
            log_prob_pos += positive_likelihoods.get(word, positive_likelihoods['<UNKNOWN>'])
            log_prob_neg += negative_likelihoods.get(word, negative_likelihoods['<UNKNOWN>'])

        # Bigram contribution
        bigrams = extract_bigrams(doc)
        for bigram in bigrams:
            log_prob_pos += bigram_lambda * bigram_positive_likelihoods.get(bigram, bigram_positive_likelihoods['<UNKNOWN>'])
            log_prob_neg += bigram_lambda * bigram_negative_likelihoods.get(bigram, bigram_negative_likelihoods['<UNKNOWN>'])

        # Assign label based on mixed probabilities
        is_pos = assign_label(log_prob_pos, log_prob_neg, pos_prior)
        results.append(is_pos)

    return results



