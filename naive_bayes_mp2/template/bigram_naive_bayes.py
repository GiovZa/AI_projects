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

"""
combine uni and bi probabilities to create a mixture model
"""
def mixture_model(log_prob_pos, log_prob_neg, bigram_log_prob_pos, bigram_log_prob_neg, bigram_lambda):
    combined_log_prob_pos = (1 - bigram_lambda) * log_prob_pos + bigram_lambda * bigram_log_prob_pos
    combined_log_prob_neg = (1 - bigram_lambda) * log_prob_neg + bigram_lambda * bigram_log_prob_neg
    return combined_log_prob_pos, combined_log_prob_neg

def compute_bigram_likelihood(counts=None, total_bigrams=None, vocab_size=None, laplace=1.0):
    likelihoods = {}
    
    # Calculate the likelihood for each known pair of words prior to nested loop dev phase
    for b in counts:
        likelihoods[b] = math.log((counts[b] + laplace) / (total_bigrams + laplace * vocab_size))
    
    # Set default value for unknown word pairs prior to nested loop dev phase
    default_likelihood = math.log(laplace / (total_bigrams + laplace * vocab_size))
    likelihoods['<UNKNOWN_BIGRAM>'] = default_likelihood
    
    return likelihoods

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
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.3, pos_prior=0.75, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    # Training Phase Variable Initialization
    positive_counts = Counter()
    negative_counts = Counter()
    positive_bigram_counts = Counter()
    negative_bigram_counts = Counter()
    total_positive_words = 0
    total_negative_words = 0
    total_positive_bigrams = 0
    total_negative_bigrams = 0
    vocab = set()
    bigram_vocab = set()

    # Training Phase Label Assignment
    POSITIVE_LABEL = 1
    for doc, label in zip(train_set, train_labels):
        bigrams = [(doc[i], doc[i+1]) for i in range(len(doc)-1)]
        
        if label == POSITIVE_LABEL:
            total_positive_words += len(doc)
            total_positive_bigrams += len(bigrams)
            positive_counts.update(doc)
            positive_bigram_counts.update(bigrams)
        else:
            total_negative_words += len(doc)
            total_negative_bigrams += len(bigrams)
            negative_counts.update(doc)
            negative_bigram_counts.update(bigrams)

        vocab.update(doc)
        bigram_vocab.update(bigrams)

    vocab_size = len(vocab)
    bigram_vocab_size = len(bigram_vocab)

    # Assign likelihoods and do constant-time lookups prior to dev phase to save computation time
    positive_likelihoods = compute_likelihood(positive_counts, total_positive_words, vocab_size, unigram_laplace)
    negative_likelihoods = compute_likelihood(negative_counts, total_negative_words, vocab_size, unigram_laplace)
    positive_bigram_likelihoods = compute_bigram_likelihood(positive_bigram_counts, total_positive_bigrams, bigram_vocab_size, bigram_laplace)
    negative_bigram_likelihoods = compute_bigram_likelihood(negative_bigram_counts, total_negative_bigrams, bigram_vocab_size, bigram_laplace)

    # Log to not underflow
    log_prior_pos = math.log(pos_prior)
    log_prior_neg = math.log(1 - pos_prior)

    results = []

    # Development Phase
    for doc in tqdm(dev_set, disable=silently):
        bigrams = [(doc[i], doc[i+1]) for i in range(len(doc)-1)]
        log_prob_pos = log_prior_pos
        log_prob_neg = log_prior_neg

        # Unigram model with O(1) look up dictionaries, no recomputations
        for word in doc:
            log_prob_pos += positive_likelihoods.get(word, positive_likelihoods['<UNKNOWN>'])
            log_prob_neg += negative_likelihoods.get(word, negative_likelihoods['<UNKNOWN>'])

        # Bigram model with O(1) look up dictionaries, no recomputations
        bigram_log_prob_pos = log_prior_pos
        bigram_log_prob_neg = log_prior_neg
        for bigram in bigrams:
            bigram_log_prob_pos += positive_bigram_likelihoods.get(bigram, positive_bigram_likelihoods['<UNKNOWN_BIGRAM>'])
            bigram_log_prob_neg += negative_bigram_likelihoods.get(bigram, negative_bigram_likelihoods['<UNKNOWN_BIGRAM>'])

        # Mixture model: Combine unigram and bigram models using bigram_lambda
        combined_log_prob_pos, combined_log_prob_neg = mixture_model(log_prob_pos, log_prob_neg, bigram_log_prob_pos, bigram_log_prob_neg, bigram_lambda)

        # Assigns label based on which prob is higher, with pos_prior > 0.5 being a tie breaker
        is_pos = assign_label(combined_log_prob_pos, combined_log_prob_neg, pos_prior)
        results.append(is_pos)

    return results