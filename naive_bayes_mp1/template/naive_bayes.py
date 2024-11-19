# naive_bayes.py
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
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
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
    NOTE: Counting pos and neg files in dev folder leads to pos_prior = 4000/5000 = 0.8  and 8000/10000 = 0.8 in the train folder
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.3505, pos_prior=0.8, silently=False):
    print_values(laplace, pos_prior)

    # Training Phase Variable Initialization
    positive_counts = Counter()
    negative_counts = Counter()
    total_positive_words = 0
    total_negative_words = 0
    vocab = set()

    # Training Phase Label Assignment
    POSITIVE_LABEL = 1
    for doc, label in zip(train_set, train_labels):
        if label == POSITIVE_LABEL:
            total_positive_words += len(doc)
            positive_counts.update(doc)
        else:
            total_negative_words += len(doc)
            negative_counts.update(doc)
        vocab.update(doc)

    vocab_size = len(vocab)

    # Assign likelihoods and do constant-time lookups prior to dev phase to save computation time
    positive_likelihoods = compute_likelihood(positive_counts, total_positive_words, vocab_size, laplace)
    negative_likelihoods = compute_likelihood(negative_counts, total_negative_words, vocab_size, laplace)

    # Log to not underflow
    log_prior_pos = math.log(pos_prior)
    log_prior_neg = math.log(1 - pos_prior)

    results = []

    # Development Phase
    for doc in tqdm(dev_set, disable=silently):
        log_prob_pos = log_prior_pos
        log_prob_neg = log_prior_neg

        # Only O(1) look up dictionaries, no recomputations
        for word in doc:
            log_prob_pos += positive_likelihoods.get(word, positive_likelihoods['<UNKNOWN>'])
            log_prob_neg += negative_likelihoods.get(word, negative_likelihoods['<UNKNOWN>'])

        # Assigns label based on which prob is higher, with pos_prior > 0.5 being a tie breaker
        is_pos = assign_label(log_prob_pos, log_prob_neg, pos_prior)
        results.append(is_pos)

    return results

