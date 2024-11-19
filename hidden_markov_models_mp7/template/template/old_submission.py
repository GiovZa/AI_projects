"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
        
    ''' 1. Count occurrences of tags, tag pairs, tag/word pairs. '''
    tags = defaultdict(lambda: 0)
    for s in sentences:
        prev_tag = "START"
        for w, t in s:
            init_prob[t] += (1 if prev_tag == "START" else 0)  # Only START word can have init prob
            trans_prob[prev_tag][t] += (1 if prev_tag != "START" else 0) # Can't have transition if word is START word
            emit_prob[t][w] += 1
            tags[t] += 1
            prev_tag = t
        trans_prob[prev_tag]["END"] += 1

    ''' 2. Compute smoothed probabilities. '''
    # Normalize init_prob
    num_sentences = len(sentences)
    for t in init_prob:
        init_prob[t] = (init_prob[t] + epsilon_for_pt) / (num_sentences + len(init_prob) * epsilon_for_pt)
    # Normalize trans_prob
    for t in trans_prob:
        total_transitions = sum(trans_prob[t].values())
        for next_t in trans_prob[t]:
            trans_prob[t][next_t] = (trans_prob[t][next_t] + epsilon_for_pt) / (total_transitions + len(trans_prob[t]) * epsilon_for_pt)
    # Normalize emit_prob
    for t in emit_prob:
        total_emissions = tags[t]
        for w in emit_prob[t]:
            emit_prob[t][w] = (emit_prob[t][w] + emit_epsilon) / (total_emissions + len(emit_prob[t]) * emit_epsilon)
        emit_prob[t]["UNKNOWN"] = emit_epsilon / (total_emissions + emit_epsilon * (len(emit_prob[t]) + 1))
        some_threshold = 40  
        scaling_factor = 0.35

        if total_emissions < some_threshold:
            emit_prob[t]["UNKNOWN"] = emit_prob[t]["UNKNOWN"] * scaling_factor
            
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    for curr_tag, emit_dict in emit_prob.items():
        highest_prob = float('-inf')
        best_prev_tag = None
        
        for prev_tag, prev_p in prev_prob.items():
            ''' 3. Take the log of each probability. '''
            trans_p = math.log(trans_prob[prev_tag].get(curr_tag, epsilon_for_pt))
            emit_p = math.log(emit_dict.get(word, emit_epsilon))
            prob = prev_p + trans_p + emit_p

            # Update highest_prob and best_prev_tag
            if prob > highest_prob:
                highest_prob = prob
                best_prev_tag = prev_tag

        ''' 4. Construct the trellis. Notice that for each tag/time pair, you must store not only the probability of the best path but also a pointer to the previous tag/time pair in that path. '''
        log_prob[curr_tag] = highest_prob
        predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.

        ''' 5. Return the best path through the trellis by backtracing. '''
        best_prev_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[best_prev_tag]

        word_tag_pairs = list(zip(sentence, best_tag_seq))
        predicts.append(word_tag_pairs)
    return predicts