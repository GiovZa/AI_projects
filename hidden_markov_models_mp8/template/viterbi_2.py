"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

from collections import defaultdict
import math

def train_model(train_data):
    tag_counts = defaultdict(lambda: 0)
    word_tag_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    hapax_tag_counts = defaultdict(lambda: 0)
    hapax_words = set()

    # Process training data to count occurrences
    for s in train_data:
        prev_t = "s"
        for w, t in s:
            tag_counts[t] += 1
            word_tag_counts[w][t] += 1
            transition_counts[prev_t][t] += 1
            prev_t = t
        transition_counts[prev_t]['/s'] += 1  # End of sentence transition

    # Identify hapax words and count occurrences by tag
    for w, tags in word_tag_counts.items():
        total_count = sum(tags.values())
        if total_count == 1:  # Hapax word
            hapax_words.add(w)
            for t in tags:
                hapax_tag_counts[t] += 1

    total_hapax = sum(hapax_tag_counts.values())  # Total hapax count across all tags

    return tag_counts, word_tag_counts, transition_counts, hapax_tag_counts, total_hapax

# Smoothing constant, adjusted for fine-tuning
alpha = 1e-5

def get_probabilities(tag_counts, word_tag_counts, transition_counts, hapax_tag_counts, total_hapax):
    emis_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
    trans_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
    hapax_probs = {}

    # Calculate emission probabilities
    for w, tags in word_tag_counts.items():
        for t, count in tags.items():
            emis_probs[t][w] = count / tag_counts[t]

    # Calculate transition probabilities
    for prev_t, next_tags in transition_counts.items():
        total_transitions = sum(next_tags.values())
        for next_tag, count in next_tags.items():
            trans_probs[prev_t][next_tag] = count / total_transitions

    # Calculate hapax probabilities (for scaling alpha by tag)
    for t, count in hapax_tag_counts.items():
        hapax_probs[t] = count / total_hapax if total_hapax > 0 else 1 / (len(tag_counts) + 1)

    return emis_probs, trans_probs, hapax_probs

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_counts, word_tag_counts, transition_counts, hapax_tag_counts, total_hapax = train_model(train)
    emis_probs, trans_probs, hapax_probs = get_probabilities(tag_counts, word_tag_counts, transition_counts, hapax_tag_counts, total_hapax)

    result = []
    all_tags = list(tag_counts.keys())

    for s in test:
        viterbi = [{}]
        back_ptr = [{}]

        # Base case (t == 0)
        for t in all_tags:
            # Adjust emission for hapax scaling
            # Using scaled alpha based on hapax tag probabilities
            if total_hapax > 0:
                scaled_alpha = alpha * hapax_probs.get(t, 1 / (len(tag_counts) + 1))  # If tag t is not seen in hapax, treat it as once seen
            else:
                scaled_alpha = alpha
            
            emission = emis_probs[t].get(s[0], scaled_alpha)  # Apply scaled alpha for unseen words
            transition = trans_probs['s'].get(t, alpha)
            viterbi[0][t] = math.log(transition) + math.log(emission) if transition > 0 and emission > 0 else float('-inf')
            back_ptr[0][t] = None

        # Recursive case (t > 0)
        for w in range(1, len(s)):
            viterbi.append({})
            back_ptr.append({})

            for t in all_tags:
                max_prob = float('-inf')
                best_prev_tag = None

                for prev_t in all_tags:
                    prev_prob = viterbi[w - 1][prev_t]
                    trans_prob = trans_probs[prev_t].get(t, alpha)
                    if total_hapax > 0:
                        scaled_alpha = alpha * hapax_probs.get(t, 1 / (len(tag_counts) + 1))
                    else:
                        scaled_alpha = alpha
                    emis_prob = emis_probs[t].get(s[w], scaled_alpha)

                    # Log probabilities to avoid underflow
                    log_trans_prob = math.log(trans_prob) if trans_prob > 0 else float('-inf')
                    log_emis_prob = math.log(emis_prob) if emis_prob > 0 else float('-inf')

                    total_prob = prev_prob + log_trans_prob + log_emis_prob
                    if total_prob > max_prob:
                        max_prob = total_prob
                        best_prev_tag = prev_t

                viterbi[w][t] = max_prob
                back_ptr[w][t] = best_prev_tag

        # Termination step
        max_prob = float('-inf')
        best_last_tag = None

        for t in all_tags:
            final_prob = viterbi[len(s) - 1][t]
            trans_prob = trans_probs[t].get('/s', alpha)
            log_trans_prob = math.log(trans_prob) if trans_prob > 0 else float('-inf')
            terminate_prob = final_prob + log_trans_prob

            if terminate_prob > max_prob:
                max_prob = terminate_prob
                best_last_tag = t

        # Backtrace to find the best path
        best_path = [best_last_tag]
        for w in range(len(s) - 1, 0, -1):
            best_path.insert(0, back_ptr[w][best_path[0]])

        result.append(list(zip(s, best_path)))

    return result
