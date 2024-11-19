from collections import defaultdict
import math

def get_classification(w):
    if w[0].isdigit() and w[-1].isdigit():
        return "NUM"
    elif len(w) < 4:
        return "V_SHORT"
    elif 3 < len(w) < 10:
        if w.endswith('s'):
            return "SHORT_S"
        else:
            return "SHORT_NO_S"
    else:  # long, i.e., at least 10 characters long
        if w.endswith('s'):
            return "LONG_S"
        else:
            return "LONG_NO_S"

def train_model(train_data):
    tag_counts = defaultdict(lambda: 0)
    word_tag_counts = defaultdict(lambda: defaultdict(lambda: 0))
    trans_counts = defaultdict(lambda: defaultdict(lambda: 0))
    hapax_tag_type_counts = defaultdict(lambda: defaultdict(lambda: 0))
    hapax_words = set()
    
    for s in train_data:
        prev_tag = "s"
        for w, t in s:
            tag_counts[t] += 1
            word_tag_counts[w][t] += 1
            trans_counts[prev_tag][t] += 1
            prev_tag = t
        trans_counts[prev_tag]['/s'] += 1
    
    for w, tags in word_tag_counts.items():
        total_count = sum(tags.values())
        if total_count == 1:
            hapax_words.add(w)
            word_type = get_classification(w)
            for t in tags:
                hapax_tag_type_counts[t][word_type] += 1
    
    total_hapax = sum(sum(type_counts.values()) for type_counts in hapax_tag_type_counts.values())
    hapax_tag_type_prob = {}

    for t in tag_counts:
        hapax_prob_for_tag = {}

        for word_type in hapax_tag_type_counts[t]:
            hapax_count = hapax_tag_type_counts[t][word_type]
            hapax_prob = hapax_count / total_hapax if total_hapax > 0 else 1 / len(tag_counts)

            hapax_prob_for_tag[word_type] = hapax_prob

        hapax_tag_type_prob[t] = hapax_prob_for_tag
    
    return tag_counts, word_tag_counts, trans_counts, hapax_tag_type_prob

alpha = 1e-6
def get_probabilities(tag_counts, word_tag_counts, transition_counts, hapax_tag_type_prob):
    emission_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
    transition_probs = defaultdict(lambda: defaultdict(lambda: 0.0))

    for w, tags in word_tag_counts.items():
        for t, count in tags.items():
            emission_probs[t][w] = (count + alpha) / (tag_counts[t] + len(word_tag_counts) * alpha)

    for prev_t, next_tags in transition_counts.items():
        total_count = sum(next_tags.values())
        for next_tag, count in next_tags.items():
            transition_probs[prev_t][next_tag] = count / total_count

    return emission_probs, transition_probs

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word, tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    tag_counts, word_tag_counts, transition_counts, hapax_tag_type_prob = train_model(train)
    emis_probs, trans_probs, alpha = get_probabilities(tag_counts, word_tag_counts, transition_counts, hapax_tag_type_prob)

    result = []
    all_tags = list(tag_counts.keys())

    for s in test:
        viterbi = [{}]
        back_ptr = [{}]

        # Base case: t == 0
        for t in all_tags:
            word_type = get_classification(s[0])
            emission = emis_probs[t].get(s[0], alpha * hapax_tag_type_prob[t].get(word_type, alpha / len(hapax_tag_type_prob)))
            transition = trans_probs['s'].get(t, 0)
            viterbi[0][t] = math.log(transition) + math.log(emission) if transition > 0 and emission > 0 else float('-inf')
            back_ptr[0][t] = None

        # All other cases
        for w in range(1, len(s)):
            viterbi.append({})
            back_ptr.append({})

            for t in all_tags:
                word_type = get_classification(s[w])
                max_prob = float('-inf')
                best_prev_tag = None  

                for prev_t in all_tags:
                    viterbi_value = viterbi[w - 1][prev_t]
                    
                    trans_prob = trans_probs[prev_t].get(t, alpha)
                    log_trans_prob = math.log(trans_prob) if trans_prob > 0 else float('-inf')

                    emis_prob = emis_probs[t].get(s[w], alpha * hapax_tag_type_prob[t].get(word_type, alpha / len(hapax_tag_type_prob)))
                    log_emis_prob = math.log(emis_prob) if emis_prob > 0 else float('-inf')

                    total_prob = viterbi_value + log_trans_prob + log_emis_prob
                    if total_prob > max_prob:
                        max_prob = total_prob
                        best_prev_tag = prev_t

                viterbi[w][t] = max_prob
                back_ptr[w][t] = best_prev_tag

        # Terminate
        viterbi_value = max(viterbi[len(s) - 1][tag] for tag in all_tags)
        best_end_tag = max((viterbi_value + math.log(trans_probs[tag].get('/s', alpha)), tag) for tag in all_tags)[1]

        # Backtrace
        best_path = [best_end_tag]
        for w in range(len(s) - 1, 0, -1):
            best_path.insert(0, back_ptr[w][best_path[0]])

        result.append(list(zip(s, best_path)))

    return result
