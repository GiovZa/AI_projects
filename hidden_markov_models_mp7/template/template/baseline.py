"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict, Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
        
    # 1. For each word w, it counts how many times w occurs with each tag in the training data
    word_tags = defaultdict(lambda: defaultdict(lambda: 0))
    tags = defaultdict(lambda: 0)
    for sentence in train:
        for w, t in sentence:
            word_tags[w][t] += 1
            tags[t] += 1

    # 2. Find the tag seen the most oftern
    most_frequent_tag = max(tags, key=tags.get)
    
    # 3. Helper function for tagging
    def tag_word(word):
        if word in word_tags:
            # Find the most common tag assigned to known word
            most_frequent_tag_for_word = max(word_tags[word], key=word_tags[word].get)
            return (word, most_frequent_tag_for_word)
        else:
            # Assign the most overall common tag for unknown words
            return (word, most_frequent_tag)
        
    # 4. Tag the test data
    tagged_sentences = []
    for sentence in test:
        tagged_sentences.append(list(map(tag_word, sentence)))

    return tagged_sentences