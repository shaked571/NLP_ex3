from nltk.corpus import dependency_treebank
import nltk
import numpy as np

# nltk.download()
parsed_sents = dependency_treebank.parsed_sents()  # Download all the data
train_set = parsed_sents[:(int(len(parsed_sents) * 0.9))]
test_set = parsed_sents[(int(len(parsed_sents) * 0.9)):]


first_sent = parsed_sents[1]
# for node in first_sent.nodes:
#     print(first_sent.nodes[node]['word'])
#
#
# p1 = ("John")


def feature_function(sentence):
    sentence_vector = {}
    for node_i in sentence.nodes:
        for node_j in sentence.nodes:
            if node_i is not node_j:
                sentence_vector[(sentence.nodes[node_i]['word'], sentence.nodes[node_j]['word'])] = 1
                sentence_vector[(sentence.nodes[node_i]['tag'], sentence.nodes[node_j]['tag'])] = 1

    return sentence_vector
ret = feature_function(first_sent)
for kv in ret:
    print(kv)


