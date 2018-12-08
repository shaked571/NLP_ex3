from nltk.corpus import dependency_treebank
import nltk
import numpy as np

# nltk.download()
TAG = 'tag'
WORD = 'word'
parsed_sents = dependency_treebank.parsed_sents()  # Download all the data
train_set = parsed_sents[:(int(len(parsed_sents) * 0.9))]
test_set = parsed_sents[(int(len(parsed_sents) * 0.9)):]


first_sent = parsed_sents[1]
# for node in first_sent.nodes:
#     print(first_sent.nodes[node]['word'])
#
#
# p1 = ("John")


def feature_function(node1, node2, sentence):
    sentence_vector = {}
    for node_i in sentence.nodes:
        for node_j in sentence.nodes:
            if node_i is not node_j:
                if sentence.nodes[node_i][WORD] is node1[WORD] and sentence.nodes[node_j][WORD] is node2[WORD]:
                    sentence_vector[(sentence.nodes[node_i][WORD], sentence.nodes[node_j][WORD])] = 1
                else:
                    sentence_vector[(sentence.nodes[node_i][WORD], sentence.nodes[node_j][WORD])] = 0
                if sentence.nodes[node_i][TAG] is node1[TAG] and sentence.nodes[node_j][TAG] is node2[TAG]:
                    sentence_vector[(sentence.nodes[node_i][TAG], sentence.nodes[node_j][TAG])] = 1
                else:
                    sentence_vector[(sentence.nodes[node_i][TAG], sentence.nodes[node_j][TAG])] = 0
    return sentence_vector

n1 = first_sent.nodes[0]
n2 = first_sent.nodes[1]
ret = feature_function(n1, n2, first_sent)
for kv in ret:
    print(kv)


