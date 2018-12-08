from nltk.corpus import dependency_treebank
from itertools import product
import nltk
import numpy as np

# nltk.download()
TAG = 'tag'
WORD = 'word'
parsed_sents = dependency_treebank.parsed_sents()  # Download all the data
train_set = parsed_sents[:(int(len(parsed_sents) * 0.9))]
test_set = parsed_sents[(int(len(parsed_sents) * 0.9)):]


def feature_function(node1, node2, sentence):
    sentence_vector = {}
    for node_i in sentence.nodes:
        for node_j in sentence.nodes:
            if node_i is not node_j:
                if sentence.nodes[node_i][WORD] == node1[WORD] and sentence.nodes[node_j][WORD] == node2[WORD]:
                    sentence_vector[(sentence.nodes[node_i][WORD], sentence.nodes[node_j][WORD])] = 1
                else:
                    sentence_vector[(sentence.nodes[node_i][WORD], sentence.nodes[node_j][WORD])] = 0
                if sentence.nodes[node_i][TAG] == node1[TAG] and sentence.nodes[node_j][TAG] == node2[TAG]:
                    sentence_vector[(sentence.nodes[node_i][TAG], sentence.nodes[node_j][TAG])] = 1
                else:
                    sentence_vector[(sentence.nodes[node_i][TAG], sentence.nodes[node_j][TAG])] = 0
    return sentence_vector

# first_sent = parsed_sents[1]
# n1 = first_sent.nodes[0]
# n2 = first_sent.nodes[1]
# ret = feature_function(n1, n2, first_sent)
# for kv in ret.items():
#     print(kv)


def distance_features(node1, node2, sentence):
    head_word = node1[WORD]
    dependent_word = node2[WORD]
    head_indices = []
    dependent_indices = []
    for i in range(len(sentence.nodes)):
        if sentence.nodes[i][WORD] == head_word:
            head_indices += [i]
    for i in range(len(sentence.nodes)):
        if sentence.nodes[i][WORD] == dependent_word:
            dependent_indices += [i]
    first, second = sorted(product(head_indices, dependent_indices), key=lambda t: abs(t[0] - t[1]))[0]
    dist = abs(first - second) - 1
    distance_dict = {'distance 0': 0, 'distance 1': 0, 'distance 2': 0, 'distance 3': 0}
    if dist is 0:
        distance_dict['distance 0'] = 1
    elif dist is 1:
        distance_dict['distance 1'] = 1
    elif dist is 2:
        distance_dict['distance 2'] = 1
    else:
        distance_dict['distance 3'] = 1
    return distance_dict


def feature_function_with_distance(node1, node2, sentence):
    feature_vec = feature_function(node1, node2, sentence)
    distance_vec = distance_features(node1, node2, sentence)
    for key, value in distance_vec.items():
        feature_vec[key] = value
    return feature_vec


# sents = dependency_treebank.parsed_sents()
# sentence_1 = sents[0]
# dist_dict = feature_function_with_distance(sentence_1.nodes[5], sentence_1.nodes[1], sentence_1)
# print(dist_dict)
