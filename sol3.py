from nltk.corpus import dependency_treebank
from itertools import product, permutations
from random import shuffle
import numpy as np
import nltk
import copy

SINK = 0

INDEX = 'address'
TAG = 'tag'
WORD = 'word'
import Chu_Liu_Edmonds_algorithm


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


def calculate_score_feature(feature_vector, weight_vector_w):
    sum_of_product = 0
    for key in feature_vector:
        if key in weight_vector_w:
            sum_of_product += weight_vector_w[key] * feature_vector[key]
    return sum_of_product


def calculate_feature_function_mst(mst, sentence):
    sum_of_vectors = {}
    for arc in mst.values():
        node1 = sentence.nodes[arc[2]]
        node2 = sentence.nodes[arc[0]]
        current_feature_vector = feature_function(node1, node2, sentence)
        for key in current_feature_vector:
            if key in sum_of_vectors:
                sum_of_vectors[key] += current_feature_vector[key]
            else:
                sum_of_vectors[key] = current_feature_vector[key]
    return sum_of_vectors


def recursive_graph(node, sentence, feature_function_node_tail_list):
    if len(node['deps']) == 0:
        return
    for node_index in node['deps']['']:
        feature_function_node_tail_list.append(feature_function(node, sentence.nodes[node_index], sentence))
        recursive_graph(sentence.nodes[node_index], sentence, feature_function_node_tail_list)
    return


def sum_feature_vectors(feature_function_list):
    sum_of_vectors = {}
    for current_feature_vector in feature_function_list:
        for key in current_feature_vector:
            if key in sum_of_vectors:
                sum_of_vectors[key] += current_feature_vector[key]
            else:
                sum_of_vectors[key] = current_feature_vector[key]
    return sum_of_vectors


def calculate_gold_tree(sentence):
    feature_function_node_tail_list = []
    node1 = sentence.nodes[0]
    for node2_index in sentence.nodes[0]['deps']['ROOT']:
        feature_function_node_tail_list.append(feature_function(node1, sentence.nodes[node2_index], sentence))
        recursive_graph(sentence.nodes[node2_index], sentence, feature_function_node_tail_list)
    return sum_feature_vectors(feature_function_node_tail_list)


def subtract_feature_vectors(vector1, vector2):
    vector1_copy = copy.deepcopy(vector1)
    for key in vector1_copy:
        if key in vector2:
            vector1_copy[key] -= vector2[key]
        else:
            vector1_copy[key] = vector2[key] * -1
    return vector1_copy


def get_final_weight_vector(list_of_weight_vectors_w, epochs, sentence_num):
    final_w = {}
    for w_vec in list_of_weight_vectors_w:
        for key in w_vec:
            if key in final_w:
                final_w[key] += w_vec[key]
            else:
                final_w[key] = w_vec[key]
    for key in final_w:
        final_w[key] /= (epochs * sentence_num)
    return final_w



def perceptron_algorithm(train_corpus):
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    print("init perceptron_algorithm")
    # labaled_data_y = test_corus
    list_of_weight_vectors_w = list()
    weight_vector_w = {}
    epochs = 2

    print("Going over all the sentences")
    for i in range(epochs):
        print("starting " + str(i + 1) + " iteration over the examples")
        shuffle(train_corpus)
        counter = 0
        for sentence in train_corpus:
            arcs_vector = []
            for pair in permutations(sentence.nodes, 2):
                curr_weight = calculate_score_feature(feature_function(sentence.nodes[pair[0]], sentence.nodes[pair[1]], sentence), weight_vector_w)
                arc = Chu_Liu_Edmonds_algorithm.Arc(pair[0], curr_weight * -1, pair[1]) # TODO verify index and pair[0] is the same
                arcs_vector.append(arc)
            # next phase
            print('Creating a new mst ' + str(counter))
            counter += 1
            mst = Chu_Liu_Edmonds_algorithm.min_spanning_arborescence(arcs_vector, SINK)
            new_mst = calculate_feature_function_mst(mst, sentence)
            gold_mst = calculate_gold_tree(sentence)
            result = subtract_feature_vectors(gold_mst, new_mst)
            weight_vector_w = sum_feature_vectors([weight_vector_w, result])
            list_of_weight_vectors_w.append(copy.deepcopy(weight_vector_w))
    final_w = get_final_weight_vector(list_of_weight_vectors_w, epochs, len(train_corpus))
    return final_w





        # for t in range(epochs):
    #     for i, x in enumerate(train_data):
    #         if (np.dot(train_data[i], weight_vector_w) * labaled_data_y[i]) <= 0:
    #             weight_vector_w = weight_vector_w + learning_rate * train_data[i] * labaled_data_y[i]



def main():

    ####################################################################################################################
    # Get the data
    ####################################################################################################################
    print("Getting data")
    train_set = []
    test_set = []
    try:
        # nltk.download()
        parsed_sents = dependency_treebank.parsed_sents()  # Download all the data
        train_set = parsed_sents[:(int(len(parsed_sents) * 0.005))]
        test_set = parsed_sents[(int(len(parsed_sents) * 0.9)):]
    except:
        print("couldn't get the data")
        exit(1)

    print("Finished Getting data")

    ####################################################################################################################
    #  Calculate the vectors using the feature and distance vectors
    ####################################################################################################################
    print("Starting perceptron algorithm ")
    weight_vector = perceptron_algorithm(train_set)
    print("finish calculating the weight vector")
    for sentence in test_set:



    # dest_dict = feature_function_with_distance()




if __name__ == '__main__':
    main()
# first_sent = parsed_sents[1]
# n1 = first_sent.nodes[0]
# n2 = first_sent.nodes[1]
# ret = feature_function(n1, n2, first_sent)
# for kv in ret.items():
#     print(kv)


# sents = dependency_treebank.parsed_sents()
# sentence_1 = sents[0]
# dist_dict = feature_function_with_distance(sentence_1.nodes[5], sentence_1.nodes[1], sentence_1)
# print(dist_dict)
