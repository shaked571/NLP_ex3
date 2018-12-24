from nltk.corpus import dependency_treebank
from itertools import product, permutations
from random import shuffle
import numpy as np
import nltk
import copy
import json
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


def calculate_feature_function_mst(mst, sentence, with_distance):
    sum_of_vectors = {}
    for arc in mst.values():
        node1 = sentence.nodes[arc[2]]
        node2 = sentence.nodes[arc[0]]
        if with_distance:
            current_feature_vector = feature_function_with_distance(node1, node2, sentence)
        else:
            current_feature_vector = feature_function(node1, node2, sentence)
        for key in current_feature_vector:
            if key in sum_of_vectors:
                sum_of_vectors[key] += current_feature_vector[key]
            else:
                sum_of_vectors[key] = current_feature_vector[key]
    return sum_of_vectors

def sum_feature_vectors(feature_function_list):
    sum_of_vectors = {}
    for current_feature_vector in feature_function_list:
        for key in current_feature_vector:
            if key in sum_of_vectors:
                sum_of_vectors[key] += current_feature_vector[key]
            else:
                sum_of_vectors[key] = current_feature_vector[key]
    return sum_of_vectors


def recursive_graph(node, sentence, feature_function_node_tail_list, with_distance):
    if len(node['deps']) == 0:
        return
    for node_index in node['deps']['']:
        if with_distance:
            feature_function_node_tail_list.append(feature_function_with_distance(node, sentence.nodes[node_index], sentence))
        else:
            feature_function_node_tail_list.append(feature_function(node, sentence.nodes[node_index], sentence))
        recursive_graph(sentence.nodes[node_index], sentence, feature_function_node_tail_list, with_distance)
    return

def calculate_gold_tree(sentence, with_distance):
    feature_function_node_tail_list = []
    node1 = sentence.nodes[0]
    for node2_index in sentence.nodes[0]['deps']['ROOT']:
        if with_distance:
            feature_function_node_tail_list.append(feature_function_with_distance(node1, sentence.nodes[node2_index], sentence))
        else:
            feature_function_node_tail_list.append(feature_function(node1, sentence.nodes[node2_index], sentence))
        recursive_graph(sentence.nodes[node2_index], sentence, feature_function_node_tail_list, with_distance)
    return sum_feature_vectors(feature_function_node_tail_list)

def recursive_on_tree(node, sentence, arcs_dict):
    if len(node['deps']) == 0:
        return
    for node_index in node['deps']['']:
        index = sentence.nodes[node_index][INDEX]
        arcs_dict[index] = Chu_Liu_Edmonds_algorithm.Arc(tail=sentence.nodes[node_index][INDEX], weight=0, head=node[INDEX])
        recursive_on_tree(sentence.nodes[node_index], sentence, arcs_dict)
    return

def get_arcs_from_sentence(sentence):
    arcs_dict = {}
    node1 = sentence.nodes[0]
    for node2_index in sentence.nodes[0]['deps']['ROOT']:
        index = sentence.nodes[node2_index][INDEX]
        arcs_dict[index] = Chu_Liu_Edmonds_algorithm.Arc(tail=sentence.nodes[node2_index][INDEX], weight=0, head=node1[INDEX])
        recursive_on_tree(sentence.nodes[node2_index], sentence, arcs_dict)
    return arcs_dict

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



def perceptron_algorithm(train_corpus, with_distance):
    ####################################################################################################################
    # INIT
    ####################################################################################################################
    print("init perceptron_algorithm")
    # labaled_data_y = test_corus
    list_of_weight_vectors_w = list()
    weight_vector_w = {}
    epochs = 2
    train_corpus_size = str(len(train_corpus))
    print("Going over all the sentences")
    for i in range(epochs):
        print("starting " + str(i + 1) + " iteration over the examples")
        shuffle(train_corpus)
        counter = 0
        for sentence in train_corpus:
            arcs_vector = []
            for pair in permutations(sentence.nodes, 2):
                if with_distance:
                    curr_weight = calculate_score_feature(feature_function_with_distance(sentence.nodes[pair[0]], sentence.nodes[pair[1]], sentence), weight_vector_w)
                else:
                    curr_weight = calculate_score_feature(feature_function(sentence.nodes[pair[0]], sentence.nodes[pair[1]], sentence), weight_vector_w)
                arc = Chu_Liu_Edmonds_algorithm.Arc(tail=pair[0], weight=curr_weight * -1, head=pair[1])
                arcs_vector.append(arc)
            # next phase
            print('Creating a new mst ' + str(counter) + " from: " + train_corpus_size)
            counter += 1
            mst = Chu_Liu_Edmonds_algorithm.min_spanning_arborescence(arcs_vector, SINK)
            new_mst = calculate_feature_function_mst(mst, sentence, with_distance)
            gold_mst = calculate_gold_tree(sentence, with_distance)
            result = subtract_feature_vectors(gold_mst, new_mst)
            weight_vector_w = sum_feature_vectors([weight_vector_w, result])
            list_of_weight_vectors_w.append(copy.deepcopy(weight_vector_w))
    final_w = get_final_weight_vector(list_of_weight_vectors_w, epochs, len(train_corpus))
    return final_w


def get_error_gold_vs_result(gold_dict, mst):
    correct = 0
    mst_arcs = mst.values()
    for mst_arc in mst_arcs:
        if Chu_Liu_Edmonds_algorithm.Arc(tail=mst_arc.tail, weight=0, head=mst_arc.head) in gold_dict.values():
            correct += 1
    return correct


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
        train_set = parsed_sents[:(int(len(parsed_sents) * 0.3))]
        test_set = parsed_sents[(int(len(parsed_sents) * 0.95)):]
    except:
        print("couldn't get the data")
        exit(1)

    print("Finished Getting data")


    ####################################################################################################################
    #  Calculate the error rate using the feature vector
    ####################################################################################################################
    print("Starting perceptron algorithm ")
    weight_vector_w = perceptron_algorithm(train_set, with_distance=False)
    print("finish to calculate the weight vector")
    error_count = []
    counter = 0
    test_set_size = str(len(test_set))
    for sentence in test_set:
        print("sen num "+str(counter) + " from : " + test_set_size)

        counter += 1
        arcs_vector = get_arcs_vector(sentence, weight_vector_w, with_distance=False)
        mst = Chu_Liu_Edmonds_algorithm.min_spanning_arborescence(arcs_vector, SINK)
        arcs_dict = get_arcs_from_sentence(sentence)
        current_error_rate = get_error_gold_vs_result(arcs_dict, mst) / len(sentence.nodes)
        error_count.append(current_error_rate)
    avg = np.average(error_count)
    print("the avg without distance is: " + str(avg), file=open("output.txt", "a"))

    f = open('result.txt', 'w+')
    f.write(json.dumps(weight_vector_w))
    weight_vector_w = None
    ####################################################################################################################
    #  Calculate the error rate using the feature and distance vectors
    ####################################################################################################################
    print("Starting perceptron algorithm ")
    weight_vector_w = perceptron_algorithm(train_set, with_distance=True)
    print("finish to calculate the weight vector")
    error_count = []
    counter = 0
    for sentence in test_set:
        print("sen num "+str(counter) + " from : " + test_set_size)
        counter += 1
        arcs_vector = get_arcs_vector(sentence, weight_vector_w, with_distance=True)
        mst = Chu_Liu_Edmonds_algorithm.min_spanning_arborescence(arcs_vector, SINK)
        arcs_dict = get_arcs_from_sentence(sentence)
        current_error_rate = get_error_gold_vs_result(arcs_dict, mst) / len(sentence.nodes)
        error_count.append(current_error_rate)
    avg = np.average(error_count)
    print("the avg with distance is: " + str(avg), file=open("output.txt", "a"))
    try:
        f = open('result2.txt', 'w+')
        f.write(json.dumps(weight_vector_w))
    # the part in the try wasnt check good
    except:
        print("coudlnt save the vector")

def get_arcs_vector(sentence, weight_vector_w, with_distance):
    arcs_vector = []
    for pair in permutations(sentence.nodes, 2):
        if with_distance:
            curr_weight = calculate_score_feature(
                feature_function_with_distance(sentence.nodes[pair[0]], sentence.nodes[pair[1]], sentence), weight_vector_w)
        else:
            curr_weight = calculate_score_feature(
                feature_function(sentence.nodes[pair[0]], sentence.nodes[pair[1]], sentence), weight_vector_w)
        arc = Chu_Liu_Edmonds_algorithm.Arc(tail=pair[1], weight=curr_weight * -1, head=pair[0])
        arcs_vector.append(arc)
    return arcs_vector


if __name__ == '__main__':
    main()
