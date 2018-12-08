from nltk.corpus import dependency_treebank
import nltk

# nltk.download()
parsed_sents = dependency_treebank.parsed_sents()  # Download all the data
train_set = parsed_sents[:(int(len(parsed_sents) * 0.9))]
test_set = parsed_sents[(int(len(parsed_sents) * 0.9)):]


# first_sent = parsed_sents[1]
# for node in first_sent.nodes:
#     print(first_sent.nodes[node]['word'])

