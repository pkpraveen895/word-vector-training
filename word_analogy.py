import os
import pickle
import numpy as np
from scipy import spatial

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

#file = open("word_analogy_test.txt", "r")

if loss_model == 'nce':
    outputFile = open("word_analogy_test_predictions_nce.txt","wb")
else:
    outputFile = open("word_analogy_test_predictions_cross_entropy.txt","wb")

file = open("word_analogy_dev.txt", "r")

for i in file:
    words = i.strip().split("||")
    samples = words[0].split(",")
    options = words[1].split(",")
    
    List3 = list()
    List4 = list()
    List6 = list()
    
    for sample in samples:
        List2 = list()
        words = sample.replace("\"", "").split(":")
        e1 = embeddings[dictionary[words[0]]]
        e2 = embeddings[dictionary[words[1]]]
        for j in (e2 - e1):
            List2.append(j)

        List3.append(List2)

    for option in options:
        List5 = list()
        optionwords = option.replace("\"", "").split(":")
        o1 = embeddings[dictionary[optionwords[0]]]
        o2 = embeddings[dictionary[optionwords[1]]]
        List4.append(optionwords)
        for k in (o2 - o1):
            List5.append(k)

        List6.append(List5)

    similarity = 1 - spatial.distance.cdist(np.asarray(List3), np.asarray(List6), 'cosine')
    mean = np.mean(similarity, axis = 1)
    max = np.argmax(mean, axis=0)
    min = np.argmin(mean, axis=0)

    answer = ""
    ques = ""

    if ( (len(List4) - 1) > min and (len(List4) - 1) > max):
        for l in range(len(List4)):
            answer += ques + "\"" + List4[l][0] + ":" + List4[l][1] + "\"" + ques + "\t"
        
        answer += ques + "\"" + List4[min][0] + ":" + List4[min][1] + "\"" + ques + "\t"
        answer += ques + "\"" + List4[max][0] + ":" + List4[max][1] + "\"" + ques + "\n"

        outputFile.write(answer)

print("Prediction file is successfully generated.")
file.close()
outputFile.close()

