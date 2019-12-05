import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Given Corpus with different documents
corpus = {1: 'i am student of computer engineering at the university of guilan',
          2: 'i am studying natural language processing right now'}

# Create a bag of words for each documents
BoW = []
# Split each document
for row in corpus:
    BoW.append(corpus[row].split(' '))
    Num_document = row  # use it for making sets
# print(BoW)
# print(Num_document)

# Remove any duplicate words
unique_Words = set(BoW[0]).union(set(BoW[1]))
# print(uniqueWord)

# Create Document-Word matrix
# Sort unique words for better visualization
unique_Words = sorted(unique_Words)
# dict.fromkeys(x=keys, y=value)
Doc1 = dict.fromkeys(unique_Words, 0)
# Count word occurrence in Doc
for word in BoW[0]:
    Doc1[word] += 1

Doc2 = dict.fromkeys(unique_Words, 0)
for word in BoW[1]:
    Doc2[word] += 1

# print("Doc1:", Doc1)
# print("Doc2:", Doc2)
# document_word = pd.DataFrame([Doc1, Doc2])
# print(document_word)
# Using stopwords is highly recommended but It depends on your tasks


# Compute TF
def tf_computation(document, bag_of_words):
    tf_doc = {}
    bow_count = len(bag_of_words)
    # print(bow_count)
    for w, count in document.items():
        tf_doc[w] = float(count / bow_count)
    return tf_doc


tfDoc1 = tf_computation(Doc1, BoW[0])
tfDoc2 = tf_computation(Doc2, BoW[1])
# print(tfDoc1)
# print(tfDoc2)


# Compute IDF
def idf_computation(docs):
    n = len(docs)
    idf_dict = dict.fromkeys(docs[0].keys(), 0)
    for document in docs:
        for w, val in document.items():
            if val > 0:
                idf_dict[w] += 1
    for w, val in idf_dict.items():
        idf_dict[w] = math.log(n/float(val))
    return idf_dict


idf_s = idf_computation([Doc1, Doc2])
# print(idf_s)


def tf_idf_computation(tf, idfs):
    tf_idf = {}
    for w, val in tf.items():
        tf_idf[w] = val * idfs[w]
    return tf_idf


tf_idf_doc1 = tf_idf_computation(tfDoc1, idf_s)
tf_idf_doc2 = tf_idf_computation(tfDoc2, idf_s)
print(tf_idf_doc1)
print(tf_idf_doc2)

# Show in a data frame
data_frame = pd.DataFrame([tf_idf_doc1, tf_idf_doc2])
print(data_frame.head())


# ................. TF-IDF implementation by Sklearn .................

corpus2 = ['i am student of computer engineering at the university of guilan',
           'i am studying natural language processing right now']
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus2)
feature_names = vectorizer.get_feature_names()
# print(x.shape)
output = x.todense()
output_list = output.tolist()
output_df = pd.DataFrame(output_list, columns=feature_names)
print(output_df.head())