import streamlit as st
import numpy as np
import pandas as pd
import keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine, cdist


df_svd = pd.read_csv("models/lsa.csv")
df_svd = df_svd.set_index("words")
vocab = list(df_svd.index)
<<<<<<< HEAD
fasttext = Word2Vec.load('models/fasttext.model')
=======

# Glove
from gensim.models import KeyedVectors
glove = KeyedVectors.load('models/glove-wiki-gigaword-300.model')
>>>>>>> 32da833 (Added pretrained GloVe model)

df = pd.read_csv('bbc-text.csv')
articles = list(df['text'])

sentences = []

for i in articles[:200]:
    sentences += i.split('.')

# Remove sentences with fewer than 3 words
corpus = [sentence for sentence in sentences if sentence.count(" ") >= 12]

# Remove punctuation in text and fit tokenizer on entire corpus
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'+"'")
tokenizer.fit_on_texts(corpus)

# Convert text to sequence of integer values
corpus = tokenizer.texts_to_sequences(corpus)
n_samples = sum(len(s) for s in corpus) # Total number of words in the corpus
V = len(tokenizer.word_index) + 1 # Total number of unique words in the corpus



def get_skipgram_embeddings():
    word_vectors = {}
    i=0
    with open("models/vectors_skipgram_300.txt", "r", encoding="utf-8") as file:
        for line in file:
            i+=1
            if i == 1:
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector
    return word_vectors


def get_cbow_embeddings():
    word_vectors = {}
    i=0
    with open("models/vectors_cbow_300.txt", "r", encoding="utf-8") as file:
        for line in file:
            i+=1
            if i == 1:
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector
    return word_vectors


def get_fasttext_embeddings():
    fasttext_model = Word2Vec.load('models/fasttext.model')
    word_vectors = fasttext_model.wv
    return word_vectors

def get_lsa_embeddings():
    word_vectors = {}
    i=0
    with open("models/vectors_lsa_300.txt", "r", encoding="utf-8") as file:
        for line in file:
            i+=1
            if i == 1:
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector
    return word_vectors


def get_k_nearest(k, target_word, word_vectors, model):
    if target_word not in word_vectors:
        st.write(f"'{input_word}' is not present in the vocabulary.")
        return -1

    # for fasttext
    if model == 'fasttext':
        embedding_vector = word_vectors[target_word]
        similar_words = fasttext_model.wv.similar_by_word(target_word)

        st.write(f"\nThe {k} nearest words to '{target_word}' are: ")
        nearest_words = [(word,e) for word, e in similar_words[:k]]
        for i in (nearest_words):
            st.write(i)
        return nearest_words        
        
    # for skipgram and cbow
    if model in ['skipgram', 'cbow', 'lsa']: 
        # Calculate cosine similarities with all words in the vocabulary
        similarities = {}
        target_vector = word_vectors[target_word]
        for word, vector in word_vectors.items():
            if word != target_word:
                cosine_sim = cosine_similarity([target_vector], [vector])
                similarities[word] = cosine_sim[0][0]

        # Sort the words by their cosine similarity scores in descending order
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Select the top-k words as the k-nearest words
        nearest_words = [(word, round(e,4)) for word, e in sorted_similarities[:k]]

        # st.write the k-nearest words
        st.write(f"\nThe {k} nearest words to '{target_word}' are: ")
        for i in (nearest_words):
            st.write(i)
        return nearest_words

    
def get_k_nearest_using_embedding(k, embed_prediction, word_vectors, model, fw):   
    
    if model == 'fasttext':
        
        vectors = fw.wv.vectors
        words = fw.wv.index_to_key

        # Calculate cosine similarity between the target vector and all other vectors
        similarity_scores = [np.dot(embed_prediction, vectors[i]) / (np.linalg.norm(embed_prediction) * np.linalg.norm(vectors[i])) for i in range(len(vectors))]

        # Find the indices of the k most similar words
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]

        # Get the corresponding words and their similarity scores
        nearest_words = [(words[i], round(similarity_scores[i], 4)) for i in top_k_indices]

        return nearest_words

        
    else: 
        embedding_matrix = [word_vectors[i] for i in word_vectors]
        
        similarity_scores = cosine_similarity([embed_prediction], embedding_matrix)[0]

        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        nearest_words = [(word,similarity_scores[tokenizer.word_index[word]]) for word in tokenizer.word_index if tokenizer.word_index[word] in top_k_indices]

        return nearest_words




def get_embeddings(model_name):
    
    if model_name == 'fasttext':
        return get_fasttext_embeddings()
    
    if model_name == 'skipgram':
        return get_skipgram_embeddings()
    
    if model_name == 'cbow':
        return get_cbow_embeddings()
    
    if model_name == 'lsa':
        return get_lsa_embeddings()


def print_analogy(analogy, model_names):
    
    # Retrieve the words from the analogy we need to compute
    word_a, word_b, word_c, word_true = analogy    
        
    # Formulate the analogy task
    analogy_task = f"{word_a} is to {word_b} as {word_c} is to ?"

    st.write(f"Analogy Task: {analogy_task}")
    st.write("---------------------------------------------------")
    
    g_word = glove.most_similar(positive=[word_a, word_c], negative=[ word_b], topn=1)
    st.write(f"Glove prediction for Analogy is : {g_word}\n\n")
    st.write("---------------------------------------------------\n")

    
    
    if word_true not in vocab or word_a not in vocab or word_b not in vocab or word_c not in vocab:
        model_names.remove('lsa')
        print("Some input word or words not in vocab of LSA\n\n")
    
    if word_true not in tokenizer.word_index or word_a not in tokenizer.word_index or word_b not in tokenizer.word_index or word_c not in tokenizer.word_index:
        model_names.remove('skipgram')
        model_names.remove('cbow')
        print("Some input word or words not in vocab of skipgram and cbow\n\n")

    for model in model_names:
        embeddings = get_embeddings(model)
        embed_a, embed_b, embed_c, embed_true = embeddings[word_a],embeddings[word_b],embeddings[word_c],embeddings[word_true]
        embed_prediction = embed_b - embed_a + embed_c
        sim1 = round(cosine(embed_true, embed_prediction), 4)
        nearest_words = get_k_nearest_using_embedding(10, embed_prediction, embeddings, model, fasttext)  
        sorted_similarities = sorted(nearest_words, key=lambda x: x[1], reverse=True)
        
        word_prediction, sim2 = sorted_similarities[0]

    
        # st.write whether or not the true word was in the top nr 
        partially_correct = word_true in [word[0] for word in nearest_words]
        
        st.write(f"Embedding: {model}")
        # st.write all top nr words with their distance
        for word in nearest_words:
            st.write(f"{word[0]} => {round(word[1], 4)}")
        st.write(f"Predicted: {word_prediction} ({round(sim2, 4)}) - True: {word_true} ({sim1})")
        st.write(f"Correct? {word_prediction == word_true} - In the top {10}? {partially_correct}")
        st.write("---------------------------------------------------\n")




# analogies = [('he', 'is', 'we', 'are'), ('love', 'hate', 'little', 'large'), ('small', 'smaller', 'large', 'larger'), ('man', 'woman', 'king', 'queen'), ('mouse', 'mice', 'cat', 'cats')]
# for analogy in analogies:
#     st.write_analogy(analogy, ['skipgram', 'cbow', 'fasttext', 'lsa'])

st.title("NLP Project: Comparison of Different Word Embedding Techniques")
st.text("Group: Chetan, Harshvivek, Udhay")


st.text("Analogy Test: word1 is to word2 as word3 is to word4")

input1 = st.text_input("Enter word1: ")
input2 = st.text_input("Enter word2: ")
input3 = st.text_input("Enter word3: ")
input4 = st.text_input("Enter word4: ")



if st.button("Process"):
    print_analogy([input1,input2,input3,input4], ['skipgram', 'cbow', 'fasttext', 'lsa'])





