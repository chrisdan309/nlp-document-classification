from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import Counter

def vectorize_tfidf(train_docs, test_docs):
    # Definir el vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    # Ajustar y transformar los documentos de entrenamiento
    X_train_tfidf = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in train_docs])
    # Transformar los documentos de prueba
    X_test_tfidf = tfidf_vectorizer.transform([' '.join(doc) for doc in test_docs])
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_word2vec_models(all_sentences):
    # Modelo CBOW
    model_cbow = Word2Vec(all_sentences, vector_size=100, window=5, min_count=2, workers=4)
    # Modelo Skip-Gram
    model_sg = Word2Vec(all_sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    return model_cbow, model_sg

def average_word_vectors(sentences, model, vector_size):
    document_vectors = []
    for sentence in sentences:
        vector = np.zeros(vector_size)
        count = 0
        for word in sentence:
            if word in model.wv:
                vector += model.wv[word]
                count += 1
        if count > 0:
            vector /= count
        document_vectors.append(vector)
    return document_vectors

def load_glove_model(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
    model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model_glove

def average_word_vectors_glove(sentences, model, vector_size):
    document_vectors = []
    for sentence in sentences:
        vector = np.zeros(vector_size)
        count = 0
        for word in sentence:
            if word in model:
                vector += model[word]
                count += 1
        if count > 0:
            vector /= count
        document_vectors.append(vector)
    return document_vectors

def build_cooccurrence_matrix(sentences, vocab_size=5000, window_size=4):
    # Construir vocabulario basado en frecuencia
    word_freq = Counter([word for sentence in sentences for word in sentence])
    most_common_words = word_freq.most_common(vocab_size)
    vocab = [word for word, _ in most_common_words]
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))

    for sentence in sentences:
        sentence = [word for word in sentence if word in word_to_id]
        for i, target_word in enumerate(sentence):
            target_id = word_to_id[target_word]
            context_ids = [word_to_id[word] for word in sentence[max(i - window_size, 0): i] + sentence[i + 1: i + window_size + 1]]
            for context_id in context_ids:
                cooccurrence_matrix[target_id][context_id] += 1
    return cooccurrence_matrix, word_to_id

def calculate_ppmi(cooccurrence_matrix):
    total_sum = np.sum(cooccurrence_matrix)
    sum_over_rows = np.sum(cooccurrence_matrix, axis=1).reshape(-1, 1)
    sum_over_cols = np.sum(cooccurrence_matrix, axis=0).reshape(1, -1)

    expected = np.dot(sum_over_rows, sum_over_cols) / total_sum
    ppmi = np.maximum(np.log((cooccurrence_matrix * total_sum) / expected), 0)
    ppmi[np.isinf(ppmi)] = 0
    ppmi[np.isnan(ppmi)] = 0
    return ppmi

def get_document_vectors_ppmi(sentences, word_to_id, word_vectors):
    document_vectors = []
    for sentence in sentences:
        vector = np.zeros(word_vectors.shape[1])
        count = 0
        for word in sentence:
            if word in word_to_id:
                idx = word_to_id[word]
                vector += word_vectors[idx]
                count += 1
        if count > 0:
            vector /= count
        document_vectors.append(vector)
    return document_vectors

def train_classifiers(X_train, y_train, model_type='TF-IDF'):
    if model_type == 'TF-IDF':
        nb_classifier = MultinomialNB()
    else:
        nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    lr_classifier = LogisticRegression(max_iter=1000)
    lr_classifier.fit(X_train, y_train)

    return nb_classifier, lr_classifier
