from data_processing import load_data, preprocess_text, preprocess, filter_top_categories, encode_labels
from models import (vectorize_tfidf, train_word2vec_models, average_word_vectors,
                    load_glove_model, average_word_vectors_glove, build_cooccurrence_matrix,
                    calculate_ppmi, get_document_vectors_ppmi, train_classifiers)
from evaluation import evaluate_model, cross_validate_model, perform_statistical_test
from visualization import plot_f1_scores
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # Cargar y preprocesar datos
    train_docs, train_labels, test_docs, test_labels = load_data()
    stop_words, lemmatizer = preprocess_text()
    train_docs_filtered, train_labels_filtered, top_categories = filter_top_categories(train_docs, train_labels)
    test_docs_filtered, test_labels_filtered, _ = filter_top_categories(test_docs, test_labels)
    y_train, y_test, label_encoder = encode_labels(train_labels_filtered, test_labels_filtered)

    # Preprocesamiento de documentos
    train_sentences = preprocess(train_docs_filtered, stop_words, lemmatizer)
    test_sentences = preprocess(test_docs_filtered, stop_words, lemmatizer)
    all_sentences = train_sentences + test_sentences

    # Generación de representaciones vectoriales
    # TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_tfidf(train_sentences, test_sentences)

    # Word2Vec
    model_cbow, model_sg = train_word2vec_models(all_sentences)
    X_train_cbow = average_word_vectors(train_sentences, model_cbow, 100)
    X_test_cbow = average_word_vectors(test_sentences, model_cbow, 100)
    X_train_sg = average_word_vectors(train_sentences, model_sg, 100)
    X_test_sg = average_word_vectors(test_sentences, model_sg, 100)

    # GloVe
    glove_input_file = 'glove.6B.100d.txt'
    word2vec_output_file = 'glove.6B.100d.word2vec.txt'
    model_glove = load_glove_model(glove_input_file, word2vec_output_file)
    X_train_glove = average_word_vectors_glove(train_sentences, model_glove, 100)
    X_test_glove = average_word_vectors_glove(test_sentences, model_glove, 100)

    # PPMI
    cooc_matrix, word_to_id = build_cooccurrence_matrix(train_sentences)
    ppmi_matrix = calculate_ppmi(cooc_matrix)
    svd = TruncatedSVD(n_components=100)
    word_vectors_ppmi = svd.fit_transform(ppmi_matrix)
    X_train_ppmi = get_document_vectors_ppmi(train_sentences, word_to_id, word_vectors_ppmi)
    X_test_ppmi = get_document_vectors_ppmi(test_sentences, word_to_id, word_vectors_ppmi)

    # Entrenamiento y evaluación de modelos
    methods = ['TF-IDF', 'CBOW', 'Skip-Gram', 'GloVe', 'PPMI']
    embeddings = {
        'TF-IDF': (X_train_tfidf, X_test_tfidf),
        'CBOW': (X_train_cbow, X_test_cbow),
        'Skip-Gram': (X_train_sg, X_test_sg),
        'GloVe': (X_train_glove, X_test_glove),
        'PPMI': (X_train_ppmi, X_test_ppmi)
    }
    f1_scores_nb = []
    f1_scores_lr = []

    for method in methods:
        X_train, X_test = embeddings[method]
        nb_classifier, lr_classifier = train_classifiers(X_train, y_train, model_type=method)
        y_pred_nb = nb_classifier.predict(X_test)
        y_pred_lr = lr_classifier.predict(X_test)

        # Evaluación
        _, _, f1_nb = evaluate_model(y_test, y_pred_nb, "Naive Bayes", method)
        _, _, f1_lr = evaluate_model(y_test, y_pred_lr, "Regresión Logística", method)
        f1_scores_nb.append(f1_nb)
        f1_scores_lr.append(f1_lr)

    # Pruebas de significancia estadística
    X_tfidf_full = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in train_sentences + test_sentences])
    y_full = np.concatenate([y_train, y_test])
    lr_classifier = LogisticRegression(max_iter=1000)
    f1_scores_lr_tfidf = cross_validate_model(X_tfidf_full, y_full, lr_classifier)

    X_cbow_full = average_word_vectors(train_sentences + test_sentences, model_cbow, 100)
    f1_scores_lr_cbow = cross_validate_model(X_cbow_full, y_full, LogisticRegression(max_iter=1000))

    perform_statistical_test(f1_scores_lr_tfidf, f1_scores_lr_cbow, 'TF-IDF', 'CBOW')

    # Visualización de resultados
    plot_f1_scores(methods, f1_scores_nb, f1_scores_lr)

if __name__ == '__main__':
    main()
