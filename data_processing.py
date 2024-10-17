import nltk
from nltk.corpus import reuters, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def encode_labels(train_labels, test_labels):
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    return y_train, y_test, label_encoder

def load_data():
    nltk.download('reuters', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    # Obtener los IDs de los archivos
    file_ids = reuters.fileids()
    # Separar en entrenamiento y prueba según particiones predefinidas en NLTK
    train_ids = [doc_id for doc_id in file_ids if doc_id.startswith('training/')]
    test_ids = [doc_id for doc_id in file_ids if doc_id.startswith('test/')]
    # Obtener los documentos y sus categorías
    train_docs, train_labels = get_documents_and_labels(train_ids)
    test_docs, test_labels = get_documents_and_labels(test_ids)
    return train_docs, train_labels, test_docs, test_labels

def get_documents_and_labels(ids):
    documents = []
    labels = []
    for doc_id in ids:
        # Texto del documento
        text = reuters.raw(doc_id)
        documents.append(text)
        # Categorías (puede tener múltiples)
        categories = reuters.categories(doc_id)
        labels.append(categories)
    return documents, labels

def preprocess_text():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def preprocess(documents, stop_words, lemmatizer):
    preprocessed_docs = []
    for text in documents:
        # Convertir a minúsculas y tokenizar
        tokens = word_tokenize(text.lower())
        # Filtrar tokens alfabéticos y eliminar stop words
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        # Lematización
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        preprocessed_docs.append(tokens)
    return preprocessed_docs

def filter_top_categories(docs, labels, N=10):
    # Contar la frecuencia de cada categoría en el conjunto de entrenamiento
    category_counts = Counter([cat for sublist in labels for cat in sublist])
    # Seleccionar las N categorías más comunes
    top_categories = [cat for cat, _ in category_counts.most_common(N)]
    # Filtrar documentos que pertenecen a las categorías seleccionadas
    filtered_docs = []
    filtered_labels = []
    for doc, cats in zip(docs, labels):
        # Intersección de categorías del documento con las principales
        cats_in_top = list(set(cats).intersection(set(top_categories)))
        if cats_in_top:
            filtered_docs.append(doc)
            # Si hay múltiples categorías, seleccionamos una (por simplicidad)
            filtered_labels.append(cats_in_top[0])
    return filtered_docs, filtered_labels, top_categories
