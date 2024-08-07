from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_vectorizer(standard_titles):
    vectorizer = TfidfVectorizer()
    standard_vectors = vectorizer.fit_transform(standard_titles)
    return vectorizer, standard_vectors


def get_semantic_match(raw_title, vectorizer, standard_vectors, standard_titles):
    raw_vector = vectorizer.transform([raw_title])
    similarities = cosine_similarity(raw_vector, standard_vectors)
    best_match_idx = similarities.argmax()
    return standard_titles[best_match_idx], similarities.max()
