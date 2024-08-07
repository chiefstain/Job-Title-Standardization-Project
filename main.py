import pandas as pd
from src.preprocess import load_data, preprocess_data
from src.fuzzy_match import match_job_title
from src.translate import translate_title
from src.semantic_match import create_vectorizer, get_semantic_match

# Load data
raw_data_path = 'data/job_title_des.csv'
raw_job_titles = load_data(raw_data_path)

# Preprocess data
raw_job_titles = preprocess_data(raw_job_titles)

# Fuzzy match
standard_job_titles = ["list", "of", "standard", "job", "titles"]
raw_job_titles['matched_title'] = raw_job_titles['title'].apply(lambda x: match_job_title(x, standard_job_titles))

# Translate titles
raw_job_titles['translated_title'] = raw_job_titles['title'].apply(lambda x: translate_title(x))

# Semantic match
vectorizer, standard_vectors = create_vectorizer(standard_job_titles)
raw_job_titles['semantic_match'], raw_job_titles['similarity_score'] = zip(*raw_job_titles['translated_title'].apply(lambda x: get_semantic_match(x, vectorizer, standard_vectors, standard_job_titles)))

# Save results
raw_job_titles.to_csv('data/matched_job_titles.csv', index=False)
