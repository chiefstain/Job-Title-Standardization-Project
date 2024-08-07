import sys
import os
import pandas as pd
import pytest

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_data, load_data
from fuzzy_match import match_job_title
from translate import translate_title
from semantic_match import create_vectorizer, get_semantic_match

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_preprocess_data():
    data = {'title': ['Data Scientist', 'Machine Learning Engineer', None]}
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df)
    assert processed_df.isna().sum().sum() == 0
    assert (processed_df['title'] == ['data scientist', 'machine learning engineer']).all()

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_load_data():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'job_title_des.csv'))
    df = load_data(file_path)
    assert not df.empty

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_match_job_title():
    standard_titles = ["data scientist", "machine learning engineer"]
    match = match_job_title("data scientist", standard_titles)
    assert match[0] == "data scientist"

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_translate_title():
    translated = translate_title("Ingenieur en apprentissage automatique", src_lang='fr', dest_lang='en')
    valid_translations = ["machine learning engineer", "automatic learning engineer"]
    assert translated.lower() in valid_translations

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_semantic_match():
    standard_titles = ["data scientist", "machine learning engineer"]
    vectorizer, standard_vectors = create_vectorizer(standard_titles)
    match, score = get_semantic_match("data scientist", vectorizer, standard_vectors, standard_titles)
    assert match == "data scientist"
    assert score > 0.8
