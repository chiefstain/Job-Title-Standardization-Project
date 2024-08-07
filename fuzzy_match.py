from rapidfuzz import process, fuzz

def match_job_title(raw_title, standard_titles):
    match = process.extractOne(raw_title, standard_titles, scorer=fuzz.token_sort_ratio)
    return match
