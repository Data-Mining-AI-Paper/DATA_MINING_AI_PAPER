import json
from tool import *
import numpy as np

SEARCH_FIELDS = [
    'abstract',
    'authors',
    'citationCount',
    'corpusId',
    'externalIds',
    'fieldsOfStudy',
    'influentialCitationCount',
    'isOpenAccess',
    'journal',
    'openAccessPdf',
    'paperId',
    'publicationDate',
    'publicationTypes',
    'publicationVenue',
    'referenceCount',
    's2FieldsOfStudy',
    'title',
    'url',
    'venue',
    'year'
]

# if __name__ == "__main__":
with open("./ACL_PAPERS.json",'r') as f:
    data = json.load(f)

data = filter_non_abstract(data)
abstract = [paper.abstract for paper in data]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(abstract)

# vectorizer.get_feature_names_out()
print(X.shape)
print(vectorizer.vocabulary_) # bog of word
print(X)
print(len(vectorizer.get_feature_names_out()))

for i, paper in enumerate(data):
    if "Knowledge Graphs" in paper.title:
        print(i, paper.title)

p = abstract[293]
print(p)
response = vectorizer.transform([p])
p_tdfidf = [(col, vectorizer.get_feature_names_out()[col], response[0, col]) for col in response.nonzero()[1]] 
p_tdfidf.sort(key=lambda x: x[2], reverse=True)
print(*p_tdfidf[:20], sep='\n')
