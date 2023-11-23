import json
from tool import dotdict
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

data = list(map(dotdict, data))
abstract_len = {}
no_abstract = []
for i, paper in enumerate(data):
    if paper.abstract:
        abstract_len[i] = len(paper.abstract)
    else:
        no_abstract.append(i)

abstract_len = sorted(abstract_len.items(), key=lambda x:x[1])

defect_abstract_idx = [ i for i,n in abstract_len if n < 100 ] + no_abstract
for i in sorted(defect_abstract_idx, reverse=True):
    del data[i]

abstract = [paper.abstract for paper in data]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(abstract)
# vectorizer.get_feature_names_out()
print(X.shape)
print(vectorizer.vocabulary_) # bog of word
print(X)
print(len(vectorizer.get_feature_names()))
print(vectorizer.idf_)

for i, paper in enumerate(data):
    if "Knowledge Graphs" in paper.title:
        print(i, paper.title)

p = abstract[293]
response = vectorizer.transform([p])
p_tdfidf = [(col, vectorizer.get_feature_names()[col], response[0, col]) for col in response.nonzero()[1]] 
p_tdfidf.sort(key=lambda x: x[2], reverse=True)
print(*p_tdfidf[:20], sep='\n')