import json
import nltk
from tqdm import tqdm
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

from tool import *

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# def preprocess(texts: list[str]):
def preprocess(texts):
    texts = list(map(remove_otr, map(remove_url, texts)))
    
    texts_pos = []
    for text in tqdm(texts):
        texts_pos.append(nltk.pos_tag(word_tokenize(text)))

    lemmatizer = WordNetLemmatizer()
    texts = []
    for text_pos in tqdm(texts_pos):
       texts.append(' '.join([lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in text_pos])) 
    return texts

with open("./ACL_PAPERS.json",'r') as f:
    data = json.load(f)

data = filter_non_abstract(data)
abstracts = [paper.abstract for paper in data]

abstracts = preprocess(abstracts[:300])
print(abstracts[293])
# print(stopwords.words('english'))

