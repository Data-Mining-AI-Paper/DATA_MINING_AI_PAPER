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

    # print(stopwords.words('english'))

    return texts

if __name__ == "__main__":
    with open("./ACL_PAPERS.json",'r') as f:
        data = json.load(f)

    data = filter_non_abstract(data)
    abstracts = [paper.abstract for paper in data]

    abstracts = preprocess(abstracts[:300])
    print(abstracts[293])

