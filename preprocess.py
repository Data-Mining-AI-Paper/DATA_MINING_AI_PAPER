
import re
import nltk
from tqdm import tqdm
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from tool import *

def remove_poor_abstracts(data):
    abstract_len = {}
    no_abstract = []
    for i, paper in enumerate(data):
        if paper.abstract:
            abstract_len[i] = len(paper.abstract)
        else:
            no_abstract.append(i)
    print("no abstracts:",len(no_abstract))

    abstract_len = sorted(abstract_len.items(), key=lambda x:x[1])
    poor_abstracts = [ i for i,n in abstract_len if n < 100 ] + no_abstract
    print("poor abstracts:",len(poor_abstracts))

    for i in sorted(poor_abstracts, reverse=True):
        del data[i]
    
    return data

def select_field(data):
    fields = ['title','abstract','year']
    for i in range(len(data)):
        data[i] = dotdict({key: data[i][key] for key in fields})
    return data

def remove_url(text):
    return re.sub(r'https?://\S+', '', text)

def remove_otr(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)

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

def preprocess(data):
    data = remove_poor_abstracts(data)
    
    data = select_field(data)

    for i, paper in enumerate(data):
        data[i].abstract = remove_otr(remove_url(paper.abstract))
    
    print("Process in tokenizing and finding part of speech")
    texts_pos = []
    for paper in tqdm(data):
        texts_pos.append(nltk.pos_tag(word_tokenize(paper.abstract)))

    print("Process in lemmatizing and removing stopword")
    lemmatizer = WordNetLemmatizer()
    for i, text_pos in enumerate(tqdm(texts_pos)):
        words = []
        for token, pos in text_pos:
            word = lemmatizer.lemmatize(token, get_wordnet_pos(pos))
            if word not in stopwords.words('english'):
                words.append(word)
        data[i].abstract = ' '.join(words)

    return data

if __name__ == "__main__":
    data = get_data("./ACL_PAPERS.json")
    data = preprocess(data[:100])
    print(data[0])

    import pickle
    data = get_data("./ACL_PAPERS.json")
    data = preprocess(data)
    with open("preprocessed_ACL_PAPERS.pickle","wb") as f:
        pickle.dump(data, f)