import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

class MyTfidfVectorizer(TfidfVectorizer):
    def get_important_words(self, X, k=-1, threshold=0):
        feature_names = self.get_feature_names_out()
        result = []
        for x in X:
            word_score = [(feature_names[idx], x[0, idx]) for idx in x.nonzero()[1]] 
            word_score.sort(key=lambda j: j[1], reverse=True)
            word_score = [(word, score) for word, score in word_score if score > threshold]
            result.append(dict(word_score[:k]))
        return result
    
    def embedding(self, X, word2vec):
        feature_names = self.get_feature_names_out()
        result = []
        for x in tqdm(X, total=X.shape[0]):
            word_score_sum = 0
            paper_vec = np.zeros(word2vec.vector_size)
            for idx in x.nonzero()[1]:
                word_score = x[0, idx]
                word_score_sum += word_score
                if feature_names[idx] in word2vec:
                    paper_vec += word2vec[feature_names[idx]] * word_score
                else:
                    print(f"paper no vocab {feature_names[idx]}")
            paper_vec /= word_score_sum
            result.append(paper_vec)
        return result

if __name__ == "__main__":
    from tool import *
    from preprocess import  preprocess

    data = get_data("./ACL_PAPERS.json")
    data = preprocess(data[:300])
    abstracts = [paper.abstract for paper in data]

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    important_words = vectorizer.get_important_words(X, k=20, threshold=0.17)
    print("important words of paper:", data[293].title)
    print(*important_words[293].items(), sep='\n')

    word_cluster = merge_dict_by_sum(important_words)
    print("\ntop sum of td-idf word by in data")
    print(*sorted(word_cluster.items(),key=lambda j: j[1], reverse=True)[:10], sep='\n')
    
