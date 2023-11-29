from sklearn.feature_extraction.text import TfidfVectorizer

class MyTfidfVectorizer(TfidfVectorizer):
    def get_important_words(self, X, k=20, threshold=0.15):
        feature_names = self.get_feature_names_out()
        result = []
        for x in X:
            word_score = [(feature_names[idx], x[0, idx]) for idx in x.nonzero()[1]] 
            word_score.sort(key=lambda j: j[1], reverse=True)
            word_score = [(word, score) for word, score in word_score if score > threshold]
            result.append(dict(word_score[:k]))
        return result

if __name__ == "__main__":
    from tool import *
    from preprocess import  preprocess

    data = get_data("./ACL_PAPERS.json")
    data = preprocess(data[:100])
    abstracts = [paper.abstract for paper in data]

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    # X[293]
    word_cluster = merge_dict_by_sum(vectorizer.get_important_words(X, k=20, threshold=0.15))
    sorted(word_cluster.items(),key=lambda j: j[1], reverse=True)[:10]
    
