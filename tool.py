import re
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def filter_non_abstract(data):
    data = list(map(dotdict, data))
    abstract_len = {}
    no_abstract = []
    for i, paper in enumerate(data):
        if paper.abstract:
            abstract_len[i] = len(paper.abstract)
        else:
            no_abstract.append(i)

    print("len of no_abstract:",len(no_abstract))

    abstract_len = sorted(abstract_len.items(), key=lambda x:x[1])

    defect_abstract_idx = [ i for i,n in abstract_len if n < 100 ] + no_abstract
    print("len of defect_abstract_idx:",len(defect_abstract_idx))

    for i in sorted(defect_abstract_idx, reverse=True):
        del data[i]
    return data

def remove_url(text):
    return re.sub(r'https?://\S+', '', text)

def remove_otr(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)