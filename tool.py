import json
from collections import Counter
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_data(fname: str):
    with open(fname,'r') as f:
        data = json.load(f)
        return list(map(dotdict, data))

def merge_dict_by_sum(dicts):
    counter = Counter()
    for d in dicts:
        counter += Counter(d)
    return dict(counter)