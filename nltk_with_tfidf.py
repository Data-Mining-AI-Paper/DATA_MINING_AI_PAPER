from sys import stdin
from collections import defaultdict
import json
import re
from tokenizers import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')



def replace_non_alphabetic(sentence):
    # 정규 표현식을 사용하여 알파벳을 제외한 모든 문자를 공백으로 대체
    result = re.sub(r'[^a-zA-Z]', ' ', sentence)
    return result



a_json = open('./ACL_PAPERS.json', encoding = 'utf-8')
a_dict = json.load(a_json)	#=> 파이썬 자료형(딕셔너리나 리스트)으로 반환
# test_json = json.dumps(a_dict, ensure_ascii=False, indent=2)
print(a_dict[0]['abstract'])
print('-' * 20)

# input comes from STDIN (standard input)
# for line in a_dict[0]['abstract']:
    # remove leading and trailing whitespace
    # line = line.strip()
    # split the line into words

abstracts = [a_dict[i]['abstract'] for i in range(10)]
abstracts.append(a_dict[293]['abstract'])

# 중요한 단어들을 담을 리스트
important_words_list = []

for abstract in abstracts:
    # 텍스트 전처리: 소문자로 변환하고 문장부호 및 불용어 제거
    abstract = abstract.lower()
    for punctuation in string.punctuation:
        abstract = abstract.replace(punctuation, "")
    tokens = word_tokenize(abstract)
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # TF-IDF 계산
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([' '.join(tokens)])

    # 단어별 중요도 출력
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    word_scores = list(zip(feature_names, scores))

    # 중요도 순으로 정렬하여 상위 단어 저장
    word_scores.sort(key=lambda x: x[1], reverse=True)
    threshold = 0.2  # 임계치 설정
    important_words = [(word, score) for word, score in word_scores if score > threshold]
    
    # 중요한 단어들을 리스트에 추가
    important_words_list.append(important_words)

# 결과 출력
for i, words in enumerate(important_words_list, 1):
    print(f"Abstract {i}의 중요한 단어들 및 중요도 수치:")
    for word, score in words:
        print(f"{word}: {score}")
    print()
print('-' * 20)

words = a_dict[0]['abstract'].split()
word_dict = defaultdict(int)
    # increase counters
for word in words:
    word = replace_non_alphabetic(word)
    word_dict[word.lower()] += 1

word_list = sorted(word_dict.items(), key=lambda x: -x[1])

# for key, val in word_list:
#     print(key, val)