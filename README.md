# ![title_2](https://github.com/Data-Mining-AI-Paper/DATA_MINING_AI_PAPER/assets/78012131/10d9f387-3088-4a6d-99a6-3daa8435c0cb)

> 2023 FALL Data Mining(SCE3313, F074) Project

## 🚩 Table of Contents

- [Project summary](#-project-summary)
- [Project structure](#-project-structure)
- [Requirements](#-requirements)
- [Methods](#-methods)
- [Results](#-results)
- [License](#-license)

## 📃 Project summary

### Analysis Challenges in NLP Papers: BOLT - Beyond Obstacles, Leap Together

- We want to solve the problem of accessing ACL papers in natural language processing research and proposes solutions.
- The methods that we tried include TF-IDF, SVD, and K-means clustering to derive insights from a dataset of 12,745 papers.
- Using the data that crawled the acl paper, we made below three outputs:
  1. Keyword trend analysis with graphs
  2. Word cloud by year
  3. Research topic trend with clusters by year

### Team member

| Dept     | Name          | Github|
| -------- | ------------- |-------|
| software | Kyunghyun Min <img src="https://avatars.githubusercontent.com/u/16879600?v=4" width="50">|[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>](https://github.com/enjoeyland)|
| software | Jongho    Baik <img src="https://github.com/Data-Mining-AI-Paper/DATA_MINING_AI_PAPER/assets/78012131/e9bf5d98-277a-492f-a6f5-924f41c8ce67" width="50">|[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>](https://github.com/JongHoB)|
| software | Junseo    Lee <img src="https://avatars.githubusercontent.com/u/78635277?v=4" width="50">   |[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>](https://github.com/LucPle)|

## 📈 Project structure

### Directory

```bash
/DATA_MINING_AI_PAPER
├── output
│   ├── k-means
│   └── wordcloud
├── tempfiles
├── 1. Crawling ACL.ipynb
├── 2. preprocess.py
├── 3. k-mean_clustering_word2vect.py
├── 4. keyword_trend.py
├── 5. wordcloud_by_year.py
├── 6. topic_trend.py
├── tf_idf.py
├── tool.py
├── ACL_PAPERS.json
├── LICENSE
├── preprocessed_ACL_PAPERS.pickle
└── README.md
```

### Details

- `output/k-means`: Contain the results of k-means++ clustering, labels of clusters, info about instance of k.
- `output/wordcloud`: Contain wordclouds by year, from 1979 to 2023.
- `1. Crawling ACL.ipynb` and `2. preprocess.py`: Crawl papers and preprocessing data.
- `3. k-mean_clustering_word2vect.py`: Make clusters by k-means++.
- `4. keyword_trend.py`: Provide graphs about changes in the importance of keywords by year.
- `5. wordcloud_by_year.py`: Provide important keywords as wordclouds by year.
- `6. topic_trend.py`: Labeling the clusters made in `3. k-mean_clustering_word2vect.py`.

## 🔑 Requirements

### Hardware Configuration

- ![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white) Compute Engine VM N1 instance
  - Custom configuration:
    - ![Intel® Xeon® E5-2696V3](https://img.shields.io/badge/Intel®-Xeon®_E5_2696V3-0071C5?style=for-the-badge&logo=intel&logoColor=white) 10vCPU
    - ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) 23.10
    - 65GB RAM
    - 200GB storage

### Software Configuration

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 3.11.5 
- ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
  - IPython: 8.15.0
  - ipykernel: 6.25.0
  - ipywidgets: 8.0.4
  - jupyter_client: 7.4.9
  - jupyter_core: 5.3.0
  - jupyter_server: 1.23.4
  - jupyterlab: 3.6.3
  - nbclient: 0.5.13
  - nbconvert: 6.5.4
  - nbformat: 5.9.2
  - notebook: 6.5.4
  - qtconsole: 5.4.2
  - traitlets: 5.7.1

### Additional Libraries

To ensure consistency in package versions, the following additional libraries are used:

- ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black): 3.5.2
- ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white): 1.23.1
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white): 1.3.0
- ![nltk](https://img.shields.io/badge/nltk-blue): 3.8.1
- ![pyclustering](https://img.shields.io/badge/pyclustering-purple): 0.10.1.2
- ![wordcloud](https://img.shields.io/badge/wordcloud-black): 1.9.2

## 🔨 Method

## 🎫 Result

## 📜 License

This software is licensed under the [MIT](https://github.com/nhn/tui.editor/blob/master/LICENSE) © 2023 Data-Mining-AI-Paper
