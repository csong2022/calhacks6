import numpy as np
import joblib
import pandas as pd

vectorizer = joblib.load("/Users/csong2022/calhacks6/ProfaneWordsFilter/resources/vectorizer.joblib")
model = joblib.load("/Users/csong2022/calhacks6/ProfaneWordsFilter/resources/model.joblib")


def _get_profane_prob(prob):
    return prob[1]


def predict(texts):
    return model.predict(vectorizer.transform(texts))


def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))


def accuracy(file, pfunc):
    data = pd.read_csv(file)
    texts = data['text'].astype(str).tolist()
    predictions = data['is_offensive'].tolist()
    results = pfunc(texts)
    matches = [i for i, j in zip(predictions, results) if i == j]
    print(len(matches)/len(predictions))
