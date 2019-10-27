from ProfaneWordsFilter.Prediction import predict
from ExplicitImageDetection.TextDetection import TextDetector
from ExplicitImageDetection.ExplicitContentDetection import ExplicitDetector


def detect_text_profanity(texts):
    if any(predict(texts)):
        return True
    else:
        return False


def detect_image_profanity(file):
    td = TextDetector()
    td.detect_text(file)
    p = False

    if td.text_list:
        p = detect_text_profanity(td.text_list)

    ed = ExplicitDetector()
    ed.detect_safe_search(file)
    e = ed.likelihoods

    if p:
        return True
    elif "POSSIBLE" in e or "LIKELY" in e or "VERY_LIKELY" in e:
        return True
    else:
        return False


def identify_profane_text(word_list):
    profane_text = []
    if detect_text_profanity(word_list):
        for word in word_list:
            if predict([word]):
                profane_text.append(word)
    return profane_text


def words_to_censor(text):
    words = text.split()
    return identify_profane_text(words)
