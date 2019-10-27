from google.cloud import vision
import io


class TextDetector:
    def __init__(self):
        self.text_list = []

    def detect_text(self, path):
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        for text in texts[1:len(texts)]:
            self.text_list.append(text.description)
