from google.cloud import vision
import io


class ExplicitDetector:
    def __init__(self):
        self.likelihoods = []

    def detect_safe_search(self, path):
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)
        response = client.safe_search_detection(image=image)
        safe = response.safe_search_annotation
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY',
                           'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
        content_types = [safe.adult, safe.medical, safe.violence]

        for content_type in content_types:
            self.likelihoods.append(likelihood_name[content_type])
