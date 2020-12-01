from typing import Dict, List

class NaiveBayesBOWClassifier:

    def __init__(self, smoothing=0.01):
        self.features = None
        self.labels = None
        self.smoothing = smoothing

    def train(self, features: List[Dict[str, int]], labels: List[int]):
        self.features = features
        self.labels = labels

        # TODO training
        # For each class (yes/no) calculate the prior probabilities P(H_i)
        # For each word in the vocabulary (using the frequencies) calculate the conditional probabilities
        self.yes_count = 0
        self.no_count = 0
        for label in labels:
            if label == 0:
                self.no_count += 1
            elif label == 1:
                self.yes_count += 1
            else:
                raise Exception
        self.yes_prior = self.yes_count / float(len(labels))
        self.no_prior = self.no_count/ float(len(labels))


    #def predict(self, )

    