from typing import Dict, List
from pandas import DataFrame

class NaiveBayesBOWClassifier:

    def __init__(self, smoothing=0.01):
        self.features = None
        self.labels = None
        self.smoothing = smoothing

    def train(self, data: DataFrame):
        self.features = data[:, :-1]
        self.labels = data['q1_label']

       
        # Calculate priors (classes)
        self.yes_count = 0
        self.no_count = 0
        for label in self.labels.items():
            if label == "no":
                self.no_count += 1
            elif label == "yes":
                self.yes_count += 1
            else:
                raise Exception
        self.yes_prior = self.yes_count / float(len(self.labels))
        self.no_prior = self.no_count / float(len(self.labels))

        # for all classes c_i
        #   for all words w_j in the vocab
        #       compute P(w_j | c_i) = count(w_j, c_i) / Sum_j(count(w_j, c_i))


    #def predict(self, )

    