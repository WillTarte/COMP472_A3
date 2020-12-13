from typing import Dict, List, Tuple
from pandas import DataFrame, Series
from math import log10
import utils

class NaiveBayesCovidTweetClassifier:

    def __init__(self, vocab_length: int, smoothing=0.01):
        """
        vocab_length: number of words in the vocabulary\n
        smoothing: smoothing values when computing word probabilities
        """
        self.smoothing: float = smoothing
        self.vocab_length: int = vocab_length
        self.yes_word_probs: Dict[str, float] = dict()
        self.no_word_probs: Dict[str, float] = dict()
        self.trained: bool = False

    def train(self, data: DataFrame):
        """
        Trains the Naive Bayes Covid Tweet classifier.\n
        Expects the data to be have columns [[...words in vocab], "q1_label"]
        """
        assert len(data.columns) == self.vocab_length + 1, "The vocab length and the number of words in the training data don't match!"
        self.trained = True

        labels: Series = data['q1_label']
        yes_tweets = data[data['q1_label'] == "yes"].drop("q1_label", axis=1)
        no_tweets = data[data['q1_label'] == "no"].drop("q1_label", axis=1)

        # for all classes c_i
        #   computer P(c_i) = count(documents in c_i) / count(all documents)
        self.yes_count: int = (labels[labels == "yes"]).count() 
        self.no_count: int = (labels[labels == "no"]).count() 
        self.yes_prior: float = float(self.yes_count) / labels.size
        self.no_prior: float = float(self.no_count) / labels.size

        # for all classes c_i
        #   for all words w_j in the vocab
        #       compute P(w_j | c_i) = count(w_j, c_i) / Sum_j(count(w_j, c_i))
        for word, count in yes_tweets.items():
            word_prob = (count.sum() + self.smoothing) / (self.yes_count + (self.smoothing * self.vocab_length))
            self.yes_word_probs[word] = word_prob
        
        for word, count in no_tweets.items():
            word_prob = (count.sum() + self.smoothing) / (self.no_count + (self.smoothing * self.vocab_length))
            self.no_word_probs[word] = word_prob

    def predict(self, X: List[Tuple[str, List[str]]]) -> DataFrame:
        """
        Given a set of tweets, predicts if each tweet contains or not a verifiable factual claim.\n
        Expects the data to be in the format of a list of Tweets, where each tweet has the twee_id, and a list of words in the tweet.\n
        Outputs a DataFrame where each row represents a prediction on an input tweet. Each row contains the tweet_id, predicted class and score.
        """

        output: DataFrame = DataFrame(columns=["tweet_id", "class", "score"])

        # for all classes c_i
        #   score(c_i) = P(c_i)
        #   for all words w_j in the tweet
        #       score(c_i) = score(c_i) + log10(P(w_j | c_i))
        for tweet in X:
            score_yes = log10(self.yes_prior) + sum([log10(self.yes_word_probs.get(word, 1)) for word in tweet[1]])
            score_no = log10(self.no_prior) + sum([log10(self.no_word_probs.get(word, 1)) for word in tweet[1]])

            if score_yes >= score_no:
                output = output.append({"tweet_id": tweet[0], "class": "yes", "score": score_yes}, ignore_index=True)
            else:
                output = output.append({"tweet_id": tweet[0], "class": "no", "score": score_no}, ignore_index=True)
        
        return output
    
    def __repr__(self):
        if not self.trained:
            return "Model not trained yet. Vocab length: " + str(self.vocab_length) + ". Smoothing: " + str(self.smoothing) + "."
        else:
            return "Model has been trained. Yes prior: " + str(self.yes_prior) + ". No prior: " + str(self.no_prior) + "."


if __name__ == "__main__":

    OV, FV = utils.generateTrainingData("covid_training.tsv")

    print("Generated training data")

    testData = utils.generatePredictionData("covid_test_public.tsv")
    testDataFrame = utils.getData("covid_test_public.tsv", False)
    
    print("Generated necessary testing data")

    nbClassifierOV = NaiveBayesCovidTweetClassifier(len(OV.columns) - 1)
    nbClassifierOV.train(OV)

    print("Trained on OV")

    nbClassifierFV = NaiveBayesCovidTweetClassifier(len(FV.columns) - 1)
    nbClassifierFV.train(FV)

    print("Trained on FV")
    
    predictionsOV = nbClassifierOV.predict(testData)

    print("Generated predictions for OV")

    predictionsFV = nbClassifierFV.predict(testData)

    print("Generated predictions for FV")

    utils.generateOutputFiles("NB-BOW-OV", predictionsOV, testDataFrame)
    utils.generateOutputFiles("NB-BOW-FV", predictionsFV, testDataFrame)



    
    








    