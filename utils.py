import pandas as pd
import nltk
# Run this command if you are getting module errors from NLTK:
# python -m nltk.downloader stopwords
from nltk.corpus import stopwords

def main():
    """ Utils """
    testFileName = './covid_test_public.tsv'
    trainingFileName = './covid_training.tsv'

    getData(trainingFileName)
    stopWords = stopwords.words('english')


def getData(fileName):
    dataset = pd.read_csv(fileName, sep='\t')
    print(dataset)
    return dataset



if __name__ == "__main__":
    main()