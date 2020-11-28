import pandas as pd
import nltk
# Run this command if you are getting module errors from NLTK:
# python -m nltk.downloader stopwords
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def main():
    """ Utils """
    testFileName = './covid_test_public.tsv'
    trainingFileName = './covid_training.tsv'
    scaledDown = './scaled_down.tsv'

    outputFileNameOV = 'NB-BOW-OV.csv'
    outputFileNameFV = 'NB-BOW-FV.csv'

    # print(generateOV(scaledDown, outputFileNameOV))
    # print(generateFV(scaledDown, outputFileNameFV))

    OV = addLabels(scaledDown, generateOV(scaledDown))
    FV = addLabels(scaledDown, generateFV(scaledDown))   

    print(OV)
    print(FV)

def generateOV(fileName):
    V = addSmoothing(generateCountVector(fileName))
    return V

def generateFV(fileName):
    """ If we need to smooth values that are below 0 remove addSmoothing function call below"""
    V = addSmoothing(generateCountVector(fileName))
    cols = [col for col in V.columns]

    for col in cols:
        V[col].values[V[col] < 2] = 0

    """ Do we still smooth for the values that are 0? """
    # V = addSmoothing(V)
    return V

def addLabels(fileName, V):
    columnsArray = ['q1_label']
    data = getData(fileName)
    for col in columnsArray:
        if (col == 'tweet_id'):
            V.insert(0, col, data[col])
        else:
            V.insert(len(V.columns), col, data[col])
    # Only used for demoing/testing code
    # V.to_csv('test.csv', sep='\t')
    # Uncomment it if you want to see file output
    return V

def generateCountVector(fileName):
    data = getData(fileName)
    textArray = [row['text'] for index,row in data.iterrows()]
    countVector = CountVectorizer()
    countVectorText = countVector.fit_transform(textArray)
    V = pd.DataFrame(countVectorText.toarray(), columns=countVector.get_feature_names())
    return V

def addSmoothing(V):
    V = V + 0.01
    return V

def getData(fileName):
    dataset = pd.read_csv(fileName, sep='\t')
    return dataset

def cleanText(text):
    unwanted_char = ('\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17'
    '\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90'
    '\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab'
    '\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6'
    '\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1'
    '\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff')

    text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])

    # Converting all to lower case
    text = text.lower()
    # Remove Punctuation?

    # If we choose to use countVectorizing or tf-idf to do our calculation, we should attempt to remove stop words
    # However it can lower our accuracy, we can test this later
    stopWords = stopwords.words('english')

    return text


if __name__ == "__main__":
    main()