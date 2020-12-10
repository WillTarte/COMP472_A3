import pandas as pd
import nltk
# Run this command if you are getting module errors from NLTK:
# python -m nltk.downloader stopwords
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string

def main():
    """ Utils """
    testFileName = './covid_test_public.tsv'
    trainingFileName = './covid_training.tsv'
    scaledDown = './scaled_down.tsv'

    outputFileNameOV = 'NB-BOW-OV.csv'
    outputFileNameFV = 'NB-BOW-FV.csv'

    # print(generateOV(scaledDown, outputFileNameOV))
    # print(generateFV(scaledDown, outputFileNameFV))

    #Training
    OV = addLabels(trainingFileName, generateOV(trainingFileName))
    FV = addLabels(scaledDown, generateFV(scaledDown))

    generatePredictionData(trainingFileName)

def generateOV(fileName):
    V = generateCountVector(fileName)
    return V

def generateFV(fileName):
    V = generateCountVector(fileName)
    cols = [col for col in V.columns]
    for col in cols:
        wordFrequency = V[col].sum()
        if wordFrequency < 2:
            V.drop([col], axis=1, inplace=True)
    return V

def addLabels(fileName, V):
    columnsArray = ['q1_label']
    data = getData(fileName, True)
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
    data = getData(fileName, True)
  
    data['text'] = cleanText(data['text'])
    V = pd.DataFrame()
    data['text'] = data['text'].str.lower()
    for i in data['text']:
        words = i;
        word = words.split(' ')
        for s in word:
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = cleanText(data['text'])
        # print(word), 
        word = list(filter(None, word))
        word = list(filter(bool, word))
        word = list(filter(len, word))
        word = list(filter(lambda item: item, word))

        d = {}
        for w in word:
            d[w] = d.get(w, 0) + 1

        # # print(d)
        # rows = []
        # V = pd.DataFrame()
        V = V.append(d, ignore_index=True)
        # V = pd.DataFrame.from_dict(d, d.keys())

    V = V.fillna(0)        

    return V

def getData(fileName, hasHeaders):
    dataset = None
    if hasHeaders:
        dataset = pd.read_csv(fileName, sep='\t')
    else:
        dataset = pd.read_csv(fileName, sep='\t', header=None)
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

def generatePredictionData(fileName):
    predictionData = []
    data = getData(fileName, False)
    tweetIdArray = [row[0] for index,row in data.iterrows()]
    data[1] = data[1].str.lower()
    for i in range(len(tweetIdArray)):
        for j in data[1]:
            words = j;
            word = words.split(' ')
            for s in word:
                s = s.translate(str.maketrans('', '', string.punctuation))
                s = cleanText(data['text'])
            # # print(word), 
            word = list(filter(None, word))
            word = list(filter(bool, word))
            word = list(filter(len, word))
            analyze = word
        predictionData.append((tweetIdArray[i], analyze))
        
    return predictionData

def generateOutputFiles(name: str, predictions: pd.DataFrame, testData: pd.DataFrame):

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    with open("trace_" + name + ".txt", 'w') as trace_file:
        for index, row in predictions.iterrows():
            trace_file.write(str(row["tweet_id"]) + "  ")
            trace_file.write(row["class"] + "  ")
            trace_file.write(str(row["score"]) + "  ")
            true_label = testData.loc[testData["tweet_id"] == row["tweet_id"]]["q1_label"]
            true_label = true_label.iloc[0]
            trace_file.write(true_label + " ")

            if row["class"] == true_label and row["class"] == "yes":
                true_positive += 1
            elif row["class"] == true_label and row["class"] == "no":
                true_negative += 1
            elif row["class"] != true_label and row["class"] == "yes":
                false_positive += 1
            else:
                false_negative += 1

            trace_file.write("correct" if row["class"] == true_label else "wrong")
            trace_file.write("\r")
        
        print("Outputted " + "trace_" + name + ".txt")

    with open("eval_" + name + ".txt", 'w') as eval_file:

        # Accuracy
        eval_file.write(str((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)))
        eval_file.write("\r")
        print(true_negative)
        print(false_negative)
        # Precision
        eval_file.write(str(true_positive / (true_positive + false_positive)))
        eval_file.write("  ")
        eval_file.write(str(true_negative / (true_negative + false_negative)))
        eval_file.write("\r")

        # Recall
        eval_file.write(str(true_positive / (true_positive + false_negative)))
        eval_file.write("  ")
        eval_file.write(str(true_negative / (true_negative + false_positive)))
        eval_file.write("\r")

        # F1 measure
        eval_file.write(str(true_positive / (true_positive + (1/2)*(false_positive + false_negative))))
        eval_file.write("  ")
        eval_file.write(str(true_negative / (true_negative + (1/2)*(false_negative + false_positive))))
        eval_file.write("\r")

        print("Outputted " + "eval_" + name + ".txt")


if __name__ == "__main__":
    main()