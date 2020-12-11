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
    # OV = addLabels(trainingFileName, generateOV(trainingFileName))
    OV = generateOV(trainingFileName, 'ovOutput.csv')
    FV = generateFV(trainingFileName, 'fvOutput.csv')

    # FV = addLabels(scaledDown, generateFV(scaledDown))

    generatePredictionData(trainingFileName)

def generateOV(fileName, outputFileName):
    V = generateCountVector(fileName)
    V.to_csv(outputFileName, sep='\t')
    return V

def generateFV(fileName, outputFileName):
    V = generateCountVector(fileName)
    cols = [col for col in V.columns]
    for col in cols:
        wordFrequency = V[col].sum()
        if col != 'q1_label':
            if int(wordFrequency) < 2:
                V.drop([col], axis=1, inplace=True)
    V.to_csv(outputFileName, sep='\t')
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
    
    # Uncomment it if you want to see file output
    return V

def generateCountVector(fileName):
    data = getData(fileName, True)
  
    # data['text'] = cleanText(data['text'])
    V = pd.DataFrame()
    # data['text'] = data['text'].str.lower()
    for i in range(len(data['text'])):
        print(i)
        wordsArr = data['text'][i]
        wordsArr = wordsArr.lower()
        words = wordsArr.split(' ')
        # words = [''.join(c for c in s if c not in string.punctuation) for s in words]
        # words = [s for s in words if s]

        d = {}

        for w in words:
            d[w] = d.get(w, 0) + 1
            # print(d)
        d['q1_label'] = data['q1_label'][i]


        V = V.append(d, ignore_index=True)
        # V = pd.DataFrame.from_dict(d, d.keys())

    V = V.fillna(0)  
    V = V.reindex(sorted(V.columns), axis=1)   
    # V = V.apply(pd.to_numeric, errors='ignore')
    V = V[ [ col for col in V.columns if col != 'q1_label' ] + ['q1_label'] ]
    print(V) 
    return V

def getData(fileName, hasHeaders):
    dataset = None
    if hasHeaders:
        dataset = pd.read_csv(fileName, sep='\t')
    else:
        dataset = pd.read_csv(fileName, sep='\t', header=None)
    return dataset

def generatePredictionData(fileName):
    predictionData = []
    data = getData(fileName, False)
    tweetIdArray = [row[0] for index,row in data.iterrows()]
    for i in range(len(tweetIdArray)):
        for wordsArr in data[1]:
            wordsArr = wordsArr.lower()
            words = wordsArr.split(' ')
            # words = [''.join(c for c in s if c not in string.punctuation) for s in words]
            # words = [s for s in words if s]
        predictionData.append((tweetIdArray[i], words))
        
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
        print(true_positive)
        print(true_negative)
        print(false_negative)
        print(false_positive)
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