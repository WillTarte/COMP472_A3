import pandas as pd
from typing import Dict, List, Tuple, Any

def generateTrainingData(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data = getData(filename, True)
    print("Got raw data")

    OV = generateCountDataFrame(data)
    print("Got OV bow data")

    FV = OV.copy()

    cols = [col for col in FV.columns]
    for col in cols:
        wordFrequency = FV[col].sum()
        if col != 'q1_label':
            if int(wordFrequency) < 2:
                FV.drop([col], axis=1, inplace=True)
    
    print("Got FV bow data")

    return (OV, FV)

def generateCountDataFrame(data: pd.DataFrame) -> pd.DataFrame:
  
    data_cleaned = pd.DataFrame()
    
    for _, row in data.iterrows():
        words = row["text"].lower().split(" ")
        tweet_word_frequencies: Dict[str, Any]= dict()
        tweet_word_frequencies["q1_label"] = row["q1_label"]
        
        for word in words:
            if word not in tweet_word_frequencies.keys():
                tweet_word_frequencies[word] = 1
            else:
                tweet_word_frequencies[word] += 1
                
        for word in tweet_word_frequencies.keys():
            if word not in data_cleaned.columns:
                data_cleaned.insert(0, word, 0)
                
        data_cleaned = data_cleaned.append(tweet_word_frequencies, ignore_index=True)

    return data_cleaned.fillna(0)

def getData(fileName: str, hasHeaders: bool) -> pd.DataFrame:
    dataset = None
    if hasHeaders:
        dataset = pd.read_csv(fileName, sep='\t')
    else:
        dataset = pd.read_csv(fileName, sep='\t', header=None)
        dataset.columns = ["tweet_id", "text", "q1_label", "q2_label", "q3_label", "q4_label", "q5_label", "q6_label", "q7_label"]
    
    dataset.set_index("tweet_id", inplace=True)
    dataset.drop(["q2_label", "q3_label", "q4_label", "q5_label", "q6_label", "q7_label"], axis=1, inplace=True)

    return dataset

def generatePredictionData(fileName: str) -> List[Tuple[str, List[str]]]:
    data = getData(fileName, False)
    prediction_data: List[Tuple[str, List[str]]] = list()

    for tweet_id, row in data.iterrows():
        prediction_data.append((tweet_id, row["text"].lower().split(" ")))
    
    return prediction_data

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
            true_label = testData.loc[row["tweet_id"]]["q1_label"]
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