#Affective Computing Project 2
#Pain Recognition from Physiologicol responses to pain
#
#Zaber Raiyan Choudhury     U67301429

#Classifier:                            Random Forest
#Data:                                  Normalized Original Data
#
#                        command: Project2Data.csv
#Data Type:                         
#                        command: dia               BP Dia_mmHg
#                        command: eda               EDA_microsiemens
#                        command: sys               LA Systolic BP_mmHg
#                        command: res               Respiration Rate_BPM
#                        command: all               Fusion (All Data Together) Shuffle = True

#run python <filename> <datatype> <datafilename>
#keep Datafile in Dir

#import lib

from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import argparse
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

#Read File from Dir
def read_file(datafile, datatype):

    # Function Reads CSV file
    # Returns the only datatype asked for

    #initialization
    dia, eda, sys, res = [], [], [], []
    y_dia, y_eda, y_sys, y_res = [], [], [], []

    #read csv file
    with open(datafile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            list = None
            list = (row[3::])
            list = np.array([float(i) for i in list])
            mean, var, min, max = np.mean(list), np.var(list), np.min(list), np.max(list)
            list = [mean, var, min, max]
            
            if row[1] == "BP Dia_mmHg":
                dia.append(list)
                y_dia.append(row[2])
            if row[1] == "EDA_microsiemens":
                eda.append(list)
                y_eda.append(row[2])
            if row[1] == "LA Systolic BP_mmHg":
                sys.append(list)
                y_sys.append(row[2])
            if row[1] == "Respiration Rate_BPM":
                res.append(list)
                y_res.append(row[2])
    
    dia, y_dia = np.array(dia), np.array(y_dia)
    eda, y_eda = np.array(eda), np.array(y_eda)
    sys, y_sys = np.array(sys), np.array(y_sys)
    res, y_res = np.array(res), np.array(y_res)

    #return the selected datatype
    if datatype == "dia":
        return dia, y_dia
    elif datatype == "sys":
        return sys, y_sys
    elif datatype == "eda":
        return eda, y_eda
    elif datatype == "res":
        return res, y_res
    elif datatype == "all":
        all, y_all = [], [] #append all data
        all.append(dia)
        all.append(eda)
        all.append(sys)
        all.append(res)
        y_all.append(y_dia)
        y_all.append(y_eda)
        y_all.append(y_sys)
        y_all.append(y_res)

        #reshape to (n, 4)
        all, y_all = np.array(all), np.array(y_all)
        all, y_all = np.reshape(all, (480,4)), np.reshape(y_all, (480))
        return all, y_all

#Model Classifier
def RF (X, y):
  
  #Random Forest
  clf = RandomForestClassifier()
  pred=[]
  test_indices=[]
  
  #10-fold cross validation
  kf = KFold(n_splits=10, shuffle=True)
  for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    #train classifier
    clf.fit(X[train_index], y[train_index])
    #get predictions and save
    pred.append(clf.predict(X[test_index]))
    #save current test index
    test_indices.append(test_index)
  
  return pred, test_indices, y

#Print Evaluation
def EvalMetrics(pred, indices, y):

    #manually merge predictions and testing labels from each of the folds to make confusion matrix
    finalPredictions = []
    groundTruth = []
    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    
    #return test values to average later
    cm = confusion_matrix(groundTruth, finalPredictions)
    p = precision_score(groundTruth, finalPredictions, average='macro')
    r = recall_score(groundTruth, finalPredictions, average='macro')
    a = accuracy_score(groundTruth, finalPredictions)

    return cm, p, r, a

#initializing Box Plot Function
def BoxPlot (data, title):
    #labels
    labels = ['mean', 'var', 'min', 'max']
    plt.boxplot(data, vert=True,
                patch_artist=True,  
                labels=labels)
    
    plt.title(f'{title}')
    plt.show()

#data Variety Graph
def DataGraph():
    graph = []

    #read data
    with open('Project2Data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            list = None
            list = (row[3::])
            list = np.array([float(i) for i in list])

            graph.append(list)        
            line_count += 1
            if line_count > 7:
                break
    
    #Plot No Pain Graph
    f = plt.figure(1)
    plt.title("No Pain")
    plt.plot(graph[0], label = "BP Dia_mmHg")
    plt.plot(graph[1], label = "EDA_microsiemens")
    plt.plot(graph[2], label = "LA Systolic BP_mmHg")
    plt.plot(graph[3], label = "Respiration Rate_BPM")
    plt.legend()
    
    #Plot Pain Graph
    g= plt.figure(2)
    plt.title("Pain")
    plt.plot(graph[4], label = "BP Dia_mmHg")
    plt.plot(graph[5], label = "EDA_microsiemens")
    plt.plot(graph[6], label = "LA Systolic BP_mmHg")
    plt.plot(graph[7], label = "Respiration Rate_BPM")
    plt.legend()
    
    plt.show()

#Argument Parser
parser = argparse.ArgumentParser(description='Project 2')
parser.add_argument('datatype', nargs='?', type=str, default='dia', help='diastolic is default')
parser.add_argument('datafile', nargs='?', type=str, default='Project2Data.csv')
args = parser.parse_args()

#MAIN:
#Call function to read and get data from file
X, y = read_file(args.datafile, args.datatype)

#normalization
X = X / np.linalg.norm(X)

#Initialization
matrix, avg_precision, avg_recall, avg_accuracy = [], [], [], []


#Running experiment 10 times to get average: Confusion Matrix, Precision, Recall, Accuracy

for test in range(0,10):
    #Run classifier
    pred, test_indices, y = RF(X, y)
    #Call Eval
    cm, prec, rec, acc = EvalMetrics(pred, test_indices, y)
    #print(test)
    
    #append to get average
    matrix.append(cm)
    avg_precision.append(prec)
    avg_recall.append(prec)
    avg_accuracy.append(acc)

#Get Mean
matrix = np.array(matrix)
matrix = matrix.mean(0)
avg_precision = np.array(avg_precision)
avg_precision = avg_precision.mean(0)
avg_recall = np.array(avg_recall)
avg_recall = avg_recall.mean(0)
avg_accuracy = np.array(avg_accuracy)
avg_accuracy = avg_accuracy.mean(0)

#Print Metrics
print("Precision: ", avg_precision)
print("Recall: ", avg_recall)
print("Accuracy: " , avg_accuracy)

#Confusion Matrix Display
hash = ["No Pain", "Pain" ] #Labels

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= hash)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=hash, yticklabels=hash)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#Box Plot Function
BoxPlot(X, args.datatype)

#Data Variety Function
DataGraph()