# -*- coding: utf-8 -*-
"""
Created on Feb 20, 2023

@author: Meysam Ahangaran
"""

import numpy as np
import pandas as pd
from DataQuality import DataReadinessRecord
from ClassOverlap import ClassOverlap
from scipy import stats
import itertools
from random import randrange
import math
from Outlier import Outlier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statistics
from sklearn.cluster import KMeans
from numpy import unique
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool, cpu_count
import functools
from tqdm import tqdm
import timeit
from sklearn.linear_model import SGDClassifier
import json

# get all subsets of s with size n
def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def getAvgPC(colsArr, df):
    pc = 0
    subsets = findsubsets(colsArr, 2)
    numOfPairs = subsets.__len__()
    for i in range(numOfPairs):
        currSubset = subsets[i]
        pearson_coef, p_value = stats.pearsonr(df[currSubset[0]], df[currSubset[1]])
        if math.isnan(pearson_coef):
            numOfPairs = numOfPairs - 1
            continue
        pc = pc + abs(pearson_coef)
    
    return round(pc/numOfPairs, precision)

def getAvgSpearmanCorr(colsArr, df):
    spearmanCorr = 0
    subsets = findsubsets(colsArr, 2)
    numOfPairs = subsets.__len__()
    for i in range(numOfPairs):
        currSubset = subsets[i]
        spearman_coef, p_value = stats.spearmanr(df[currSubset[0]], df[currSubset[1]])
        if math.isnan(spearman_coef):
            numOfPairs = numOfPairs - 1
            continue
        spearmanCorr = spearmanCorr + abs(spearman_coef)
    
    return round(spearmanCorr/numOfPairs, precision)

def getNullRatio(df):
    rows = len(df.index)
    cols = len(df.columns)
    totalElem = rows*cols
    nullRatio = df.isnull().values.sum() / totalElem
    return round(nullRatio, precision)

def getOutlierUnivariate(df):
    outlier = Outlier()
    rows = len(df.index)
    cols = len(df.columns)
    totalElem = rows*cols
    nonNullElements = totalElem - (df.isnull().values.sum())
    totalOutliers = 0
    for i in range(cols):
        currCol = df.columns[i]
        _,numOfOutliers = outlier.mad_method(df, currCol)
        totalOutliers = totalOutliers + numOfOutliers
    
    avgOutliers = totalOutliers / nonNullElements
    return round(avgOutliers, precision)
    
def createDF_ZeroFile(df, fileName):
    df.to_csv(fileName, index = False, na_rep='0')    
########################################################################################
#json config file data
print('DREAMER started!')
print('Read json configuration.')
config = {}
with open("DREAMER_Config.json", 'r') as config_file:
          config = json.loads(config_file.read())
search_space = config['search_space']
data_settings = config['data_settings']
free_params = config['free_params']
     
d = search_space['subtables']    #number of sub-tables in each run of the algorithm
R = search_space['cores']   #number of cores for multiprocessing (0 for getting all cores)
file_path = data_settings['file_path']   #main CSV file including empty cells
target_column = data_settings['target_column']  #target column of the dataset for supervised learning
clustering_weight = free_params['clustering_weight']
classification_weight = free_params['classification_weight']
RER = free_params['rows_exclusion_ratio']   #rows exclusion ratio (excluded rows < RER)
CER = free_params['columns_exclusion_ratio']   #columns exclusion ratio (excluded columns < CER)
precision = free_params['precision'] #precision of floating numbers
if R == 0:
   R = cpu_count() #get all CPU cores

print(f'The process runs on {R} cores.')

print('Read master dataset.')
numOfQualityMeasures = 5    #number of data quality measures

# 2-D matrix of runs with dimension [Rxd]
runsArray = [[DataReadinessRecord() for k in range(d)] for p in range(R)]

#new object of data readiness recoord for master table
readinessRecord_master = DataReadinessRecord() 

df_master = pd.read_csv(file_path) #original data frame including nan elements and target column

fileNameLength = len(file_path)
fileName = file_path[0:fileNameLength-4]    #file name without .csv extension
dfZero_fileName = fileName+'_Zero.csv'

createDF_ZeroFile(df_master, dfZero_fileName)

print('Zero dataset file was created.')
dfZero = pd.read_csv(dfZero_fileName)
dfZero_orig = dfZero.copy()
del dfZero[target_column]
df = pd.read_csv(file_path) #data frame including nan elements without target column
del df[target_column]
dfRows = len(dfZero.index) #number of rows in data frame
dfCols = len(dfZero.columns) #number of columns in data frame
minNumOfRows = math.floor(dfRows - (dfRows * RER))  #minimum number of rows in random selection
minNumOfCols = math.floor(dfCols - (dfCols * CER))  #minimum number of columns in random selection


#convert dataframe of subtable into a 2-D matrix       
def getMat_df(dfRandRows):
    indexes = dfRandRows.index
    randRows = len(indexes)
    target = np.zeros(randRows, dtype=str)
    for i in range(len(indexes)):
        target[i] = dfZero_orig.xs(indexes[i])[target_column]
    
    new_df = dfRandRows.copy()
    
    new_df[target_column] = target
    dfRows_new = len(new_df.index) #number of rows in new data frame
    dfCols_new = len(new_df.columns) #number of columns in new data frame
    df_mat = np.zeros((dfRows_new, dfCols_new), dtype=str) 
    df_mat = new_df.to_numpy()
    
    return df_mat


def getClassifyAccuracy(df_mat):
    cols = len(df_mat[0])
    X = df_mat[:,0:(cols-1)]
    y = df_mat[:, (cols-1)]
    
    #RF classifier
    clf_RF = RandomForestClassifier(n_estimators=20)
    scores_RF = cross_val_score(clf_RF, X, y, scoring='accuracy', cv=10)
    accuracy_RF = round(scores_RF.mean(), precision)
    
    #SGD classifier
    clf_SGD = SGDClassifier(shuffle=True, loss='log')
    scores_SGD = cross_val_score(clf_SGD, X, y, scoring='accuracy', cv=10)
    accuracy_SGD = round(scores_SGD.mean(), precision)
    
    accuracy_avg = (accuracy_RF+accuracy_SGD)/2
    
    return np.round(accuracy_avg, precision)


def getClusteringAccuracy(df_mat):
    cols = len(df_mat[0])
    X = df_mat[:,0:(cols-1)] #excluding the last column target
    
    #Agglomerative
    model_Agg = AgglomerativeClustering()
    yhat_Agg = model_Agg.fit_predict(X)
    clusters_Agg = unique(yhat_Agg)
    labels_Agg = model_Agg.labels_
    score_Agg = metrics.silhouette_score(X, labels_Agg, metric='euclidean')
    if score_Agg < 0:
        score_Agg = 0
    
    #K-Means
    model_KM = KMeans()    
    yhat_KM = model_KM.fit_predict(X)    
    clusters_KM = unique(yhat_KM)    
    labels_KM = model_KM.labels_    
    score_KM = metrics.silhouette_score(X, labels_KM, metric='euclidean')
    if score_KM < 0:
        score_KM = 0
        
    score_avg = (score_Agg + score_KM) / 2
    return np.round(score_avg, precision)
    
    
#create random sub-tables for training weights of qaulity measures

def getRandomSubtablesWeights(coreIdx):
    
        dataReadinessArray = [DataReadinessRecord() for k in range(d)]  #array of sub-tables
        for j in tqdm(range(d)):  #create a random sub-table
           
            readinessRecord = DataReadinessRecord() #new object of data readiness recoord
            randRows = randrange(minNumOfRows, dfRows+1)  #number of random rows
            randCols = randrange(minNumOfCols, dfCols+1)  #number of random columns
            
            dfRandCols = dfZero.sample(randCols, axis=1)    #random columns selection
            dfRandRows = dfRandCols.sample(randRows, axis=0)    #random rows selection
            
            #create the same dataframe with NaN elements
            dfNanRandCols = df[dfRandRows.columns]
            dfNanRandRows = dfNanRandCols.iloc[dfRandRows.index,:]
            
            df_mat = getMat_df(dfRandRows)  #df_random matrix
            
            accuracy_classify = getClassifyAccuracy(df_mat)
            
            accuracy_clustering = getClusteringAccuracy(df_mat)
            accuracy_avg = (clustering_weight*accuracy_clustering)+(classification_weight*accuracy_classify)
            classOverlapObj = ClassOverlap()
           
            
            #calculating rows and columns of a dataframe
            readinessRecord.rowsIndexes = dfRandRows.index #index of rows of the subtable
            readinessRecord.featuresList = dfRandRows.columns #columns names of the subtable
            
            avgPC = getAvgPC(dfRandRows.columns,dfZero)
            readinessRecord.avgPCMeasure = round(1-avgPC, precision) #average PC of current subtable (1-PC)
            
            avgSpearmanCorr = getAvgSpearmanCorr(dfRandRows.columns, dfZero)
            readinessRecord.avgSpearmanCorr = round(1-avgSpearmanCorr, precision) #average Spearman Corr of current subtable (1-SpearmanCorr)
            
            nullRatio = getNullRatio(dfNanRandRows) #calculate null ratio for the current sub-table
            
            outlierUnivariate = getOutlierUnivariate(dfNanRandRows) #univariate outliers for the current sub-table
            readinessRecord.nullRatioMeasure = round(1-nullRatio, precision) #values near 1 are better [0,1]
            readinessRecord.outlierUnivariate = round(1-outlierUnivariate, precision) #values near 1 are better
            
            classOverlap = 1 - classOverlapObj.getRvalue(df_mat) #values near 1 are better
            readinessRecord.accuracy_classify =round(accuracy_classify, precision) #accuracy of classification
            readinessRecord.accuracy_clustering = round(accuracy_clustering, precision)
            readinessRecord.accuracy_avg = round(accuracy_avg, precision)
            readinessRecord.classOverlap = round(classOverlap, precision)
            dataReadinessArray[j] = readinessRecord
            
        return dataReadinessArray    
    
    
#print master table data quality details
def printMasterTableFile():
    
    #calculate the data quality of master table
    readinessRecord_master.rowsIndexes = dfZero.index #index of rows of the master table
    readinessRecord_master.featuresList = dfZero.columns #columns names of the master table
    df_mat_master = getMat_df(dfZero)   #matrix of the master table
    accuracy_classify_master = getClassifyAccuracy(df_mat_master)  #classify accuracy of the master table
    accuracy_clustering_master = getClusteringAccuracy(df_mat_master)    #clustering accuracy of the master table
    accuracy_avg_master = statistics.mean([accuracy_classify_master, accuracy_clustering_master])
    classOverlapObj_master = ClassOverlap()
    avgPC_master = getAvgPC(dfZero.columns,dfZero)
    readinessRecord_master.avgPCMeasure = round(1-avgPC_master, precision) #average PC of the master table (1-PC)
    avgSpearmanCorr_master = getAvgSpearmanCorr(dfZero.columns, dfZero)
    readinessRecord_master.avgSpearmanCorr = round(1-avgSpearmanCorr_master, precision) #average Spearman Corr of the master table (1-SpearmanCorr)
    
    nullRatio_master = getNullRatio(df) #calculate null ratio for the master table
    outlierUnivariate_master = getOutlierUnivariate(df) #univariate outliers for the master table
    readinessRecord_master.nullRatioMeasure = round(1-nullRatio_master, precision) #values near 1 are better [0,1]
    readinessRecord_master.outlierUnivariate = round(1-outlierUnivariate_master, precision) #values near 1 are better

    classOverlap_master = 1 - classOverlapObj_master.getRvalue(df_mat_master) #values near 1 are better
    readinessRecord_master.accuracy_classify =round(accuracy_classify_master, precision) #accuracy of classification
    readinessRecord_master.accuracy_clustering = round(accuracy_clustering_master, precision)
    readinessRecord_master.accuracy_avg = round(accuracy_avg_master, precision)
    readinessRecord_master.classOverlap = round(classOverlap_master, precision)
    
    masterTable = np.zeros((2,11), dtype=object)
    masterTable[0][0] = "Rows"
    masterTable[0][1] = "Columns"
    masterTable[0][2] = "1-PC"
    masterTable[0][3] = "1-SpearmanCorrelation"
    masterTable[0][4] = "1-NullRatio"
    masterTable[0][5] = "1-Outliers"
    masterTable[0][6] = "1-ClassOverlap"
    masterTable[0][7] = "Classify Accuracy"
    masterTable[0][8] = "Clustering Accuracy"
    masterTable[0][9] = "Average Accuracy"
    masterTable[0][10] = "Total Quality"
    
    rows = len(readinessRecord_master.rowsIndexes)
    cols = len(readinessRecord_master.featuresList)
    avgPC = readinessRecord_master.avgPCMeasure
    spearmanCorr = readinessRecord_master.avgSpearmanCorr
    nullRatio = readinessRecord_master.nullRatioMeasure
    outliers = readinessRecord_master.outlierUnivariate
    classOverlap = readinessRecord_master.classOverlap
    classifyAccuracy = readinessRecord_master.accuracy_classify
    clusteringAccuracy = readinessRecord_master.accuracy_clustering
    avgAccuracy = readinessRecord_master.accuracy_avg
    totalQuality = readinessRecord_master.getTotalQuality(precision)
    
    masterTable[1][0] = rows
    masterTable[1][1] = cols
    masterTable[1][2] = avgPC
    masterTable[1][3] = spearmanCorr
    masterTable[1][4] = nullRatio
    masterTable[1][5] = outliers
    masterTable[1][6] = classOverlap
    masterTable[1][7] = classifyAccuracy
    masterTable[1][8] = clusteringAccuracy
    masterTable[1][9] = avgAccuracy
    masterTable[1][10] = totalQuality
    
    path = "./Output/Runs/MasterTableDetails.csv"
    np.savetxt(path, masterTable, fmt="%s", delimiter=",")
    
    
#print all sub-tables of all runs into separate files
def printSubtableFilesRuns():
    print('Exporting DREAMER results into output files ...')
    runTable = np.zeros((d+1,11), dtype=object) #table for each run (sub-tables of the run)
    runTable[0][0] = "Rows"
    runTable[0][1] = "Columns"
    runTable[0][2] = "1-PC"
    runTable[0][3] = "1-SpearmanCorrelation"
    runTable[0][4] = "1-NullRatio"
    runTable[0][5] = "1-Outliers"
    runTable[0][6] = "1-ClassOverlap"
    runTable[0][7] = "Classify Accuracy"
    runTable[0][8] = "Clustering Accuracy"
    runTable[0][9] = "Average Accuracy"
    runTable[0][10] = "Total Quality"

    
    totalRunTable = np.zeros((R+1,11), dtype=object) #Table includes best sub-tables of each run
    totalRunTable[0][0] = "Rows"
    totalRunTable[0][1] = "Columns"
    totalRunTable[0][2] = "1-PC"
    totalRunTable[0][3] = "1-SpearmanCorrelation"
    totalRunTable[0][4] = "1-NullRatio"
    totalRunTable[0][5] = "1-Outliers"
    totalRunTable[0][6] = "1-ClassOverlap"
    totalRunTable[0][7] = "Classify Accuracy"
    totalRunTable[0][8] = "Clustering Accuracy"
    totalRunTable[0][9] = "Average Accuracy"
    totalRunTable[0][10] = "Total Quality"
    
    
    for i in range(R):
        for j in range(d):
            readinessRecord = runsArray[i][j]
            rows = len(readinessRecord.rowsIndexes)
            cols = len(readinessRecord.featuresList)
            avgPC = readinessRecord.avgPCMeasure
            spearmanCorr = readinessRecord.avgSpearmanCorr
            nullRatio = readinessRecord.nullRatioMeasure
            outliers = readinessRecord.outlierUnivariate
            classOverlap = readinessRecord.classOverlap
            classifyAccuracy = readinessRecord.accuracy_classify
            clusteringAccuracy = readinessRecord.accuracy_clustering
            avgAccuracy = readinessRecord.accuracy_avg
            totalQuality = readinessRecord.getTotalQuality(precision)
            
            runTable[j+1][0] = rows
            runTable[j+1][1] = cols
            runTable[j+1][2] = avgPC
            runTable[j+1][3] = spearmanCorr
            runTable[j+1][4] = nullRatio
            runTable[j+1][5] = outliers
            runTable[j+1][6] = classOverlap
            runTable[j+1][7] = classifyAccuracy
            runTable[j+1][8] = clusteringAccuracy
            runTable[j+1][9] = avgAccuracy
            runTable[j+1][10] = totalQuality

        currRun = i+1
        path = "./Output/Runs/Run"+str(currRun)+".csv"
        np.savetxt(path, runTable, fmt="%s", delimiter=",")
        
        #find best sub-table
        bestSubtableIdx = getMaxQualityIdx(i) #best sub-table of current run
        bestSubtable = runsArray[i][bestSubtableIdx]
        
        rowsBest = len(bestSubtable.rowsIndexes)
        colsBest = len(bestSubtable.featuresList)
        avgPCBest = bestSubtable.avgPCMeasure
        spearmanCorrBest = bestSubtable.avgSpearmanCorr
        nullRatioBest = bestSubtable.nullRatioMeasure
        outliersBest = bestSubtable.outlierUnivariate
        classOverlapBest = bestSubtable.classOverlap
        classifyAccuracyBest = bestSubtable.accuracy_classify
        clusteringAccuracyBest = bestSubtable.accuracy_clustering
        avgAccuracyBest = bestSubtable.accuracy_avg
        totalQualityBest = bestSubtable.getTotalQuality(precision)
        
        totalRunTable[i+1][0] = rowsBest
        totalRunTable[i+1][1] = colsBest
        totalRunTable[i+1][2] = avgPCBest
        totalRunTable[i+1][3] = spearmanCorrBest
        totalRunTable[i+1][4] = nullRatioBest
        totalRunTable[i+1][5] = outliersBest
        totalRunTable[i+1][6] = classOverlapBest
        totalRunTable[i+1][7] = classifyAccuracyBest
        totalRunTable[i+1][8] = clusteringAccuracyBest
        totalRunTable[i+1][9] = avgAccuracyBest
        totalRunTable[i+1][10] = totalQualityBest
 
    np.savetxt("./Output/Runs/Run_BestSubtables.csv", totalRunTable, fmt="%s", delimiter=",")
    bestSubtable = getBestSubtable() #best subtable among all runs
    printBestSubtableToFile(bestSubtable)
    
#print all sub-tables of all runs for training weights into separate files
def printFileRunsWeights():
    print('print sub-tables in files.')
    runTable = np.zeros((d+1,10), dtype=object) #table for each run (sub-tables of the run)
    runTable[0][0] = "Rows"
    runTable[0][1] = "Columns"
    runTable[0][2] = "1-PC"
    runTable[0][3] = "1-SpearmanCorrelation"
    runTable[0][4] = "1-NullRatio"
    runTable[0][5] = "1-Outliers"
    runTable[0][6] = "1-ClassOverlap"
    runTable[0][7] = "Classify Accuracy"
    runTable[0][8] = "Clustering Accuracy"
    runTable[0][9] = "Average Accuracy"
    
    for i in range(R):
        for j in range(d):
            readinessRecord = runsArray[i][j]
            rows = len(readinessRecord.rowsIndexes)
            cols = len(readinessRecord.featuresList)
            avgPC = readinessRecord.avgPCMeasure
            spearmanCorr = readinessRecord.avgSpearmanCorr
            nullRatio = readinessRecord.nullRatioMeasure
            outliers = readinessRecord.outlierUnivariate
            classOverlap = readinessRecord.classOverlap
            accuracy_classify = readinessRecord.accuracy_classify
            accuracy_clustering = readinessRecord.accuracy_clustering
            accuracy_avg = readinessRecord.accuracy_avg

            runTable[j+1][0] = rows
            runTable[j+1][1] = cols
            runTable[j+1][2] = avgPC
            runTable[j+1][3] = spearmanCorr
            runTable[j+1][4] = nullRatio
            runTable[j+1][5] = outliers
            runTable[j+1][6] = classOverlap
            runTable[j+1][7] = accuracy_classify
            runTable[j+1][8] = accuracy_clustering
            runTable[j+1][9] = accuracy_avg
        currRun = i+1
        path = "./Output/Runs/Weights/Run_Weights"+str(currRun)+".csv"
        np.savetxt(path, runTable, fmt="%s", delimiter=",")
            
#Print best sub-table details into a file
def printBestSubtableToFile(bestSubtable):
    file = open("./Output/Runs/BestSubtable.txt", 'w')
    columns = bestSubtable.featuresList
    rows = bestSubtable.rowsIndexes
    rowsBest = len(rows)
    colsBest = len(columns)
    avgPCBest = bestSubtable.avgPCMeasure
    spearmanCorrBest = bestSubtable.avgSpearmanCorr
    nullRatioBest = bestSubtable.nullRatioMeasure
    outliersBest = bestSubtable.outlierUnivariate
    classOverlapBest = bestSubtable.classOverlap
    classifyAccuracyBest = bestSubtable.accuracy_classify
    clusteringAccuracyBest = bestSubtable.accuracy_clustering
    avgAccuracyBest = bestSubtable.accuracy_avg
    totalQualityBest = bestSubtable.getTotalQuality(precision)
    file.write("Best sub-table as global optimum:\n")
    rowsBestStr = "Number of rows = " + str(rowsBest) + str('\n')
    file.write(rowsBestStr)
    colsBestStr = "Number of columns = " + str(colsBest) + str('\n')
    file.write(colsBestStr)
    PCBestStr = "PC measure = " + str(avgPCBest) + str('\n')
    file.write(PCBestStr)
    spearmanBestStr = "Spearman Correlation = " + str(spearmanCorrBest) + str('\n')
    file.write(spearmanBestStr)
    nullRatioBestStr = "Null ratio measure = " + str(nullRatioBest) + str('\n')
    file.write(nullRatioBestStr)
    outliersBestStr = "Outliers = " + str(outliersBest) + str('\n')
    file.write(outliersBestStr)
    classOverlapStr = "Class Overlap = " + str(classOverlapBest) + str('\n')
    file.write(classOverlapStr)
    
    classifyAccuracyStr = "Classify accuracy = " + str(classifyAccuracyBest) + str('\n')
    file.write(classifyAccuracyStr)
    clusteringAccuracyStr = "Clustering accuracy = " + str(clusteringAccuracyBest) + str('\n')
    file.write(clusteringAccuracyStr)
    avgAccuracyStr = "Average accuracy = " + str(avgAccuracyBest) + str('\n')
    file.write(avgAccuracyStr)
    
    totalQualityBestStr = "Total quality = " + str(totalQualityBest) + str('\n\n')    
    file.write(totalQualityBestStr)
    
    #print list of columns
    file.write('-----------------------------------------------------------------------\n')
    columnsListBestStr = "Columns = \n"
    for i in range(colsBest):
        columnsListBestStr = columnsListBestStr + str(columns[i]) + ","
        if i % 10 == 0 and i != 0:
            columnsListBestStr = columnsListBestStr + "\n"
    columnsListBestStr = columnsListBestStr + str('\n\n')
    file.write(columnsListBestStr)
    
    #print list of rows indexes
    file.write('-----------------------------------------------------------------------\n')
    rowsListBestStr = "Rows indexes = \n"
    for i in range(rowsBest):
        rowsListBestStr = rowsListBestStr + str(rows[i]) + ","
        if i % 20 == 0 and i != 0:
            rowsListBestStr = rowsListBestStr + "\n"
    rowsListBestStr = rowsListBestStr + str('\n\n')
    file.write(rowsListBestStr)
    
    file.close()
    
    #generate the final clean data after harmonization
    columnsTarget = np.zeros(colsBest+1, dtype=object)
    columnsTarget[0] = target_column
    for i in range(colsBest):
        columnsTarget[i+1] = columns[i]
    
    best_df_cols = df_master.loc[:, columnsTarget]
    best_df_rows = best_df_cols.loc[rows, :]
    path_CleanData = "./Output/Runs/CleanData.csv"
    best_df_rows.to_csv(path_CleanData)
    
    #Print best subtable details in a CSV file
    bestTable = np.zeros((2,11), dtype=object)
    bestTable[0][0] = "Rows"
    bestTable[0][1] = "Columns"
    bestTable[0][2] = "1-PC"
    bestTable[0][3] = "1-SpearmanCorrelation"
    bestTable[0][4] = "1-NullRatio"
    bestTable[0][5] = "1-Outliers"
    bestTable[0][6] = "1-ClassOverlap"
    bestTable[0][7] = "Classify Accuracy"
    bestTable[0][8] = "Clustering Accuracy"
    bestTable[0][9] = "Average Accuracy"
    bestTable[0][10] = "Total Quality"
    
    rows_best = len(bestSubtable.rowsIndexes)
    cols_best = len(bestSubtable.featuresList)
    avgPC = bestSubtable.avgPCMeasure
    spearmanCorr = bestSubtable.avgSpearmanCorr
    nullRatio = bestSubtable.nullRatioMeasure
    outliers = bestSubtable.outlierUnivariate
    classOverlap = bestSubtable.classOverlap
    classifyAccuracy = bestSubtable.accuracy_classify
    clusteringAccuracy = bestSubtable.accuracy_clustering
    avgAccuracy = bestSubtable.accuracy_avg
    totalQuality = bestSubtable.getTotalQuality(precision)
    
    bestTable[1][0] = rows_best
    bestTable[1][1] = cols_best
    bestTable[1][2] = avgPC
    bestTable[1][3] = spearmanCorr
    bestTable[1][4] = nullRatio
    bestTable[1][5] = outliers
    bestTable[1][6] = classOverlap
    bestTable[1][7] = classifyAccuracy
    bestTable[1][8] = clusteringAccuracy
    bestTable[1][9] = avgAccuracy
    bestTable[1][10] = totalQuality
    
    path = "./Output/Runs/BestTableDetails.csv"
    np.savetxt(path, bestTable, fmt="%s", delimiter=",")
    
#get best sub-table with max total quality    
def getBestSubtable():
    # maxQualityIdx = -1
    maxQuality = 0
    bestSubtable = DataReadinessRecord()
    for i in range(R):
        for j in range(d):
            readinessRecord = runsArray[i][j]
            if readinessRecord.getTotalQuality(precision) > maxQuality:
                maxQuality = readinessRecord.getTotalQuality(precision)
                # maxQualityIdx = i
                bestSubtable = readinessRecord
    
    return bestSubtable

#get sub-table with max total quality
def getMaxQualityIdx(runIdx):
    maxQualityIdx = -1
    maxQuality = 0
    for i in range(d):
        readinessRecord = runsArray[runIdx][i]
        if readinessRecord.getTotalQuality(precision) > maxQuality:
            maxQuality = readinessRecord.getTotalQuality(precision)
            maxQualityIdx = i
    
    return maxQualityIdx
    
#Train weights of data quality measures
def learnWeights():
    print('Learning weights of data quality measures ...')
    weightsTable = np.zeros((R+1,numOfQualityMeasures), dtype=object) #weights for each run (sub-tables of the run)
    weightsTable[0][0] = "PC"
    weightsTable[0][1] = "Spearman"
    weightsTable[0][2] = "Null Ratio"
    weightsTable[0][3] = "Outliers"
    weightsTable[0][4] = "Class Overlap"
    
    file = "./Output/Runs/Weights/Weights.csv"
    
    weightsAvgTable = np.zeros((2,numOfQualityMeasures), dtype=object) #average weights for all runs 
    weightsAvgTable[0][0] = "PC"
    weightsAvgTable[0][1] = "Spearman"
    weightsAvgTable[0][2] = "Null Ratio"
    weightsAvgTable[0][3] = "Outliers"
    weightsAvgTable[0][4] = "Class Overlap"
    
    fileAvg = "./Output/Runs/Weights/Average_Weights.csv"
    
    pathOrig = "./Output/Runs/Weights/Run_Weights"
    
    for i in range(R):
            path = pathOrig + str(i+1) + ".csv"
            df = pd.read_csv(path, usecols=
                             ["1-PC", "1-SpearmanCorrelation", "1-NullRatio", "1-Outliers", "1-ClassOverlap", "Classify Accuracy", "Clustering Accuracy", "Average Accuracy"])        
            indexes = df.index
            dfRows = len(df.index) #number of rows in new data frame
            dfCols = len(df.columns) #number of columns in new data frame
            df_mat = np.zeros((dfRows, dfCols), dtype=object) 
    
            for r in range(dfRows):
                for c in range(dfCols):
                    df_mat[r][c] = df.xs(indexes[r])[c]
            
            cols = len(df_mat[0])
            """three last columns are classify, clustering, and average accuracies
            and the first two columns are row and column that should be ignored"""
            X = df_mat[:,0:(cols-3)] 
            y = df_mat[:, (cols-1)] #last column is the target value for regression
            clf = RandomForestRegressor(n_estimators=20)
            clf.fit(X, y)
            importance = clf.feature_importances_
            weightsTable[(i+1), :] = np.round(importance, precision)
            
    np.savetxt(file, weightsTable, fmt="%s", delimiter=",")
    
    for j in range(numOfQualityMeasures):
        weightsAvgTable[1][j] = round(statistics.mean(weightsTable[1:R+1, j]), precision)
    np.savetxt(fileAvg, weightsAvgTable, fmt="%s", delimiter=",")

#multiprocessing map
def smap(f):
    return f()

def runDREAMER():

    cores = R #number of CPU cores for multiprocessing
    f_array = np.empty(cores, dtype=object)
    
    for i in range(R):
        f_array[i] = functools.partial(getRandomSubtablesWeights,i)
    
    with Pool() as pool:
        res = pool.map(smap, f_array)
    
    # #RunsArray as a global variable
    global runsArray
    runsArray = res
    
    printFileRunsWeights()  #print subtables accuracies runs into files 
    learnWeights() #learn weights and print average weights
    printSubtableFilesRuns() #print subtables runs into files    
    printMasterTableFile()

if __name__ == '__main__':
    
    start = timeit.default_timer()  #Start timer
    print('Creating random sub-tables ...')
    runDREAMER()
    stop = timeit.default_timer()   #Stop timer
    file = open("./Output/Runs/Time.txt", 'w')
    duration = (stop - start) / 60
    duration = round(duration,2)
    print('DREAMER finished successfully.')
    file.write('Running time (minutes): ')
    file.write(str(duration))
    file.close()
