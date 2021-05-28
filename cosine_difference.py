from scipy import spatial
import pandas
import numpy as np


def getPrediction(index, predictionDataFrame):
    predictionSplitter = predictionDataFrame.iloc[index][0].split(",[")
    predictionList = predictionSplitter[1]
    predictionList[:-2]
    predictionArray = np.fromstring(predictionList[1:], dtype=float, sep=",")
    maxIndex = np.argmax(predictionArray)
    return predictionArray, maxIndex


def getCosineDifference(predictionArray, predictionArray2):
    dataSetI = predictionArray.tolist()
    dataSetII = predictionArray2.tolist()
    result = spatial.distance.cosine(dataSetI, dataSetII)
    # print("cosine difference between: ")
    # # print(dataSetI)
    # # print(dataSetII)
    # print("is: " + str(result))
    return result
    # print(dictionaryDataFrame.iloc[0][1])
    # print(dictionaryDataFrame.iloc[1][1])


dictionaryDataFrame = pandas.read_csv("saved/word_dict.txt")
predictionDataFrame = pandas.read_csv("saved/predictions.txt", sep="\n")
indexWar = 620
indexWife = 858
indexLight = 737
indexLucio = 363
indexFlesh = 276
macbethIndex = 746
swordIndex = 779

print("====== war ======")
print(dictionaryDataFrame.iloc[indexWar][1])
predictionArray1, maxIndex = getPrediction(indexWar, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== wife ======")
print(dictionaryDataFrame.iloc[indexWife][1])
predictionArray2, maxIndex = getPrediction(indexWar, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== light ======")
print(dictionaryDataFrame.iloc[indexLight][1])
predictionArray3, maxIndex = getPrediction(indexWar, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== lucio ======")
print(dictionaryDataFrame.iloc[indexLucio][1])
predictionArray4, maxIndex = getPrediction(indexWar, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== flesh ======")
print(dictionaryDataFrame.iloc[indexFlesh][1])
predictionArray, maxIndex = getPrediction(indexFlesh, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== macbeth ======")
print(dictionaryDataFrame.iloc[macbethIndex][1])
predictionArray, maxIndex = getPrediction(indexFlesh, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== sword ======")
print(dictionaryDataFrame.iloc[swordIndex][1])
predictionArray5, maxIndex = getPrediction(indexFlesh, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("cosine difference between war and wife:")
cosineDifference = getCosineDifference(predictionArray1, predictionArray2)
print(cosineDifference)

print("cosine difference between war and light:")
cosineDifference = getCosineDifference(predictionArray1, predictionArray3)
print(cosineDifference)

print("cosine difference between war and lucio:")
cosineDifference = getCosineDifference(predictionArray1, predictionArray4)
print(cosineDifference)

print("cosine difference between war and sword:")
cosineDifference = getCosineDifference(predictionArray1, predictionArray5)
print(cosineDifference)

palaceIndex = 599
buckinghamIndex = 726

print("====== palace ======")
print(dictionaryDataFrame.iloc[macbethIndex][1])
predictionArrayP, maxIndex = getPrediction(palaceIndex, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("====== buckingham ======")
print(dictionaryDataFrame.iloc[macbethIndex][1])
predictionArrayB, maxIndex = getPrediction(buckinghamIndex, predictionDataFrame)
print(dictionaryDataFrame.iloc[maxIndex][1])

print("cosine difference between buckingham and palace:")
cosineDifference = getCosineDifference(predictionArrayP, predictionArrayB)
print(cosineDifference)
