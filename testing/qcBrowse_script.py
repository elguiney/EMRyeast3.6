'''
rgbQC evaluation
'''
import pickle
import matplotlib.pyplot as plt

targetFolder = 'C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/Art1Quant'
resultsDirectory = targetFolder + '/results/'
resultsName = '/2018-06-19_analysis.p'
resultsData = pickle.load(open(resultsDirectory+resultsName,'rb'))

start = 25
n = 25
for field in range(start,start+n):
    plt.figure(field)
    plt.imshow(resultsData['totalQC'][field])