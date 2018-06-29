''' 
testing scripts

test test testf
'''
import ympy

folderPath = ('C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/'
              '2018-06-29_Art1wt-YPD-timcourse_exp1')

testPipeline = ympy.pipelines.GFPwMarkerPipeline(measureFields=[0,1])
testPipeline.initialize(folderPath)
testPipeline.returnTotalResults = True
resultsDic = testPipeline.runPipeline()