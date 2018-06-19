import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import clear_output, display
import numpy as np
import matplotlib.pyplot as plt
import EMRyeast36


def makeQCbuttons():
    b1 = widgets.Button(
        description=' accept',
        icon='check-square',
        tooltip='accept cell and continute')
    b2 = widgets.Button(
        description=' reject',
        icon='minus-square',
        tooltip='reject cell and continue')
    b3 = widgets.Button(
        description=' previous',
        icon='chevron-left',
        tooltip='go back to previously reviewed cell')
    b4 = widgets.Button(
        description=' start',
        icon='step-backward',
        tooltip='start analysis')
    return(b1,b2,b3,b4)

def makeQCoutputs():
    out1 = widgets.Output(layout=Layout(
            height='400px', width = '600px', border='solid'))
    out2 = widgets.Output(layout=Layout(
            border='solid'))
    return out1, out2

def makeQC_clickfunctions(randLookupStart, resultsData, df, pathList,
                          frameTitles):
    global randLookup
    global total
    total = len(df)
    randLookup = randLookupStart
    randArray = df['randomIdx']
    indexArray = [randArray.index[i] for i in range(len(randArray))]
    status_list = list(df['qcStatus'])
    out1,out2 = makeQCoutputs()
    
    def makeQC_Art1mNG(randLookup,resultsData,df):
        scalingFactors = [0.2,0.2]
        fieldIdx = int(df.loc[df['randomIdx'] == randLookup,'fieldIdx'])
        cellLbl = int(df.loc[df['randomIdx'] == randLookup,'localLbl'])
        rgbQC = resultsData['totalQC'][fieldIdx]
        mcl = resultsData['totalMcl'][fieldIdx]
        dvImage = EMRyeast36.basicDVreader(pathList[fieldIdx],rolloff=64)
        greenFluor = dvImage[1,3,:,:].astype(float)
        redFluor = np.amax(dvImage[0,:,:,:],axis=0).astype(float)
        qcDict = EMRyeast36.helpers.make_qcFrame(
                rgbQC, greenFluor, redFluor, mcl, cellLbl, scalingFactors, 10)
        return(qcDict) 
        
    def click_b1(b):
        global randLookup
        global total
        global statuslist
        location = indexArray[randLookup]
        status_list[location] = 'accepted'
        pastRandIdces = list(range(max(0,randLookup-5),randLookup+1))
        pastStatus = [status_list[pastloc] for 
                      pastloc in indexArray[max(0,randLookup-5):randLookup+1]]
        if randLookup == total-1:
            with out2:
                out1.clear_output()
            with out2:
                print('finshed')
        else:
            randLookup += 1
            qcDict = makeQC_Art1mNG(randLookup,resultsData,df)
            qcFrame = qcDict['qcFrame']
            redInvFrame = qcDict['redInvFrame']
            greenInvFrame = qcDict['greenInvFrame']
            frameSize = len(redInvFrame)
            with out1:
                fig = plt.figure(figsize=(12,8))
                qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
                qcAx.imshow(qcFrame)
                qcAx.axis('off')
                plt.title('Main display: blinded cell idx = '+str(randLookup))
                redAx = plt.subplot2grid((2,3), (0,2))
                redAx.imshow(redInvFrame, cmap='gray')
                redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
                redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
                redAx.xaxis.set_ticklabels([])
                redAx.yaxis.set_ticklabels([])
                redAx.grid()
                plt.title(frameTitles[0])
                grnAx = plt.subplot2grid((2,3), (1,2))
                grnAx.imshow(greenInvFrame, cmap='gray')
                grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
                grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
                grnAx.xaxis.set_ticklabels([])
                grnAx.yaxis.set_ticklabels([])
                grnAx.grid()
                plt.title(frameTitles[1])
                clear_output(wait=True)
                plt.show()
            with out2:
                out2.clear_output()
            with out2:
                print('status:\ncurrent cell (blind Idx):',
                      str(randLookup),
                      '\n\nprevious cells:\n')
                for pair in zip(reversed(pastRandIdces),reversed(pastStatus)):
                    print(pair)
            return(status_list)
                
    def click_b2(b):
        global randLookup
        global total
        global statuslist
        location = indexArray[randLookup]
        status_list[location] = 'rejected'
        pastRandIdces = list(range(max(0,randLookup-5),randLookup+1))
        pastStatus = [status_list[pastloc] for 
                      pastloc in indexArray[max(0,randLookup-5):randLookup+1]]
        if randLookup == total-1:
            with out2:
                out1.clear_output()
            with out2:
                print('finshed')
        else:
            randLookup += 1
            qcDict = makeQC_Art1mNG(randLookup,resultsData,df)
            with out1:
                qcFrame = qcDict['qcFrame']
                redInvFrame = qcDict['redInvFrame']
                greenInvFrame = qcDict['greenInvFrame']
                frameSize = len(redInvFrame)
                fig = plt.figure(figsize=(12,8))
                qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
                qcAx.imshow(qcFrame)
                qcAx.axis('off')
                plt.title('Main display: blinded cell idx = '+str(randLookup))
                redAx = plt.subplot2grid((2,3), (0,2))
                redAx.imshow(redInvFrame, cmap='gray')
                redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
                redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
                redAx.xaxis.set_ticklabels([])
                redAx.yaxis.set_ticklabels([])
                redAx.grid()
                plt.title(frameTitles[0])
                grnAx = plt.subplot2grid((2,3), (1,2))
                grnAx.imshow(greenInvFrame, cmap='gray')
                grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
                grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
                grnAx.xaxis.set_ticklabels([])
                grnAx.yaxis.set_ticklabels([])
                grnAx.grid()
                plt.title(frameTitles[1])
                
                clear_output(wait=True)
                plt.show()
            with out2:
                out2.clear_output()
            with out2:
                print('status:\ncurrent cell (blind Idx):',
                      str(randLookup),
                      '\n\nprevious cells:\n')
                for pair in zip(reversed(pastRandIdces),reversed(pastStatus)):
                    print(pair)
            return(status_list)
        
    def click_b3(b):
        global randLookup
        global total
        global statuslist
        randLookup -= 1
        pastRandIdces = list(range(max(0,randLookup-5),randLookup+1))
        pastStatus = [status_list[pastloc] for 
                      pastloc in indexArray[max(0,randLookup-5):randLookup+1]]
        qcDict = makeQC_Art1mNG(randLookup,resultsData,df)
        qcFrame = qcDict['qcFrame']
        redInvFrame = qcDict['redInvFrame']
        greenInvFrame = qcDict['greenInvFrame']
        frameSize = len(redInvFrame)
        with out1:
            fig = plt.figure(figsize=(12,8))
            qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
            qcAx.imshow(qcFrame)
            qcAx.axis('off')
            plt.title('Main display: blinded cell idx = '+str(randLookup))
            redAx = plt.subplot2grid((2,3), (0,2))
            redAx.imshow(redInvFrame, cmap='gray')
            redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
            redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
            redAx.xaxis.set_ticklabels([])
            redAx.yaxis.set_ticklabels([])
            redAx.grid()
            plt.title(frameTitles[0])
            grnAx = plt.subplot2grid((2,3), (1,2))
            grnAx.imshow(greenInvFrame, cmap='gray')
            grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
            grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
            grnAx.xaxis.set_ticklabels([])
            grnAx.yaxis.set_ticklabels([])
            grnAx.grid()
            plt.title(frameTitles[1])
            clear_output(wait=True)
            plt.show()
        with out2:
            out2.clear_output()
        with out2:
            print('status:\ncurrent cell (blind Idx):',
                  str(randLookup),
                  '\n\nprevious cells:\n')
            for pair in zip(reversed(pastRandIdces),reversed(pastStatus)):
                print(pair)
        
                
    def click_b4(b):
        global randLookup
        global total
        global statuslist
        randLookup = 0
        pastRandIdces = list(range(max(0,randLookup-5),randLookup+1))
        pastStatus = [status_list[pastloc] for 
                      pastloc in indexArray[max(0,randLookup-5):randLookup+1]]
        qcDict = makeQC_Art1mNG(randLookup,resultsData,df)
        qcFrame = qcDict['qcFrame']
        redInvFrame = qcDict['redInvFrame']
        greenInvFrame = qcDict['greenInvFrame']
        frameSize = len(redInvFrame)
        with out1:
            fig = plt.figure(figsize=(12,8))
            qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
            qcAx.imshow(qcFrame)
            qcAx.axis('off')
            plt.title('Main display: blinded cell idx = '+str(randLookup))
            redAx = plt.subplot2grid((2,3), (0,2))
            redAx.imshow(redInvFrame, cmap='gray')
            redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
            redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
            redAx.xaxis.set_ticklabels([])
            redAx.yaxis.set_ticklabels([])
            redAx.grid()
            plt.title(frameTitles[0])
            grnAx = plt.subplot2grid((2,3), (1,2))
            grnAx.imshow(greenInvFrame, cmap='gray')
            grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
            grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
            grnAx.xaxis.set_ticklabels([])
            grnAx.yaxis.set_ticklabels([])
            grnAx.grid()
            plt.title(frameTitles[1])
            
            clear_output(wait=True)
            plt.show()
        with out2:
            out2.clear_output()
        with out2:
            print('status:\ncurrent cell (blind Idx):',
                  str(randLookup),
                  '\n\nprevious cells:\n')
            for pair in zip(reversed(pastRandIdces),reversed(pastStatus)):
                print(pair)
            
                
    return(click_b1, click_b2, click_b3, click_b4, out1, out2, status_list)