#!/usr/bin/env python

import mioAnalysis as an
import numpy as np
import fnmatch
import os
import shelve
import matplotlib.pylab as plt

class DetectOccultation(object):
    """
    class for all the methods of detecting TNOs in MIOSOTYS data
    """
    def __init__(self):
        pass
        
    def setTimeSeriesData(self, time, originFlux, originQuality):
        self.time = time
        # if the flux value is below zero, then reset the value to zero
        ineg = np.where(originFlux <= 0.0)
        originFlux[ineg] = 0.0
        self.flux = originFlux
        originQuality[ineg] = 0
        self.quality = originQuality
        
    def setWindowSizeInSecond(self, windowSizeInSecond):
        self.windowSizeInBin = (windowSizeInSecond/0.05) + 1
    
    def setKaiserWindowBeta(self, beta):
        self.beta = beta
    
    def setThresholdLevel(self, thresholdLevelInSigma):
        self.thresholdLevel = thresholdLevelInSigma
            
    def smoothWKaiserWindow(self):
        """ 
        kaiser window smoothing 
        """
        # extending the data at beginning and at the end in oder to apply the window at the borders
        s = np.r_[self.flux[self.windowSizeInBin-1:0:-1], self.flux, self.flux[-1:-self.windowSizeInBin:-1]]
        w = np.kaiser(self.windowSizeInBin, self.beta)
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y[(self.windowSizeInBin-1)/2:len(y) - (self.windowSizeInBin-1)/2]
    
    def setFluxNormalised(self):
        """
        normalise the original lightcurves
        """  
        normFlux = self.flux/self.smoothWKaiserWindow()
        self.normFlux = normFlux
        
    def getDynamicVariance(self):
        """
        create an array of dynamic variance for each bin, base on the window size (windowSizeInSecond)
        """
        # normalise the original lightcurves 
        self.setFluxNormalised()
        # extending the data at beginning and at the end in oder to have equal statistics results
        windSizeInBin = (self.windowSizeInBin-1)/2
        extendFlux = np.r_[self.normFlux[windSizeInBin:0:-1], self.normFlux, self.normFlux[-2:-windSizeInBin-2:-1]]
        dynamicVariance = np.array([])
        for i in range(self.normFlux.size):
            j = i + windSizeInBin    # indices to the exact array
            binStdDev = np.std(np.r_[extendFlux[(j - windSizeInBin):j], extendFlux[(j + 1):(j + windSizeInBin + 1)]])
            dynamicVariance = np.append(dynamicVariance, binStdDev)
        return dynamicVariance
        
    def detOccuProfile(self, windowOccultationSelector):
        """
        searching dataset according to the TNO occultation profile
        There are 5 windowOccultationSelector: 3, 4, 5, 6, 7 bins
        
        windows definition:
         |-----------|--|--|-----------|
         A           B  i  C           D
        -q          -p     p           q
        however, if value of window_occultation is even
        -q          -p     r           q
        """
        windowOccultationDic = {3:1, 4:[2,1], 5:2, 6:[3,2], 7:3}    # profile setup
        
        # normalise the original lightcurves 
        self.setFluxNormalised()
        # extending the data at beginning and at the end in oder to have equal statistics results
        windsInBin = (self.windowSizeInBin-1)/2    # for each side of the window environment
        extendFlux = np.r_[self.normFlux[windsInBin:0:-1], self.normFlux, self.normFlux[-2:-windsInBin-2:-1]]
        # extending the quality for data quality check
        extendQual = np.r_[self.quality[windsInBin:0:-1], self.quality, self.quality[-2:-windsInBin-2:-1]]
        
        if (windowOccultationSelector % 2) == 0:
            profOnTheLeft = windowOccultationDic[windowOccultationSelector][0]
            profOnTheRight = windowOccultationDic[windowOccultationSelector][1]
        else:
            profOnTheLeft = profOnTheRight = windowOccultationDic[windowOccultationSelector]
        
        # to save the method result
        profVariance = np.array([])
        arrThreshold = np.array([])
        eventDirection = np.array([])
        
        for i in range(self.flux.size):
            j = i + windsInBin    # indices to the exact array

            # check quality of the window Occultation Profile, if there is any bad data with the window, than no further detection will be proceeded.
            windowOccuProfileQual = extendQual[j-profOnTheLeft:j+profOnTheRight+1]
            if np.sum(windowOccuProfileQual == 0) > 0:
                tempVariance = 0.0
                tempThreshold = 1.0
                tempEventDirection = 0
            else:
                windowEnvironmentQual = np.r_[extendQual[j-windsInBin:j-profOnTheLeft], extendQual[j+profOnTheRight+1:j+windsInBin+1]]
                goodFraction = np.sum(windowEnvironmentQual)/windowEnvironmentQual.size
                if goodFraction > 0.99:
                    windowEnvironmentFlux = np.r_[extendFlux[j-windsInBin:j-profOnTheLeft], extendFlux[j+profOnTheRight+1:j+windsInBin+1]]
                    nonzerowindowEnvironmentFlux = windowEnvironmentFlux[np.nonzero(windowEnvironmentFlux)]
                    mWindowEnvironmentFLux = np.median(nonzerowindowEnvironmentFlux)
                    #stdWindowEnvoronmentFlux = np.std(nonzerowindowEnvironmentFlux)
                    stdWindowEnvoronmentFlux = np.std(windowEnvironmentFlux)    # need to keep the original variance!
                    # detection method: VARIANCE
                    windowOccuProfileFlux = extendFlux[j-profOnTheLeft:j+profOnTheRight+1]
                    tempVariance = np.sum((windowOccuProfileFlux - mWindowEnvironmentFLux)**2)/windowOccuProfileFlux.size
                    # detection threshold
                    tempThreshold = (self.thresholdLevel * stdWindowEnvoronmentFlux)**2
                    # check direction
                    if (extendFlux[j] - mWindowEnvironmentFLux) < 0:
                        tempEventDirection = -1
                    elif ((extendFlux[j] - mWindowEnvironmentFLux) > self.thresholdLevel*stdWindowEnvoronmentFlux):    # especially large positive peak
                        tempEventDirection = 1
                    else:
                        tempEventDirection = 0
                else:
                    tempVariance = 0.0
                    tempThreshold = 1.0
                    tempEventDirection = 0
            
            profVariance = np.append(profVariance, tempVariance)
            arrThreshold = np.append(arrThreshold, tempThreshold)
            eventDirection = np.append(eventDirection, tempEventDirection)
        
        self.windowOccultationSelector = windowOccultationSelector    
        return profVariance, arrThreshold, eventDirection
        
    def removePeakEvent(self, resultVariance, resultThreshold, resultDirection):
        # mask the bins exceed the dynamic threshold
        iEvt = (resultVariance > resultThreshold)
        # mask the data with good quality
        iBonQual = (self.quality == 1)
        
        nonPeakEventList = np.array([])
        if np.sum(iEvt*iBonQual) > 0:
            binAboveThres = np.where((iEvt*iBonQual) > 0)[0]
            toCut = np.where((binAboveThres[1:] - binAboveThres[0:-1]) > self.windowOccultationSelector)[0] + 1
            binSect = np.array_split(binAboveThres, toCut)    
            for value in binSect:
                # check if there is large peak in the winds of "value", BASED ON eventDirection
                if np.sum(resultDirection[value[0]-self.windowOccultationSelector:value[0]+self.windowOccultationSelector] == 1) == 0:
                    nonPeakEventList = np.append(nonPeakEventList, value[len(value)/2])
                else:
                    pass
        else:
            pass
            
        return nonPeakEventList
        
    def removeSingleEvent(self, eventList, resultVariance, resultThreshold):
        """
        check if the detection is a single event caused by cosmic-ray or induced-electron
        """
        nonSingleEventList = np.array([])
        windsInBin = (self.windowSizeInBin-1)/2    # for each side of the window environment
        exclRegion = self.windowOccultationSelector
        for bin in eventList:
            fluxAroundBin = np.r_[self.normFlux[bin-windsInBin:bin-exclRegion], self.normFlux[bin+exclRegion+1:bin+windsInBin+1]]
            mFluxAroundBin = np.mean(np.abs(fluxAroundBin - 1.0))    # actually the mean of absolute deviation within the section 
            dFluxInBin = np.abs(self.normFlux[bin-exclRegion:bin+exclRegion+1] - 1.0)
            # if the number of bin whose deviation larger than the set threshold level equal to 1, it is considered as an electronic event, such as cosmic-ray or induced-electron within the EMCCD 
            if np.sum(dFluxInBin > self.thresholdLevel*mFluxAroundBin) == 1:
                pass
            else:
                nonSingleEventList = np.append(nonSingleEventList, bin)
                    
        return nonSingleEventList
        
    def removeBadQEvent(self, eventList):
        """
        remove events contain bad quality data
        """
        nonBadQEventList = np.array([])
        for bin in eventList:
            checkQual = self.quality[bin-self.windowOccultationSelector:bin+self.windowOccultationSelector+1]
            if np.sum(checkQual == 0) == 0:
                nonBadQEventList = np.append(nonBadQEventList, bin)
            else:
                pass
                    
        return nonBadQEventList
        
    def removeSeeingEvent(self, eventList, nearbyFibreFluxDic):
        """
        remove events apparently caused by seeing
        """
        nonSeeingEventList = np.array([])
        windSizeInBin = self.windowSizeInBin/2
        nFibreList = nearbyFibreFluxDic.keys()
        for bin in eventList:
            leftBound = bin - windSizeInBin
            rightBound = bin + windSizeInBin+1
            if leftBound < 0:
                leftBound = 0
            if rightBound > self.normFlux.size - 1:
                rightBound = self.normFlux.size - 1
            #leftBound = bin - 20
            #rightBound = bin + 20 + 1

            mainFlux = self.normFlux[leftBound:rightBound]
            mMFlux = np.mean(mainFlux)
            sMFlux = np.std(mainFlux)
            corrAlarm = 0
            tempCorr = {}        
            for nFibre, nFlux in nearbyFibreFluxDic.iteritems():
                compFlux = nFlux[leftBound:rightBound]
                mCFlux = np.mean(compFlux)
                sCFlux = np.std(compFlux)
                # cross correlation at lag zero
                COV2Flux = np.mean((mainFlux - mMFlux)*(compFlux - mCFlux))
                COR2Flux = COV2Flux/(sMFlux*sCFlux)
                tempCorr[nFibre] = COR2Flux
                if abs(COR2Flux) > 0.25:
                    corrAlarm = corrAlarm + 1
                else:
                    pass
                    
            if corrAlarm == 0:
                nonSeeingEventList = np.append(nonSeeingEventList, bin)
        
        return nonSeeingEventList, tempCorr

class mioDB:

    def __init__(self, obsdate, obsid, series, fibre, events):
        self.obsDate = obsdate
        self.obsId = obsid
        self.series = series
        self.fibre = fibre
        self.events = events
        
    def addEvt(self, bin, eSigma, eType):
        self.events[bin] = [eSigma, eType]
        
    def changeEvtSigma(self, bin, eSigma):
        self.events[bin][0] = eSigma
            
    def changeEvtType(self, bin, eType):
        self.events[bin][1] = eType
    
    def modifyEvtDic(self, bin, eSigma): 
        self.events[bin] = [eSigma, 'z']
        
    def __str__(self):
        return '{}, {}, {}, {} => {}'.format(self.obsDate, self.obsId, self.series, self.fibre, self.events.keys())


#def checkBin(lcPath, lcFile, fibre, bin, windowSize, beta, windowOccuProfSelector, detThresholdLevel):
#    """
#    for test only
#    """
#    windowOccultationDic = {3:1, 4:[2,1], 5:2, 6:[3,2], 7:3}    # profile setup
#    
#    TT = an.ReadLcsFits(lcPath)
#    TT._get_filename(lcFile)
#    time, initFlux, initQuality = TT.getLcs(fibre)
#    TT.close()
    
#    DO = DetectOccultation()
#    DO.setKaiserWindowBeta(beta)
#    DO.setWindowSizeInSecond(windowSize)
#    windSizeInBin = (DO.windowSizeInBin - 1)/2
#    DO.setThresholdLevel(detThresholdLevel)
#    DO.setTimeSeriesData(time, initFlux, initQuality)
#    redQuality = DO.quality
#    DO.setFluxNormalised()
    
#    if (windowOccuProfSelector % 2) == 0:
#        profOnTheLeft = windowOccultationDic[windowOccuProfSelector][0]
#        profOnTheRight = windowOccultationDic[windowOccuProfSelector][1]
#    else:
#        profOnTheLeft = profOnTheRight = windowOccultationDic[windowOccuProfSelector]
            
#    windowOccuProfileTime = time[bin-profOnTheLeft:bin+profOnTheRight+1]
#    windowOccuProfileFlux = DO.normFlux[bin-profOnTheLeft:bin+profOnTheRight+1]
#    windowEnvironmentTime = np.r_[time[bin-windSizeInBin:bin-profOnTheLeft], time[bin+profOnTheRight+1:bin+windSizeInBin+1]]
#    windowEnvironmentFlux = np.r_[DO.normFlux[bin-windSizeInBin:bin-profOnTheLeft], DO.normFlux[bin+profOnTheRight+1:bin+windSizeInBin+1]]
    
#    mWindowEnvironmentFlux = np.mean(windowEnvironmentFlux)
#    mdWindowEnvironmentFlux = np.median(windowEnvironmentFlux)
#    stdWindowEnvironmentFlux = np.std(windowEnvironmentFlux)
#    tempVariance = np.sum((windowOccuProfileFlux - mWindowEnvironmentFlux)**2)/windowOccuProfileFlux.size
#    tempThreshold = (detThresholdLevel * stdWindowEnvironmentFlux)**2
    # new Threshold definition
    #thArray = np.ones(windowOccuProfileFlux.size)*mWindowEnvironmentFlux + (detThresholdLevel*stdWindowEnvironmentFlux)
    #print(np.sum((thArray - 0)**2)/windowOccuProfileFlux.size)
    #newThreshold = np.sum((thArray - 1)**2)/windowOccuProfileFlux.size
    
#    secTime = time[bin-windSizeInBin:bin+windSizeInBin+1]
#    secNormFlux = DO.normFlux[bin-windSizeInBin:bin+windSizeInBin+1]
    # normalise the original lightcurves 
#    DO.setFluxNormalised()
    
#    plt.figure('test')
#    plt.clf()
#    plt.plot(windowOccuProfileTime, windowOccuProfileFlux, 'ro')
#    plt.plot(windowEnvironmentTime, windowEnvironmentFlux, 'bo')
#    plt.plot(secTime, secNormFlux, 'k-', drawstyle='steps-mid')
#    plt.axhline(y=mWindowEnvironmentFlux, color='g', ls='--')
#    plt.axhline(y=mdWindowEnvironmentFlux, color='y', ls='--')
#    plt.text(time[bin], 1.3, 'StdD WE: {:.4f}\nVARI WO: {:.4f}\nTHRESHOLD: {:.4f}'.format(stdWindowEnvironmentFlux, tempVariance, tempThreshold))
#    plt.xlim(secTime[0], secTime[-1])
#    plt.ylim(0, 1.5)
#    plt.show()

def occultation(diskPath, obsDate, obsId, lcAffix, windowSize, beta, windowOccuProfSelector, detThresholdLevel, fibreDistance, dryRun=True, output=''):

    if dryRun == False:
        DB = shelve.open('{}/{}'.format(diskPath, output))
    else:
        pass
    
    # setup detection parameters
    DO = DetectOccultation()
    DO.setKaiserWindowBeta(beta)
    DO.setWindowSizeInSecond(windowSize)
    DO.setThresholdLevel(detThresholdLevel)
    
    # setup lightcurve data
    LCPATH = '{}/{}/reduced/{}_{}_lcs'.format(diskPath, obsDate, obsId, lcAffix)
    TT = an.ReadLcsFits(LCPATH)
    
    # make sure the list contains only FITS file ---
    fitsFileMatch = []
    for fitsFile in os.listdir(LCPATH):
        if fnmatch.fnmatch(fitsFile, '*.fits'):
            fitsFileMatch.append(fitsFile)
    #-----------------------------------------------
    numOfFile = str(len(fitsFileMatch))
    for iFits, LCFILE in enumerate(fitsFileMatch):
        TT._get_filename(LCFILE)
        if iFits == 0:
            TT.setFibreExtensionList()
            fibreList = TT.fibreExtDic.keys()
        else:
            pass
                   
        for FIBRE in fibreList:
            time, initFlux, initQuality = TT.getLcs(FIBRE)
            if time.size < DO.windowSizeInBin:    # if the data is smaller than the length the window function, than the analysis is skipped. 
                break
            DO.setTimeSeriesData(time, initFlux, initQuality)
            resultVariance, resultThreshold, resultDirection = DO.detOccuProfile(windowOccuProfSelector)
            tempEventList = DO.removePeakEvent(resultVariance, resultThreshold, resultDirection)
    
            if tempEventList.size != 0:
                tempEventList = DO.removeSingleEvent(tempEventList, resultVariance, resultThreshold)
            else:
                pass
    
            if tempEventList.size != 0:
                tempEventList = DO.removeBadQEvent(tempEventList)
            else:
                pass
                
            if tempEventList.size !=0:
                # retreive nearby fibres
                nearbyFibres = TT._getDistance(fibreDistance)[int(FIBRE)]
                DE = DetectOccultation()
                DE.setKaiserWindowBeta(beta)
                DE.setWindowSizeInSecond(windowSize)
                nFibreDictionary = {}
                for nFibre in nearbyFibres:
                    nTime, nInitFlux, nInitQuality = TT.getLcs(nFibre)
                    DE.setTimeSeriesData(nTime, nInitFlux, nInitQuality)
                    if np.sum(nInitQuality) > 0.99*nInitQuality.size:
                        DE.setFluxNormalised()
                        nFibreDictionary[nFibre] = DE.normFlux
                    else:
                        pass        
                tempEventList, tempCorr = DO.removeSeeingEvent(tempEventList, nFibreDictionary)
            else:
                pass
        
            if tempEventList.size != 0:
                # create an event list sub-dic
                eventDic = {}
                for bin in tempEventList:
                    eventDic[bin] = [detThresholdLevel, 'z']
                print('{}..F{}: {} event(s) at {} at {} sigma.'.format(LCFILE[:-5], FIBRE.zfill(2), tempEventList.size, time[list(tempEventList)], detThresholdLevel))
                
                if dryRun == False:
                    keyheader = 'F{}_{}_{}_F{}'.format(obsDate, obsId, str(iFits+1).zfill(len(numOfFile)), FIBRE.zfill(2))
                    tempDB = mioDB(obsDate, obsId, str(iFits+1).zfill(len(numOfFile)), FIBRE, eventDic)
                    DB[keyheader] = tempDB
                else:
                    pass

            else:
                print('{}..F{}: no event at {} sigma.'.format(LCFILE[:-5], FIBRE.zfill(2), detThresholdLevel))
                
    if dryRun == False:
        DB.close()
    else:
        pass                        

obslist = {'20100228':['TNO01_01', 'XRB01_01'],
            '20100301':['TNO01_01', 'XRB01_01'],
            '20100521':['CVO01_01'],
            '20100523':['CVO01_01', 'WDO01_01'],
            '20100524':['CVO01_01', 'WDO02_01'],
            '20100525':['TNO01_01'],
            '20101020':['TNO02_01', 'WDO01_01'],
            '20101021':['TNO02_01', 'WDO01_01'],
            '20101022':['WDO02_01'],
            '20101023':['CVO01_01'],
            '20101024':['TNO03_01'],
            '20101202':['CVO02_01', 'TNO01_01'],
            '20101203':['CVO02_01', 'TNO01_01'],
            '20101204':['CVO02_01', 'TNO02_01'],
            '20110321':['TNO01_01', 'TNO02_01'],
            '20110322':['TNO02_01', 'TNO04_01'],
            '20110323':['TNO02_01', 'TNO05_01'],
            '20110324':['TNO02_01', 'TNO05_01'],
            '20111214':['TNO05_01'],
            '20111215':['TNO01_01'],
            '20111216':['TNO05_01', 'TNO08_01'],
            '20111217':['TNO05_01', 'TNO08_01'],
            '20111218':['TNO05_01', 'TNO08_01'],
            '20111219':['TNO05_01', 'TNO08_01'],
            '20120112':['TNO01_01', 'TNO03_01'],
            '20120113':['TNO01_01', 'EXO02_01'],
            '20120114':['TNO01_01', 'TNO03_01'],
            '20120116':['EXO01_01', 'TNO03_01'],
            '20120315':['TNO02_01', 'TNO03_01'],
            '20120316':['TNO02_01', 'TNO03_01'],
            '20120317':['TNO03_01', 'TNO04_01'],
            '20120318':['TNO03_01', 'TNO04_01'],
            '20120319':['TNO03_01', 'TNO04_01'],
            '20120614':['WDO01_01'],
            '20120615':['WDO01_01'],
            '20120616':['WDO01_01'],
            '20120617':['WDO01_01'],
            '20120618':['WDO01_01']}
            
junelist = {'20120614':['WDO01_01'],
            '20120615':['WDO01_01'],
            '20120616':['WDO01_01'],
            '20120617':['WDO01_01'],
            '20120618':['WDO01_01']}
            
#lcAffix = 'gf06'
#wKBeta = 32
                
def mioOccultation(disk, obslist, lcAffix, windowSize, windowOccuProfSelector, detThresholdLevel, dryRun=True, output='shelve-default'):
    """
    batch-run occultation detection
    """ 
    for obsDate, value in obslist.iteritems():
        for obsId in value:
            print('Processing: {}//{}'.format(obsDate, obsId))
            occultation(disk, obsDate, obsId, lcAffix, windowSize, 32, windowOccuProfSelector, detThresholdLevel, 15.0, dryRun, output)

class Inquery:

    def __init__(self, disk):
        self.disk = disk
    
    def searchObsDate(self, rootShelveDB, subShelveDB, obsDate):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].obsDate == obsDate:
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchObsId(self, rootShelveDB, subShelveDB, obsId):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].obsId == obsId:
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchSeries(self, rootShelveDB, subShelveDB, series):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].series == series:
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchFibre(self, rootShelveDB, subShelveDB, fibre):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].fibre == fibre:
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
                
    def searchSigma(self, rootShelveDB, subShelveDB, sigmaLowLimit):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][0] == sigmaLowLimit:
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchCandidate(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 'c':
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchSeeing(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 's':
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchBad(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 'b':
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchElectrical(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 'e':
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchUnknown(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 'u':
                    subDB[evtKey] = rootDB[evtKey]
        subDB.close()
        rootDB.close()
        
    def searchNDefine(self, rootShelveDB, subShelveDB):
        rootDB = shelve.open('{}/{}'.format(self.disk, rootShelveDB))
        subDB = shelve.open('{}/{}'.format(self.disk, subShelveDB))
        for evtKey in rootDB:
            for bin in rootDB[evtKey].events.keys():
                if rootDB[evtKey].events[bin][1] == 'z':
                    subDB[evtKey] = rootDB[evtKey]

    def listDB(self, shelveDB):
        db = shelve.open('{}/{}'.format(self.disk, shelveDB))
        for evtKey in db:
            print(db[evtKey])
    
    def combineDB(self, DBList, outputDB):
        dbOut = shelve.open('{}/{}'.format(self.disk, outputDB))
        for subDB in DBList:
            db = shelve.open('{}/{}'.format(self.disk, subDB))
            for evtKey in db:
                dbOut[evtKey] = db[evtKey]
            db.close()
        dbOut.close()
            
    def reviewDB(self, shelveDB, lcAffix, windowSize, windowOccuProfSelector, detThresholdLevel):
        db = shelve.open('{}/{}'.format(self.disk, shelveDB))
        nOEvts = len(db)
        #index = 0
        #while (index >= 0) and (index <= (nOEvts - 1)):
        print(nOEvts)
        
class mioDBPlot:

    def __init__(self, fileName, fibre, time, detThresholdLevel, eventList):
        self.fileName = fileName
        self.fibre = fibre
        self.time = time
        self.dTL = detThresholdLevel
        self.eventList = eventList
        
    def plotGeneral(self, initFlux, normFlux, xpos, ypos, quality, direction, variance, threshold):
        plt.figure('General')
        plt.clf()
        
        plt.subplot(5,1,1)
        plt.plot(self.time, initFlux, 'k-', drawstyle='steps-mid')
        for event in self.eventList:
            plt.axvline(x=self.time[event], color='red', ls='--')
        plt.xlim(self.time[0], self.time[-1])
        plt.ylabel('Original LCs')
        plt.xticks(visible=False)
        plt.title('{}, f{}'.format(self.fileName, self.fibre))
        
        plt.subplot(5,1,2, sharex=plt.subplot(5,1,1))
        plt.plot(self.time, normFlux, 'g+')
        plt.plot(self.time, normFlux, 'k-', drawstyle='steps-mid')
        for event in self.eventList:
            plt.axvline(x=self.time[event], color='red', ls='--')
        plt.xlim(self.time[0], self.time[-1])
        plt.ylim(0.0, 1.5)
        plt.ylabel('Normalised LCs')
        plt.xticks(visible=False)
    
        plt.subplot(5,1,3, sharex=plt.subplot(5,1,1))
        plt.plot(self.time, quality, 'k-', drawstyle='steps-mid', label='quality')
        plt.plot(self.time, direction, 'r-', drawstyle='steps-mid', label='direction')
        plt.xlim(self.time[0], self.time[-1])
        plt.ylim(-2, 2)
        plt.xticks(visible=False)
        plt.legend()
        
        plt.subplot(5,1,4, sharex=plt.subplot(5,1,1))
        plt.plot(self.time, xpos, 'b-', drawstyle='steps-mid', label='x')
        plt.plot(self.time, ypos, 'r-', drawstyle='steps-mid', label='y')
        plt.xlim(self.time[0], self.time[-1])
        plt.ylim(1,40)
        plt.xticks(visible=False)
        plt.legend()
    
        plt.subplot(5,1,5, sharex=plt.subplot(5,1,1))
        plt.plot(self.time, variance, 'k-', drawstyle='steps-mid')
        plt.plot(self.time, threshold, 'r-', label=r'{}$\sigma$'.format(self.dTL))
        plt.xlim(self.time[0], self.time[-1])
        plt.ylabel('Variance')
        plt.xlabel('Time (s)')
        plt.legend()

        plt.show()
    
    def plotEventImages(self, lcPath):
        imgPath = '{}_images'.format(lcPath[:-9])
        IM = an.ReadImgFITS(imgPath)
        IM._getFileName(self.fileName)
        imgcube = IM._getImage(self.fibre)
        IM.close()
        for i, bin in enumerate(self.eventList):
            plt.figure('Cube Image around {}s'.format(self.time[bin]))
            plt.clf()
            binArray = np.arange(bin-2, bin+3, 1)
            for pos, value in enumerate(binArray):
                img = imgcube[value]
                plt.subplot(1, 5, pos+1)
                plt.imshow(img, vmin=100, vmax=5000)
                plt.text(5, 5, '{}s'.format(self.time[value]), fontsize=10, color='white')
                plt.xticks(visible=False)
                plt.yticks(visible=False)
            plt.show()
                        
    def plotNearbyFibresFull(self, normFlux, nFibreDictionary):
        plt.figure('Neighbour Fibres (Full)')
        plt.clf()
        plt.plot(self.time, normFlux, 'k-', drawstyle='steps-mid')
        base = 1.0
        plt.text(self.time[-1] + 10.0, base, 'F{:02}'.format(self.fibre), horizontalalignment='center', verticalalignment='center')
        plt.xlim(plt.xlim(self.time[0] - 5.0, self.time[-1] + 20.0))
        upshift = 1.0
        for nfib, nflux in nFibreDictionary.iteritems():
            plt.plot(self.time, nflux+upshift, drawstyle='steps-mid')
            plt.text(self.time[-1] + 10.0, base+upshift, 'F{:02}'.format(nfib), horizontalalignment='center', verticalalignment='center')
            upshift = upshift + 1.0
            plt.ylim(0.0, upshift+1.5)
        plt.show() 
            
    def plotNearbyFibresOnEvents(self, normFlux, windSizeInBin, nFibreDictionary, CorrelationCoeff):
        plt.figure('Neighbour Fibres')
        plt.clf()
        
        for i, bin in enumerate(self.eventList):
            plt.subplot(1, self.eventList.size, i+1)
            leftBound = bin-windSizeInBin
            rightBound = bin+windSizeInBin+1
            if leftBound < 0:
                leftBound = 0
            if rightBound > self.time.size - 1:
                rightBound = self.time.size - 1
            plt.plot(self.time[leftBound:rightBound], normFlux[leftBound:rightBound], 'k-', drawstyle='steps-mid')
            base = 1.0
            plt.text(self.time[rightBound] + 1.0, base, 'F{:02}'.format(self.fibre), horizontalalignment='center', verticalalignment='center')
            plt.xlim(self.time[leftBound] - 2.0, self.time[rightBound] + 2.0)
            
            plt.ylabel('Normalised fluxes')
            plt.title('{}, f{}'.format(self.fileName, self.fibre))
            upshift = 1.0
            
            for nfib, nflux in nFibreDictionary.iteritems():
                plt.plot(self.time[leftBound:rightBound], nflux[leftBound:rightBound]+upshift, drawstyle='steps-mid')
                plt.text(self.time[rightBound] + 1.0, base+upshift, 'F{:02}'.format(nfib), horizontalalignment='center', verticalalignment='center')
                plt.text(self.time[leftBound] - 1.0, base+upshift, '{:.2f}'.format(np.abs(CorrelationCoeff[nfib])), horizontalalignment='center', verticalalignment='center')
                upshift = upshift + 1.0
                plt.ylim(0.0, upshift+1.5)
            plt.xlim(self.time[leftBound] - 2.0, self.time[rightBound] + 2.0)
        plt.show()
        
    
def checkOccultation(lcPath, lcFile, fibre, windowSize, beta, windowOccuProfSelector, detThresholdLevel, fibreDistance, display=False):
    TT = an.ReadLcsFits(lcPath)
    TT.setFileName(lcFile)
    time, initFlux, xpos, ypos, aptp, skyv, initQuality = TT.getFullData(str(fibre))
    
    DO = DetectOccultation()
    DO.setKaiserWindowBeta(beta)
    DO.setWindowSizeInSecond(windowSize)
    # if the data is smaller than the length the window function, than the analysis is skipped.
    if time.size < DO.windowSizeInBin:
        return []
    else:        
        windSizeInBin = (DO.windowSizeInBin - 1)/2
        DO.setThresholdLevel(detThresholdLevel)
        DO.setTimeSeriesData(time, initFlux, initQuality)
        redQuality = DO.quality

        resultVariance, resultThreshold, resultDirection = DO.detOccuProfile(windowOccuProfSelector)
        normFlux = DO.normFlux
    
        tempEventList = DO.removePeakEvent(resultVariance, resultThreshold, resultDirection)
        print('After first selection: {} result(s) remain: {}'.format(tempEventList.size, tempEventList))
    
        if tempEventList.size != 0:
            tempEventList = DO.removeSingleEvent(tempEventList, resultVariance, resultThreshold)
            print('After second selection: {} result(s) remain: {}'.format(tempEventList.size, tempEventList))
        else:
            pass
    
        if tempEventList.size != 0:
            tempEventList = DO.removeBadQEvent(tempEventList)
            print('After third selection: {} result(s) remain: {}'.format(tempEventList.size, tempEventList))
        else:
            pass
    
        if tempEventList.size != 0:
            # retreive nearby fibres
            nearbyFibres = TT._getDistance(fibreDistance)[fibre]
            DE = DetectOccultation()
            DE.setKaiserWindowBeta(beta)
            DE.setWindowSizeInSecond(windowSize)
            nFibreDictionary = {}
            for nFibre in nearbyFibres:
                nTime, nInitFlux, nInitQuality = TT.getLcs(nFibre)
                DE.setTimeSeriesData(nTime, nInitFlux, nInitQuality)
                if np.sum(nInitQuality) > 0.99*nInitQuality.size:
                    DE.setFluxNormalised()
                    nFibreDictionary[nFibre] = DE.normFlux
                else:
                    pass            
            tempEventList, tempCorr = DO.removeSeeingEvent(tempEventList, nFibreDictionary)
            print('After fourth selection: {} result(s) remain: {}'.format(tempEventList.size, tempEventList))
        else:
            pass
                
        finalEventList = tempEventList
    
        if display == True:
            MP = mioDBPlot(lcFile, fibre, time, detThresholdLevel, finalEventList)
            MP.plotGeneral(initFlux, normFlux, xpos, ypos, redQuality, resultDirection, resultVariance, resultThreshold)  
                
        # ---
        if finalEventList.size == 0:
            listEventList = []
            print('{}..F{:02}: no event at {} sigma.'.format(lcFile[:-5], fibre, detThresholdLevel))
        else:
            listEventList = list(finalEventList)
            #nFibreDictionary[fibre] = normFlux
            #print(nFibreDictionary)
            print('{}..F{:02}: {} event(s) at {} at {} sigma.'.format(lcFile[:-5], fibre, finalEventList.size, time[listEventList], detThresholdLevel))
            if display == True:
                MP.plotNearbyFibresOnEvents(normFlux, windSizeInBin, nFibreDictionary, tempCorr)
            
                plotimg = raw_input('plot images? (y/n) ')
                if plotimg == 'y':
                    MP.plotEventImages(lcPath)
            
                if plotimg == 'n':
                    pass
        return listEventList
    TT.close()