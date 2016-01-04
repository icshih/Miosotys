#!/usr/bin/env python

import os
import os.path
import shutil
import shelve
import numpy as np
from mioOccultation3 import DetectOccultation, checkOccultation, mioDB, Inquery
from mioAnalysis import ReadLcsFits
import matplotlib.pylab as plt

def dbReviewAuto(disk, shelveDB, lcAffix, windowSize, windowOccuProfSelector, detThresholdLevel, subShelvDB=''):
    """review occultation detection result (saved in Python shelve database)
    """

    db = shelve.open('{}/{}'.format(disk, shelveDB))
    
    if len(subShelvDB) != 0:
        subdb = shelve.open('{}/{}'.format(disk, subShelvDB))
        dbKeys = subdb.keys()
        subdb.close()
    else:
        dbKeys = db.keys()

    for eventKey in dbKeys:
        obsDate, obsId = db[eventKey].obsDate, db[eventKey].obsId
        
        lcPath = '{}/{}/reduced/{}_{}_lcs'.format(disk, obsDate, obsId, lcAffix)
        fileName = '{}_2_{}_{}.fits'.format(obsDate, obsId, db[eventKey].series)
        print(fileName)
        if os.path.exists('{}/{}'.format(lcPath, fileName)) == False:
            print('Warning: file {} does not exist!'.format(fileName))
        else:
            fibre = int(db[eventKey].fibre)
            temp = db[eventKey]
            ogEventList = temp.events.keys()
            ogEventList.sort()
            rvEventList = checkOccultation(lcPath, fileName, fibre, windowSize, 32, windowOccuProfSelector, detThresholdLevel, 15.0, display=False)
        
            print('{} => {}'.format(ogEventList, rvEventList))
            if len(rvEventList) != 0:
                for rbin in rvEventList:
                    # if the distance between rbin and the ogEventList is less than assigned windowOccuProfSelector
                    # it is considered as the same event
                    dist2bin = np.array(ogEventList) - rbin
                    i = np.where(np.abs(dist2bin) <= windowOccuProfSelector)[0]
                    if len(i) == 0:
                        temp.addEvt(rbin, detThresholdLevel, 'z')
                    else:
                        temp.changeEvtSigma(ogEventList[i[0]], detThresholdLevel)
                db[eventKey] = temp
            else:
                pass

    db.close()

def dbReviewManu(disk, shelveDB, lcAffix, windowSize, windowOccuProfSelector, detThresholdLevel, subShelvDB=''):
    """review occultation detection result (saved in Python shelve database)
    """

    db = shelve.open('{}/{}'.format(disk, shelveDB))

    if len(subShelvDB) != 0:
        subdb = shelve.open('{}/{}'.format(disk, subShelvDB))
        dbKeys = subdb.keys()
        subdb.close()
    else:
        dbKeys = db.keys()

    count = 0
    while count <= len(dbKeys) - 1:
        eventKey = dbKeys[count]
        obsDate, obsId = db[eventKey].obsDate, db[eventKey].obsId
        lcPath = '{}/{}/reduced/{}_{}_lcs'.format(disk, obsDate, obsId, lcAffix)
        fileName = '{}_2_{}_{}.fits'.format(obsDate, obsId, db[eventKey].series)
        
        print(count + 1, obsDate, obsId, db[eventKey].fibre, db[eventKey].events)
        # check the existance of the file to avoid the interruption of the program
        if os.path.exists('{}/{}'.format(lcPath, fileName)) == False:
            print('Warning: file {} does not exist!'.format(fileName))
        else:
            fibre = int(db[eventKey].fibre)
            temp = db[eventKey]
            ogEventList = temp.events.keys()
            ogEventList.sort()
            rvEventList = checkOccultation(lcPath, fileName, fibre, windowSize, 32, windowOccuProfSelector, detThresholdLevel, 15.0, display=True)
            if len(rvEventList) != 0:
                for rbin in rvEventList:
                    # if the distance between rbin and the ogEventList is less than assigned windowOccuProfSelector
                    # it is considered as the same event
                    dist2bin = np.array(ogEventList) - rbin
                    i = np.where(np.abs(dist2bin) <= windowOccuProfSelector)[0]
                    if len(i) == 0:
                        temp.addEvt(rbin, detThresholdLevel, 'z')
                        eType = raw_input('{} => event Type: (b)ad, (c)andidate, (e)lectric, (s)eeing, (u)nknown? => '.format(rbin))
                        if not eType:
                            pass
                        else:
                            temp.changeEvtType(rbin, eType)
                    else:
                        temp.changeEvtSigma(ogEventList[i[0]], detThresholdLevel)
                        eType = raw_input('{} => event Type: (b)ad, (c)andidate, (e)lectric, (s)eeing, (u)nknown? => '.format(rbin))
                        if not eType:
                            pass
                        else:
                            temp.changeEvtType(ogEventList[i[0]], eType)
                db[eventKey] = temp
            else:
                pass
        count = count + 1
        
        cont = raw_input('Continue? (q) for quit... ')
        if cont == 'q':
            break
        else:
            pass
            
    db.close()

def quickReview(disk, dbFile):
    """Review the lc data from miosotys database
       database format: obsDate, obsId, numSeries, fibre, eventDic
    """
    db = shelve.open('{}/{}'.format(disk, dbFile))
    DO = DetectOccultation()    # just call for cleaning time series data
    DO.setKaiserWindowBeta(32)
    DO.setWindowSizeInSecond(10.0)
    
    keyList = db.keys()
    for i, key in enumerate(keyList):
        print('{}: {}'.format(i, key))    
    lastDBItem = i
    
    initial = raw_input('select event number: ')
    if not initial:
        j = 0
    else:
        j = int(initial)
        
    while (j >= 0) and (j <= i):
        
        evtKey = keyList[j]
        obsDate = db[evtKey].obsDate
        obsId = db[evtKey].obsId
        series = db[evtKey].series
        fibre = db[evtKey].fibre
        events = db[evtKey].events
        lcPath = '{}/{}/reduced/{}_gf06_lcs'.format(disk, obsDate, obsId)
        lcFileName = '{}_2_{}_{}.fits'.format(obsDate, obsId, series)
        
        LC = ReadLcsFits(lcPath)
        LC.setFileName(lcFileName)
        time, initFlux, xPosition, yPosition, effApeSize, medSky, initQuality = LC.getFullData(fibre)
        DO.setTimeSeriesData(time, initFlux, initQuality)
        DO.setFluxNormalised()
        flux = DO.flux
        nFlux = DO.normFlux
        quality = DO.quality
        
        plt.figure('Lightcurves')
        plt.clf()
        
        plt.subplot(4, 1, 1)
        plt.plot(time, nFlux, 'k-', drawstyle='steps-mid')
        for k, m in events.iteritems():
            sigma = m[0]
            eType = m[1]
            plt.axvline(x=time[k], color='red', ls='--')
            plt.text(time[k], 1.3, '{},  {}'.format(eType, sigma))
        plt.xlim(time[k]-10.0, time[k]+10.0)
        plt.ylim(0, 1.5)
        plt.xticks(visible=False)
        plt.ylabel('Normalised\nlcs')
        plt.title('{}: {}'.format(j, evtKey))
        
        plt.subplot(4, 1, 2, sharex=plt.subplot(4, 1, 1))
        plt.plot(time, flux, 'k-', drawstyle='steps-mid')
        for k, m in events.iteritems():
            plt.axvline(x=time[k], color='red', ls='--')
        plt.xlim(time[k]-10.0, time[k]+10.0)
        plt.ylabel('Original\nlcs')
        plt.xticks(visible=False)
        
        plt.subplot(4, 1, 3, sharex=plt.subplot(4, 1, 1))
        plt.plot(time, xPosition, 'r-')
        plt.plot(time, yPosition, 'b-')
        plt.xlim(time[k]-10.0, time[k]+10.0)
        plt.ylim(0, 39)
        plt.ylabel('Centroid')
        plt.xticks(visible=False)
        
        plt.subplot(4, 1, 4, sharex=plt.subplot(4, 1, 1))
        plt.plot(time, quality, 'k-')
        plt.xlim(time[k]-10.0, time[k]+10.0)
        plt.ylim(-0.5, 1.5)
        plt.ylabel('Quality')
        plt.xlabel('Time (s)')
        
        plt.show()
        
        answer = raw_input('(p)revious or (n)ext lightcurves? ')
        if answer == 'p' and j == 0:
            print('It is the first in the list.')
        elif answer == 'p':
            j = j - 1
        elif answer == 'n' and j == lastDBItem:
            print('It is the last in the list.')
        elif answer == 'n':
            j = j + 1
        elif answer == 'q':
            break
        
    db.close()

def selectDB(disk, rootDB, subDB):
    
    EI = Inquery(disk)
    
    selectDic = {'1': 'Obs Date', '2': 'Obs Id', '3': 'Fibre', '4': 'Detection Level', '5': 'Event Type'}
    selectOrder = []
    while True:
        select = raw_input('Searching for\n(1) {}\n(2) {}\n(3) {}\n(4) {}\n(5) {}\nor press Enter to start =>? '.format(selectDic['1'], selectDic['2'], selectDic['3'], selectDic['4'], selectDic['5']))
        if not select:
            break
            
        if select in selectOrder:
            print('{} has been selected!'.format(selectDic[select]))
        elif (select in selectDic.keys()) == False:
            print('Your selection is out of range.')
        else:
            selectOrder.append(select)
    
    if len(selectOrder) != 0:
        srcRoot = '{}'.format(rootDB)
        inimidDB = 'DB_temp_ini'
        middleDB = 'DB_temp_old'
        filtedDB = 'DB_temp_new'
        extrctDB = '{}'.format(subDB)

        for isel in selectOrder:
    
            if isel == '1':
                if os.path.exists(disk + '/' + inimidDB) == False:
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + inimidDB)
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + middleDB)
            
                obsdate = raw_input('Which obsDate? (ex: 20100228) ')
                EI.searchObsDate(middleDB, filtedDB, obsdate)
                #
                os.remove(disk + '/' + middleDB)
                shutil.copy(disk + '/' + filtedDB, disk + '/' + middleDB)
                os.remove(disk + '/' + filtedDB)
            
            if isel == '2':
                if os.path.exists(disk + '/' + inimidDB) == False:
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + inimidDB)
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + middleDB)
                
                obsid = raw_input('Which obsID? (ex: TNO01_01) ')
                EI.searchObsId(middleDB, filtedDB, obsid)
                #
                os.remove(disk + '/' + middleDB)
                shutil.copy(disk + '/' + filtedDB, disk + '/' + middleDB)
                os.remove(disk + '/' + filtedDB)
            
            if isel == '3':
                if os.path.exists(disk + '/' + inimidDB) == False:
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + inimidDB)
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + middleDB)
                
                fibre = raw_input('Which fibre? (ex: 28) ')
                EI.searchFibre(middleDB, filtedDB, fibre)
                #
                os.remove(disk + '/' + middleDB)
                shutil.copy(disk + '/' + filtedDB, disk + '/' + middleDB)
                os.remove(disk + '/' + filtedDB)
    
            if isel == '4':
                if os.path.exists(disk + '/' + inimidDB) == False:
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + inimidDB)
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + middleDB)
                
                sigma = raw_input('Upper threshold level limit(in sigma)? (ex: 3.0) ')
                EI.searchSigma(middleDB, filtedDB, float(sigma))
                #
                os.remove(disk + '/' + middleDB)
                shutil.copy(disk + '/' + filtedDB, disk + '/' + middleDB)
                os.remove(disk + '/' + filtedDB)
            
            if isel == '5':
                if os.path.exists(disk + '/' + inimidDB) == False:
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + inimidDB)
                    shutil.copy(disk + '/' + srcRoot, disk + '/' + middleDB)

                etype = raw_input('Which event type?\n(b)ad, (c)andidate, (e)lectric, (s)eeing, (u)nknown, (n)on-defined ')
                if etype == 'b':
                    EI.searchBad(middleDB, filtedDB)
                if etype == 'c':
                    EI.searchCandidate(middleDB, filtedDB)
                if etype == 'e':
                    EI.searchElectrical(middleDB, filtedDB)
                if etype == 's':
                    EI.searchSeeing(middleDB, filtedDB)
                if etype == 'u':
                    EI.searchUnknown(middleDB, filtedDB)
                if etype == 'n':
                    EI.searchNDefine(middleDB, filtedDB)
                if not etype:
                    pass
                #
                os.remove(disk + '/' + middleDB)
                shutil.copy(disk + '/' + filtedDB, disk + '/' + middleDB)
                os.remove(disk + '/' + filtedDB) 
            
        shutil.copy(disk + '/' + middleDB, disk + '/' + extrctDB)
        os.remove(disk + '/' + middleDB)
        os.remove(disk + '/' + inimidDB)
        
    else:
        pass

def countEvents(disk, DB):
    db = shelve.open('{}/{}'.format(disk, DB))
    count1 = 0
    count2 = 0
    for evtKey in db:
        count2 = count2 + len(db[evtKey].events)
        for event in db[evtKey].events:            
            if db[evtKey].events[event][1] == 'c':
                print(evtKey)
                count1 = count1 + 1
                
    print(len(db), count2, count1)
    db.close()
    
def combineDB(disk, DBList, outputDB):
    EI = Inquery(disk)
    EI.combineDB(DBList, outputDB)