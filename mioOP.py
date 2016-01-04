#!/usr/bin/env python

import os
import os.path as op
import fnmatch as fm
import pprint
import fnmatch
import shutil
import pickle

# Principle Invesitgator list
piList = {'TNO': 'A. Doressoundiram',
          'EXO': 'S. Renner',
          'WDO': 'H.-K. Chang',
          'CVO': 'I-C. Shih',
          'XRB': 'I-C. Shih'}

# Observation List
obslist = {'20100228': ['TNO01_01', 'XRB01_01'],
 '20100301': ['TNO01_01', 'XRB01_01'],
 '20100521': ['CVO01_01'],
 '20100523': ['CVO01_01', 'WDO01_01'],
 '20100524': ['CVO01_01', 'WDO02_01'],
 '20100525': ['TNO01_01'],
 '20101020': ['TNO02_01', 'WDO01_01'],
 '20101021': ['TNO02_01', 'WDO01_01'],
 '20101022': ['WDO02_01'],
 '20101023': ['CVO01_01'],
 '20101024': ['TNO03_01'],
 '20101202': ['CVO02_01', 'TNO01_01'],
 '20101203': ['CVO02_01', 'TNO01_01'],
 '20101204': ['CVO02_01', 'TNO02_01'],
 '20110321': ['TNO01_01', 'TNO02_01'],
 '20110322': ['TNO02_01', 'TNO04_01'],
 '20110323': ['TNO02_01', 'TNO05_01'],
 '20110324': ['TNO02_01', 'TNO05_01'],
 '20111214': ['TNO05_01'],
 '20111215': ['TNO01_01'],
 '20111216': ['TNO05_01', 'TNO08_01'],
 '20111217': ['TNO05_01', 'TNO08_01'],
 '20111218': ['TNO05_01', 'TNO08_01'],
 '20111219': ['TNO05_01', 'TNO08_01'],
 '20120112': ['TNO01_01', 'TNO03_01'],
 '20120113': ['TNO01_01', 'EXO02_01'],
 '20120114': ['TNO01_01', 'TNO03_01'],
 '20120116': ['EXO01_01', 'TNO03_01'],
 '20120315': ['TNO02_01', 'TNO03_01'],
 '20120316': ['TNO02_01', 'TNO03_01'],
 '20120317': ['TNO03_01', 'TNO04_01'],
 '20120318': ['TNO03_01', 'TNO04_01'],
 '20120319': ['TNO03_01', 'TNO04_01'],
 '20120614': ['WDO01_01'],
 '20120615': ['WDO01_01'],
 '20120616': ['WDO01_01'],
 '20120617': ['WDO01_01'],
 '20120618': ['WDO01_01']
 }

newlist = {'20120618': ['WDO01_01']}

def convertWithLens(disk, dictionary, dryRun=True):
    """
    converting SPE image to FITS format.
    Usage: convertWithLens(disk, obslist, dryRun=False)
    Warning: This method is designed for observations with focus lens from December 2011
    """
    from converter2011b import convert_tout
    
    for obsdata, obsid_list in dictionary.iteritems():
        for obsid in obsid_list:
            pi = piList[obsid[:3]]
            print(obsdata, obsid, pi)
            if dryRun == False:
                convert_tout('{}/{}'.format(disk, obsdata), obsid, pi)
                
def createLcs(disk, dictionary, affix, aperture, inner, outer, dryRun=True):
    """
    creating lightcurves
    Usage: createLcs(disk, obslist, affix, aperture, inner, outer, dryRun=False)
    """
    from mioApphot import readLcs2Fits
    
    for obsdate, obsid_list in dictionary.iteritems():
        for obsid in obsid_list:
            imgDir = '{}/{}/reduced/{}_images'.format(disk, obsdate, obsid)
            lcDir = '{}/{}/reduced/{}_{}_lcs'.format(disk, obsdate, obsid, affix)
            
            print('{}:{}'.format(obsdate, obsid))
            
            if dryRun == False:
                if op.exists(lcDir) != 1:
                    os.mkdir(lcDir)
                else:
                    pass
                
                # make sure the list contains only FITS file ---
                fitsFileMatch = []
                for fitsFile in os.listdir(imgDir):
                    if fm.fnmatch(fitsFile, '*.fits'):
                        fitsFileMatch.append(fitsFile)
                #-----------------------------------------------    
                for fitsName in fitsFileMatch:
                    print('{}'.format(fitsName))
                    readLcs2Fits(imgDir, fitsName, aperture, inner, outer, lcDir)   
                    
            else:
                pass
            
def copylcfiles(srcDisk, dstDisk, lcAffix):
    """
    """
    for (dirpath, dirnames, filenames) in os.walk(srcDisk):
        if 'reduced' in dirnames:
            for dirs in os.listdir('{}/reduced'.format(dirpath)):
                if dirs[-8:] == lcAffix:
                    print(dirpath[-8:], dirs)
                
                    distPath = '{}/{}/reduced/{}'.format(dstDisk, dirpath[-8:], dirs)    
                    if os.path.exists(distPath) == False:
                        os.makedirs(distPath)
                    
                    for filename in os.listdir('{}/reduced/{}'.format(dirpath, dirs)):
                        if fnmatch.fnmatch(filename, '*.fits'):
                            src = '{}/reduced/{}/{}'.format(dirpath, dirs, filename)
                            shutil.copy(src, distPath)

def manageObsList(obsListDic, saveDir, update=True):
    """
    The obsvation list database is named miosotys-obslist.
    Usage: manageObsList('/sample/directory', '/sample/save/directory', update=True)
    """
    
    if update == False:
        dbfile = open('{}/miosotys-obslist'.format(saveDir), 'wb')
        pickle.dump(obsListDic, dbfile)
        dbfile.close()
        
    if update == True:
        dbfile = open('{}/miosotys-obslist'.format(saveDir), 'rb')
        db = pickle.load(dbfile)
        dbfile.close()
        
        pprint.pprint(db)
        
        obsIdList = []
        while True:
            od = raw_input('Enter obsDate => ')
            if not od:
                break
            
            if (od in db) == True:
                print('{}\nThis ObsDate has already exist!'.format(db[od]))
                db[od] = []
                while True:
                    oi = raw_input('Update obsId => (ex. TNO01_01) ')
                    if not oi:
                        break
                    else:
                        obsIdList.append(oi)
                db[od] = obsIdList
                
            else:
                while True:
                    oi = raw_input('Enter obsId => (ex. TNO01_01) ')
                    if not oi:
                        break
                    else:
                        obsIdList.append(oi)
                db[od] = obsIdList
        
        dbfile = open('{}/miosotys-obslist'.format(saveDir), 'wb')
        pickle.dump(db, dbfile)        
        dbfile.close()