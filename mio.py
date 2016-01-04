#!/usr/bin/env python

# Miosotys data transform, reduction, and analysis classes

#rawImagePath = '{}/{}/raw/'.format(rootDir, obsDate)
#reducedImagePath = '{}/{}/reduced/{}_imgs'.format(rootDir, obsDate, obsID)
    
class Transform:
    def __init__(self, rootDir, obsDate, obsID):
        self.rawImagePath = '{}/{}/raw'.format(rootDir, obsDate)
        
    def setFullFilePath(self, speFileName):
        self.speFile = open('{}/{}'.format(self.rawImagePath, speFileName), 'rb')
        self.getDimension()
    
    def readBits(self, pos, ntype, size):
        self.speFile.seek(pos)
        return np.fromfile(self.speFile, ntype, size)
    
    def getDimension(self):
        self.xDim = int(self.readBits(42, np.uint16, 1)[0])
        self.yDim = int(self.readBits(656, np.uint16, 1)[0])
        self.zDim = int(self.readBits(1446, np.int32, 1)[0])
        
    def getImage(self):
        img = self.readBits(4100, np.uint16, self.xDim * self.yDim * self.zDim)
        return img.reshape((self.zDim, self.yDim, self.xDim))
        
    def getObsDate(self):
        dat = self.readBits(20, 'S10', 1)[0]
        monthDic = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        return ('DATE-OBS', '{}-{}-{}'.format(dat[5:10], monthDic[dat[2:5]], dat[:2]))
        
    def getObsUtc(self):
        utc = self.readBits(179, 'S7', 1)[0]
        return ('UTC-OBS', '{}:{}:{}'.format(utc[:2], utc[2:4], utc[4:6]), 'Universal Time')
        
    def getObsLst(self):
        lst = self.readBits(172, 'S7', 1)[0]
        return ('LST-OBS', '{}:{}:{}'.format(lst[:2], lst[2:4], lst[4:6]), 'Local Time')
        
    def getIso8601Date(self):
        return ('DATE-OBS', '{}T{}'.format(self.getObsDate()[1], self.getObsLst()[1]), 'Beginning date and time of observation (LT).')

class Reduction(object):

class Analysis(object):
