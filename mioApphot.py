#!/usr/bin/env python
# Aperture Photometry for MIOSOTYS FITS image data
# new version: 03 May 2012
# This is an all-in-one program, contains all necessary photometry module in single package.

import os
import os.path as op
import fnmatch as fm
import analysis as an
import numpy as np
from mioAnalysis import ReadImgFITS
from scipy.optimize import leastsq
import matplotlib.pylab as plt
import pyfits as pf

disk2 = '/Volumes/MIOSOTYS2'
obslist2 = {'20100228':['TNO01_01', 'XRB01_01'],
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
            '20111217':['TNO05_01', 'TNO08_01']
            }
            
disk3 = '/Volumes/MIOSOTYS3'
obslist3 = {'20111218':['TNO05_01', 'TNO08_01'],
            '20111219':['TNO05_01', 'TNO08_01'],
            '20120112':['TNO01_01', 'TNO03_01'],
            '20120113':['TNO01_01', 'EXO02_01'],
            '20120114':['TNO01_01', 'TNO03_01'],
            '20120116':['EXO01_01', 'TNO03_01'],
            '20120315':['TNO02_01', 'TNO03_01'],
            '20120316':['TNO02_01', 'TNO03_01'],
            '20120317':['TNO03_01', 'TNO04_01'],
            '20120318':['TNO03_01', 'TNO04_01'],
            '20120319':['TNO03_01', 'TNO04_01']
            }
            
testlist = {'20100521':['CVO01_01']}

class Apphot(object):

    def __init__(self, imageCube):
        """
        Please notice that this class only accepts imageCube from single fibre.
        Users need to extract the imageCube by extracting the FITS image using the ReadImgFITS class outside this class.
        """
        self.imageCube = imageCube 
        self.zsize, self.ysize, self.xsize = imageCube.shape
        
    def _getAvgImage(self):
        avgImageCube = np.average(self.imageCube, axis=0)
        self.avgImageCube = avgImageCube
    
    def _getInitPosition(self):
        """
        obtain initial x-y positions of the target from the averaged image cube.
        """
        self._getAvgImage()
        maxValue = np.amax(self.avgImageCube)
        y, x = np.where(self.avgImageCube == maxValue)
        self.height = maxValue
        self.yPosition, self.xPosition = y[0], x[0]
      
    def _gResiduals(self, param, y, x):
        height, mu, sigma = param
        err = y - (height/(sigma*np.sqrt(2*np.pi))) * np.exp((-(x-mu)**2)/(2*sigma**2))
        return err
    
    def _gPeval(self, x, p):
        return (p[0]/(p[2]*np.sqrt(2*np.pi))) * np.exp((-(x-p[1])**2)/(2*p[2]**2))
        
    def oneDGaussianFit(self, initParam, y, x):
        """
        1-D Gaussian fitting to obtain the center
        """
        plsq = leastsq(self._gResiduals, initParam, args=(y, x))
        self.gHeight, self.gMu, self.gSigma = plsq[0]
        
    def _getRadiusMap(self, frame, xPosition, yPosition):
        yIndice, xIndice = np.indices(self.imageCube[frame].shape)
        xOff = xIndice - xPosition
        yOff = yIndice - yPosition
        self.radiusMap = np.sqrt(xOff**2 + yOff**2)
        
    def apValue(self, frame, xPosition, yPosition, apRadius):
        # creating aperture masks
        apMask = self.radiusMap <= apRadius
        # obtainning photons within the mask
        iApMask = np.where(apMask == 1)
        apPhoton = self.imageCube[frame][iApMask]

        return apPhoton
        
    def ringSkyValue(self, frame, xPosition, yPosition, innRadius, outRadius):
        # creating sky masks
        skyMask = np.logical_and(self.radiusMap > innRadius, self.radiusMap <= outRadius)
        # obtainning photons within the mask
        iSkyMask = np.where(skyMask == 1)
        skyPhoton = self.imageCube[frame][iSkyMask]
        # remove extreme values
        MedSky = np.median(skyPhoton)
        PosSky = np.sqrt(MedSky)    # !backgroung is in Possion distribution!
        iGoodMask = np.logical_and(skyPhoton > (MedSky - 3.0*PosSky), skyPhoton < (MedSky + 3.0*PosSky))
        goodSkyPhoton = skyPhoton[iGoodMask]
        
        return goodSkyPhoton
        
    def getAptPhot(self, frame, xPosition, yPosition, apRadius, innRadius, outRadius):
        aptFlux = self.apValue(frame, xPosition, yPosition, apRadius)
        skyValue = self.ringSkyValue(frame, xPosition, yPosition, innRadius, outRadius)
        totAptFlux = np.sum(aptFlux)
        if skyValue.size == 0:
            avgSkyValue = -666
            redAptFlux = -666
        else:
            avgSkyValue = np.sum(skyValue)/skyValue.size
            redAptFlux = totAptFlux - (avgSkyValue * aptFlux.size)
        
        return redAptFlux, aptFlux.size, avgSkyValue

# ---

def testGfit(imgPath, imgFile, fibre, frame=0):
    """
    """
    IM = ReadImgFITS(imgPath)
    IM._getFileName(imgFile)
    imgCube = IM._getImage(fibre)
    IM.close()
    
    AP = Apphot(imgCube)
    AP._getInitPosition()
    iyPos, ixPos = AP.yPosition, AP.xPosition
    
    roughLevel = np.median(AP.imageCube[frame])
    iHeight = AP.height - roughLevel 
    mCrosX = np.mean(AP.imageCube[frame], axis=0) - roughLevel    # bring the floor to ~ zero
    xAxis = np.arange(0, AP.xsize, 1)
    AP.oneDGaussianFit([iHeight, ixPos, 1.0], mCrosX, xAxis)
    fxPos, xS = AP.gMu, AP.gSigma
    CrosXFit = AP._gPeval(xAxis, [AP.gHeight, AP.gMu, AP.gSigma])
    
    mCrosY = np.mean(AP.imageCube[frame], axis=1) - roughLevel    # bring the floor to ~ zero
    yAxis = np.arange(0, mCrosY.size, 1)
    AP.oneDGaussianFit([iHeight, iyPos, 1.0], mCrosY, yAxis)
    fyPos, yS = AP.gMu, AP.gSigma
    CrosYFit = AP._gPeval(yAxis, [AP.gHeight, AP.gMu, AP.gSigma])    
            
    plt.figure(20)
    plt.clf()
    
    plt.subplot(1,2,1)
    plt.plot(mCrosX, 'k+')
    plt.plot(CrosXFit, 'r-')
    plt.text(fxPos, 80, '[{:.1f}, {:.1f}]'.format(fxPos, xS), horizontalalignment='center')
    plt.xlim(0,40)
    plt.ylim(-20, 100)
    plt.title('From image in \n{}/{} f#{}'.format(imgFile, fibre, frame))
    
    plt.subplot(1,2,2)
    plt.plot(mCrosY, 'k+')
    plt.plot(CrosYFit, 'r-')
    plt.text(fyPos, 80, '[{:.1f}, {:.1f}]'.format(fyPos, yS), horizontalalignment='center')
    plt.xlim(0,40)
    plt.ylim(-20, 100)
    
    plt.show()

# ---

def apPhotometry(imgCube, apSize, innSize, outSize):
    """
    """
    AP = Apphot(imgCube)
    AP._getInitPosition()        
    iyPos, ixPos = AP.yPosition, AP.xPosition
    
    FR = np.array([])
    FL = np.array([])
    XP = np.array([])
    YP = np.array([])
    EA = np.array([])
    MS = np.array([])
    QC = np.array([])
        
    for frame in range(AP.zsize):
    
        if frame == 0:
            iHeight = AP.height - np.median(AP.imageCube[frame])
            initXCond = [iHeight, ixPos, 1.0]
            initYCond = [iHeight, iyPos, 1.0]
        else:
            initXCond = prevXCond
            initYCond = prevYCond

        # calculate X/Y positions
        # performing two 1-D guassian fit for X and Y axes
        roughFloorLevel = np.median(AP.imageCube[frame])    # bring the floor to ~ zero, this is important so the fitting result is right!
        mCrosX = np.mean(AP.imageCube[frame], axis=0) - roughFloorLevel
        xAxis = np.arange(0, AP.xsize, 1)
        AP.oneDGaussianFit(initXCond, mCrosX, xAxis)
        prevXCond = [AP.gHeight, AP.gMu, AP.gSigma]
        fxPos, fxRad = AP.gMu, AP.gSigma
    
        mCrosY = np.mean(AP.imageCube[frame], axis=1) - roughFloorLevel
        yAxis = np.arange(0, AP.ysize, 1)
        AP.oneDGaussianFit(initYCond, mCrosY, yAxis)
        prevYCond = [AP.gHeight, AP.gMu, AP.gSigma]
        fyPos, fyRad = AP.gMu, AP.gSigma
        
        #apSize = np.sqrt(fxRad**2 + fyRad**2)*2.0        
        # photometry
        AP._getRadiusMap(frame, fxPos, fyPos)
        redFlux, effAptSize, mSkyValue = AP.getAptPhot(frame, fxPos, fxPos, apSize, innSize, outSize)
        
        if (frame < 3) and (redFlux == -666):
            QC = np.append(QC, 0)
            redFlux = 0.0
            mSkyValue = 0.0
            print('x {:4d}: {:.1f}, {:.1f}, {:.1f}, {}, {:.1f}'.format(frame, redFlux, fxPos, fyPos, effAptSize, mSkyValue))
            
        elif (frame >= 3) and redFlux == -666:
            QC = np.append(QC, 0)
            redFlux = np.median(FL[-3::])
            mSkyValue = np.median(MS[-3::])
            if (frame % 500) == 0:
                print('x {:4d}: {:.1f}, {:.1f}, {:.1f}, {}, {:.1f}'.format(frame, redFlux, fxPos, fyPos, effAptSize, mSkyValue))
            
        else:
            QC = np.append(QC, 1)
            if (frame % 1200) == 0:
                print('v {:4d}: {:.1f}, {:.1f}, {:.1f}, {}, {:.1f}'.format(frame, redFlux, fxPos, fyPos, effAptSize, mSkyValue))
        
        FR = np.append(FR, frame)
        FL = np.append(FL, redFlux)
        XP = np.append(XP, fxPos)
        YP = np.append(YP, fyPos)
        EA = np.append(EA, effAptSize)
        MS = np.append(MS, mSkyValue)
                
    return FR, FL, XP, YP, EA, MS, QC
       
def readLcs2Fits(imgPath, imgFile, apSize, innSize, outSize, savePath):
    """
    """
    IM = ReadImgFITS(imgPath)
    IM._getFileName(imgFile)
    priHeader = IM.temp[0].header
    snapShot = IM.temp[0].data
    numFibre = priHeader['FIBREACT']
    expTime = priHeader['EXPTIME']
    timeZero = 0.0

    cols_ext = []
    for index in range(numFibre):
        fibre = priHeader['EXTEN_{:02}'.format(index+1)]
        print('{}: fibre {}...'.format(imgFile, fibre))
        imgCube = IM._getImage(fibre)
        
        # excute photometry
        frame, srcFlux, xCentre, yCentre, srcPixl, mSkyValue, quality = apPhotometry(imgCube, apSize, innSize, outSize)
        
        time = frame*expTime
        
        col1 = pf.Column(name='Time (sec)', format='E', array=time)
        col2 = pf.Column(name='SrcFlux (count)', format='E', array=srcFlux)
        col3 = pf.Column(name='XCentre (pixel)', format='E', array=xCentre)
        col4 = pf.Column(name='YCentre (pixel)', format='E', array=yCentre)
        col5 = pf.Column(name='SrcArea (pixel)', format='E', array=srcPixl)
        col6 = pf.Column(name='AvgSky (count)', format='E', array=mSkyValue)
        col7 = pf.Column(name='Quality (bool)', format='E', array=quality)

        cols = pf.ColDefs([col1, col2, col3, col4, col5, col6, col7])
        tbhdu = pf.new_table(cols)
        cols_ext.append(tbhdu)
    
    IM.close()    
    
    hdu = pf.PrimaryHDU(snapShot, priHeader)
    thdulist = pf.HDUList([hdu] + cols_ext)
    
    # Update extension header    
    for jndex in range(numFibre):
        extTable = thdulist[jndex+1]
        extTable.header.update('FIBRE', priHeader['EXTEN_{:02}'.format(jndex+1)])
        extTable.header.update('RA', priHeader['RA_{:02}'.format(jndex+1)])
        extTable.header.update('DEC', priHeader['DEC_{:02}'.format(jndex+1)])
        extTable.header.update('MAG', priHeader['MAG_{:02}'.format(jndex+1)])
        extTable.header.update('EQUINOX', priHeader['EQUINOX'])
        extTable.header.update('APERTURE', apSize, 'aperture radius in pixels')
        extTable.header.update('SKYRADIN', innSize, 'inner sky radius in pixels')
        extTable.header.update('SKYRADOU', outSize, 'outer sky radius in pixels')

    thdulist.writeto('{}/{}'.format(savePath, imgFile), output_verify='fix')    # pyfits 3.0.3 changes the behaviour of verify (default to 'exception'), but is reversed to 'fix' in 3.0.4

def mioApphot(disk, dictionary, lcDirAff, apertureSize, innerRadius, outerRadius):
    """
    front-end of launch photometry process and save lcs to FITS format
    """
    for obsdate, obsid_list in dictionary.iteritems():
        for obsid in obsid_list:
            imgDir = '{}/{}/reduced/{}_images'.format(disk, obsdate, obsid)
            lcDir = '{}/{}/reduced/{}_{}_lcs'.format(disk, obsdate, obsid, lcDirAff)
            
            if op.exists(lcDir) != 1:
                os.mkdir(lcDir)
            else:
                pass
            
            print('{}:{}'.format(obsdate, obsid))    
            
            # make sure the list contains only FITS file ---
            fitsFileMatch = []
            for fitsFile in os.listdir(imgDir):
                if fm.fnmatch(fitsFile, '*.fits'):
                    fitsFileMatch.append(fitsFile)
            #-----------------------------------------------
            
            for fitsName in fitsFileMatch:
                print('{}'.format(fitsName))
                readLcs2Fits(imgDir, fitsName, apertureSize, innerRadius, outerRadius, lcDir)

import sys
                
if __name__ == "__main__":

    usage = "Usage: python mioApphot.py /Volumes/MIOSOTYS2 obslist2 nv06 6.0 9.0 11.0"
    
    if len(sys.argv) != 7:
        print "Improper number of arguments."
        print usage
        sys.exit()
        
    imgSaveDisk = sys.argv[1]
    dataStructure = sys.argv[2]
    outLcDirAffix = sys.argv[3]
    apertureSize = sys.argv[4]
    innerRadius = sys.argv[5]
    outerRadius = sys.argv[6]
    
    mioApphot(imgSaveDisk, dataStructure, outLcDirAffix, apertureSize, innerRadius, outerRadius)