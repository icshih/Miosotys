#!/usr/bin/env python
#
# version 0.1: 28/07/2010
# version 1.0: 26/09/2011
# functions of creating lightcurves has been moved to apphoto.py
# version 3.0: 12/04/2011
# 12/04/2012: using an external module "astLib" to calculate the distance of two fibres
# 17/04/2012: add new ReadImgFITS() class
# Last update: 17/04/2012

import os
import math
import os.path as op
#import fnmatch as fm
import pyfits as pf
import mefos as mf
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import astLib.astCoords as ac    # "astLib" module from http://astlib.sourceforge.net/

class TimingAnalysis(object):

    def __init__(self):
        pass

    def fourier(self, flux, time_res):
        """ DFT
            Usage: fourier(flux, time_res, length)
        """
        length = len(flux)
        thz = np.fft.fftfreq(length, d=time_res)
        temp = np.fft.fft(flux)
        power = (np.abs(temp[1:length/2])**2)/np.var(flux)
        
        return thz[1:length/2], power
                            
    def scargle(self, time, flux, time_res, nfreq=512, min_freq=0, fap=0.01, graph=False):
        """ Scargle periodogram
            Usage: scargle(time, flux, time_res)
        """
        # get total time interval
        time_interval = time[-1] - time[0]
                
        # get frequencies range
        if min_freq == 0:
            fund_freq = 2.0*np.pi/time_interval
            print('Fundational Frequency: %5.4f Hz' % (fund_freq/(2.0*np.pi)))
        elif min_freq != 0:
            fund_freq = min_freq*2.0*np.pi
            print('Fundational Frequency: %5.4f Hz' % (fund_freq/(2.0*np.pi)))
        nyqu_freq = np.pi/time_res
        print('Nyquist Frequency: %5.4f Hz' % (nyqu_freq/(2.0*np.pi)))
        if nfreq == 0:
            omega = np.linspace(fund_freq, nyqu_freq, len(flux)/2.0)
        elif nfreq != 0:
            omega = np.linspace(fund_freq, nyqu_freq, nfreq)
        print('Number of frequency searched... %5d' % (len(omega)))
        # total variance of the data
        var_d = np.var(flux)
        
        # calculating tau
        a = np.array([])
        b = np.array([])
        for ondex in range(len(omega)):
            temp_a = np.sum(np.sin(2*omega[ondex]*time))
            temp_b = np.sum(np.cos(2*omega[ondex]*time))
            a = np.append(a, temp_a)
            b = np.append(b, temp_b)
        tau = np.arctan(a/b)/(2*omega)
                
        # calculating power
        power = np.array([])
        
        for pndex in range(len(omega)):
            temp_c = np.sum(flux*np.cos(omega[pndex]*(time-tau[pndex])))
            temp_d = np.sum(flux*np.sin(omega[pndex]*(time-tau[pndex])))
            temp_e = np.sum(np.cos(omega[pndex]*(time-tau[pndex]))**2)
            temp_f = np.sum(np.sin(omega[pndex]*(time-tau[pndex]))**2)
            temp_p = (1./2.)*(temp_c**2/temp_e + temp_d**2/temp_f)
            power = np.append(power, temp_p)
        
        nor_power = power/var_d
        frequency = omega/(2.0*np.pi)
        
        # false alarm probability
        z_zero = -np.log(1.0-(1.0-fap)**(1./len(omega)))
        
        if graph == 1:
            plt.clf()
            plt.plot(frequency, nor_power, 'k-')
            plt.axhline(y=z_zero, color='r', ls='-.')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Normalised Power')
        else:
            pass

        return frequency, nor_power, z_zero
        
    def foldlc(self, time, flux, time_zero, period):
        """Folding light curve
        """
        nor_time = time - time_zero
        t2phase_f, t2phase_i = np.modf(nor_time/period)
        # reverse nagtive value before the timezero
        nag_value = np.where(t2phase_f < 0)
    
        if len(nag_value[0]) != 0:
            t2phase_f[nag_value] = t2phase_f[nag_value] + 1.0
        else:
            pass
    
        phase = np.linspace(0, 1, 11)
        fflux = np.array([])
        fferr = np.array([])
        for index in range(phase[:-1].size):
            rg = np.logical_and(t2phase_f >= phase[index], t2phase_f < phase[index+1])
            idx = np.where(rg == 1)
            if idx[0].size != 0:
                fflux = np.append(fflux, np.median(flux[idx]))
                fferr = np.append(fferr, np.std(flux[idx]))
            else:
                fflux = np.append(fflux, 0)
                fferr = np.append(fferr, 0)
    
        # phase repeating from 0 to 2
        phase1 = phase[:-1] + 0.05
        phase2 = phase[:-1] + 1.05
        phase_out = np.concatenate((phase1, phase2))
        fflux_out = np.concatenate((fflux, fflux))
        fferr_out = np.concatenate((fferr, fferr))
        return phase_out, fflux_out, fferr_out
        
    def highpassfilter(self, flux, timeresolution, threshold):
        """
        low frequency filter to detrend the lightcurves
        threshold is given in second.
        """
        # using FFT
        length = flux.size
        freq = np.fft.fftfreq(length, d=timeresolution)
        ft = np.fft.fft(flux)
        
        # giving high pass threshold
        hzthreshold = 1.0/threshold    # convert from second to Hz
        hpf = abs(freq) > hzthreshold
        hpfflux = np.fft.ifft(ft*hpf)    # inverse FFT of high pass filtered original FT
        
        return np.median(flux) + hpfflux.real
    
    def lowpassfilter(self, flux, timeresolution, threshold):
        """
        high frequency filter to remove noise from the lightcurves
        threshold is given in second.
        """
        # using FFT
        length = flux.size
        freq = np.fft.fftfreq(length, d=timeresolution)
        ft = np.fft.fft(flux)
        
        # giving low pass threshold
        hzthreshold = 1.0/threshold    # convert from second to Hz
        lpf = abs(freq) <= hzthreshold
        lpfflux = np.fft.ifft(ft*lpf)    # inverse FFT of low pass filtered original FT
        
        return lpfflux.real
            
    def standardscore(self, time, flux, window_size):
        """
        Checking the local mean (a concept of moving average process, in private communication with Joe). Then a standarad deviation (rms) is calucated. The output includes local means, and deviation score (standard score) and value of rms of the data.
        window_size equals to two sides of wind, unit in second
        window definition:
         |-------------|-------------|
         A             i             B
        -p                           p
        window: AB, exc point i
        
        Changes: rms calulation
        Changes: defining window   3/04/2012 
        """
        p = int((window_size/0.05)/2.0)    # number of bins on each side
        mwflux = np.array([])
        sscore = np.array([])

        #for index, value in enumerate(time):
        endpoint = flux.size - 1
        for i in range(flux.size):
            A, B = i-p, i+p+1
            if A < 0:
                B = B + abs(A)
                A = 0
            if B > endpoint:
                A = A - (B - endpoint)
                B = -1
            wflux = np.r_[flux[A:i+1], flux[i+1:B]]
            # statistics within window
            mwflux = np.append(mwflux, np.mean(wflux))
            
        rms = np.sqrt(np.sum((flux - mwflux)**2)/flux.size)
        sscore = (flux - mwflux)/rms
        
        return [mwflux, sscore], rms
        
    def CCF(self, series_a, series_b, bintime, lagtime):
        """
        Cross-Correlation function
        """
        # The cross-correlation function starts here
        # means
        mean_series_a = np.mean(series_a)    # time series A
        mean_series_b = np.mean(series_b)    # time series B
        
        length = len(mean_series_a)    # after both data are the same length
        pccf_array = np.array([])
        nccf_array = np.array([])
        exptime = bintime
        lagstep = int(lagtime/exptime)
        # Positive Part
        k_value = np.arange(0, lagstep-1)
        for k in k_value:
            sampa = (series_a[0:length-k-1] - mean_series_a)
            stdspa = np.std(series_a[0:length-k-1])
            sampb = (series_b[0+k:length-k-1+k] - mean_series_b)
        stdspb = np.std(series_b[0+k:length-k-1+k])
        
        pccvf = np.sum((sampa * sampb))/length
        pccf = pccvf/(stdspa*stdspb)
        pccf_array = np.append(pccf_array, pccf)
        # Negative Part
        p_value = np.arange(1, lagstep-1)
        for p in p_value:
            sampb = (series_b[0:length-p-1] - mean_series_b)
            stdspb = np.std(series_b[0:length-p-1])
            sampa = (series_a[0+p:length-p-1+p] - mean_series_a)
            stdspa = np.std(series_a[0+p:length-p-1+p])
        
            nccvf = np.sum((sampb * sampa))/length
            nccf = nccvf/(stdspb*stdspa)
            nccf_array = np.append(nccf_array, nccf)
        # flip nagative part array
        p_value = p_value[-1::-1]*-1
        nccf_array = nccf_array[-1::-1]
        # conbine result
        lag = np.concatenate((p_value, k_value))*exptime
        ccf = np.concatenate((nccf_array, pccf_array))
    
        return lag, ccf

class FibreExtensionList2(object):

    def __init__(self, openFITSFile):
        priHeader = openFITSFile[0].header
        numFibre = priHeader['FIBREACT']
        translate = {}
        for i in range(numFibre):
            Ext = 'EXTEN_{:02}'.format(i+1)
            translate[priHeader[Ext]] = i+1
        self.translate = translate
        
    def fibre2Extension(self, fibre):
        """
        Use this one instead of get_tranlation()
        """
        extension = self.translate[str(fibre)]
        return extension
                
class FibreExtensionList(object):

    def __init__(self, filename):
        temp = pf.open(filename)
        priheader = temp[0].header
        temp.close()
        num_actfibre = priheader['FIBREACT']
        translate = {}
        for index in range(num_actfibre):
            Ext = 'EXTEN_{:02}'.format(index+1)
            translate[priheader[Ext]] = index+1
        self.translate = translate
                
    def get_translation(self, fibre):
        """
        This is going to be phased out! Don't use this method, instead, using fibre2Extension()
        """
        extension = self.translate[str(fibre)]
        return extension
        
    def fibre2Extension(self, fibre):
        """
        Use this one instead of get_tranlation()
        """
        extension = self.translate[str(fibre)]
        return extension

class ReadImgFITS(object):

    def __init__(self, imgDir):
        self.imgDir = imgDir
        
    def _getFileName(self, FileName):
        fullImgPath = '{}/{}'.format(self.imgDir, FileName)
        self.temp = pf.open(fullImgPath)
        self.feList = FibreExtensionList(fullImgPath)
        self.FileName = FileName
        
    def _getImage(self, fibre):
        extension = self.feList.fibre2Extension(fibre)
        imgCube = self.temp[extension].data
        return imgCube
        
    def close(self):
        self.temp.close()
                      
class ReadLcsFits(object):

    def __init__(self, lc_dir):
        self.lc_dir = lc_dir
    
    def setFileName(self, fileName):
        filePath = '{}/{}'.format(self.lc_dir, fileName)
        self.temp = pf.open(filePath)
        self.fileName = fileName
        self.fibreExtDic = self.getFibreExtensionList()
        
    def getFibreExtensionList(self):
        fibreExtDic = FibreExtensionList2(self.temp).translate
        return fibreExtDic
            
    def _view_primary_header(self):
        print self.temp[0].header
        
    def viewPrimaryHeader(self):
        print self.temp[0].header
    
    def _get_jd_timezero(self):
        header = self.temp[0].header
        jd = header['JD']
        return jd
        
    def getJDTimezero(self):
        header = self.temp[0].header
        jd = header['JD']
        return jd
        
    def _get_exptime(self):
        header = self.temp[0].header
        exptime = np.float32(header['EXPTIME'])
        return exptime
        
    def getExptime(self):
        header = self.temp[0].header
        exptime = np.float32(header['EXPTIME'])
        return exptime
        
    def _get_coord(self, fibre):
        header = self.temp[0].header
        extension = self.fibreExtDic[str(fibre)]
        RA = header['RA_{:02}'.format(extension)]
        DEC = header['DEC_{:02}'.format(extension)]
        return RA, DEC
        
    def getCoordinate(self, fibre):
        header = self.temp[0].header
        extension = self.fibreExtDic[str(fibre)]
        RA = header['RA_{:02}'.format(extension)]
        DEC = header['DEC_{:02}'.format(extension)]
        return RA, DEC
        
    def _getDistance(self, threshold=3.0):
        """
        looking for nearby fibres. The unit of distance is arcmin (')
        """
        fibrelist = self.fibreExtDic.keys()
        fibdis = {}
        for fibre in fibrelist:
            fRA = ac.hms2decimal(self.getCoordinate(fibre)[0], ':')
            fDEC = ac.dms2decimal(self.getCoordinate(fibre)[1], ':')
            
            temp_neigbhour = []
            for key, value in self.fibreExtDic.iteritems():
                if key == fibre:    # if the two fibres are actually the same, do nothing
                    pass
                else:
                    dRA = ac.hms2decimal(self.getCoordinate(key)[0], ':')
                    dDEC = ac.dms2decimal(self.getCoordinate(key)[1], ':')
                    distance = np.sqrt((dRA - fRA)**2 + (dDEC - fDEC)**2)*60.0
                    if distance <= threshold:
                        temp_neigbhour.append(int(key))
            fibdis[int(fibre)] = temp_neigbhour
        return fibdis

    def getFullData(self, fibre):
        extension = self.fibreExtDic[str(fibre)]
        data = self.temp[extension].data
        time = data.field(0)
        flux = data.field(1)
        xpos = data.field(2)
        ypos = data.field(3)
        aptp = data.field(4)
        skyv = data.field(5)
        qual = data.field(6)
        return time, flux, xpos, ypos, aptp, skyv, qual
        
    def getLcs(self, fibre, rebin=False, bintime=1.0):
        extension = self.fibreExtDic[str(fibre)]
        data = self.temp[extension].data
        time = data.field(0)
        flux = data.field(1)
        qual = data.field(6)
        # check NaN value in flux
        inan = np.where(np.isnan(flux) == 1)
        flux[inan] = 0.0
        
        if rebin == 1:
            nbin = int(np.float32(bintime)/self._get_exptime())
            numCut = len(time)/nbin
            newTime = np.mean(np.split(time, numCut), axis=1)
            newFlux = np.mean(np.split(flux, numCut), axis=1)
            time, flux = newTime, newFlux
            return time, flux
        else:
            return time, flux, qual
        
        
    def lcReview(self, fibre, eventList=[], save=False, saveName='default.png'):
        """
        """
        time, flux, xpos, ypos, apts, msky, qual = self.getFullData(fibre)
        
        TA = TimingAnalysis()
        result, rms = TA.standardscore(time, flux, 2.0)
        wmflux = result[0]
        mSNR = np.median(wmflux/rms)

        plt.figure(100, figsize=(8,12))
        plt.clf()
        plt.subplots_adjust(hspace=0.2)
        
        plt.subplot(4,1,1)
        plt.plot(time, flux, 'k+-', drawstyle='steps-mid')
        plt.plot(time, wmflux, 'r-', label='averaged')
        if len(eventList) == 0:
            pass
        else:
            for value in eventList:
                plt.axvline(x=value, color='red', ls='--')
        plt.xlim(time[0], time[-1])
        plt.ylim(0, np.median(flux) + 0.5*np.median(flux))
        plt.ylabel('Flux (count)')
        plt.xticks(visible=False)
        plt.legend()
        plt.title('{}, fibre {}. mSNR: {:.2f}'.format(self.filename[:-5], fibre, mSNR))
        
        plt.subplot(4,1,2, sharex=plt.subplot(4,1,1))
        plt.plot(time, xpos, 'b+', time, ypos, 'r+')
        plt.xlim(time[0], time[-1])
        plt.ylim(5,35)
        plt.xticks(visible=False)
        plt.ylabel('Position (pixel)')
        
        plt.subplot(4,1,3, sharex=plt.subplot(4,1,1))
        plt.plot(time, apts, 'k+')
        plt.xlim(time[0], time[-1])
        plt.ylim(0, np.mean(apts) + 0.5*np.mean(apts))
        plt.xticks(visible=False)
        plt.ylabel('Apt Size (pixel)')
        
        plt.subplot(4,1,4, sharex=plt.subplot(4,1,1))
        plt.plot(time, msky, 'k+')
        plt.xlim(time[0], time[-1])
        plt.ylim(0, np.median(msky) + 0.5*np.median(msky))
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Sky Value (count)')
        
        plt.show()
        
    def close(self):
        self.temp.close()

class OnSiteAnalysis(object):

    def __init__(self, filename, fibre):
        temp = pf.open(filename)
        FEList = FibreExtensionList(filename)
        extension = FEList.fibre2Extension(fibre)
        self.primaryheader = temp[0].header
        self.filename = filename
        self.fibre = fibre
        self.exposuretime = self.primaryheader['EXPTIME']
        self.dateobs = self.primaryheader['DATE-OBS']
        self.jd = self.primaryheader['JD']
        self.ra = self.primaryheader['RA_' + str(extension).zfill(2)]
        self.dec = self.primaryheader['DEC_' + str(extension).zfill(2)]
        self.extimage = temp[extension].data
        temp.close()
    
    def useringsky(self, aperture, inner_annulus, outer_annulus, centroid='c'):
        lightcurves = []
        for index in range(self.extimage.shape[0]):
            print index
            photometry = mf.APhot(self.extimage[index])
            # Image centring
            if centroid == 'g':
                best_gfit = photometry.fitguassian()
                centre = photometry.gauss_centre(best_gfit)
            if centroid == 'c':
                centre = photometry.centroid()
            
            # Measurement of source
            apt = photometry.aptvalue(centre, aperture)
            # Measurement of background
            sky = photometry.skyvalue(centre, inner_annulus, outer_annulus)
            # measuring source intensity
            intensity = apt[1] - apt[0]*sky[2]
            # column: time, intensity, x, y, aperture size, sky value
            lightcurves.append([index*self.exposuretime, intensity, centre[0], centre[1], apt[0], sky[2]])
        self.r_lightcurves = np.asarray(lightcurves)
        self.aperture = aperture
        
    def usefullsky(self, aperture, inner_annulus, centroid='c'):
        lightcurves = []
        for index in range(self.extimage.shape[0]):
            print index
            photometry = mf.APhot(self.extimage[index])
            # Image centring
            if centroid == 'g':
                best_gfit = photometry.fitguassian()
                centre = photometry.gauss_centre(best_gfit)
            if centroid == 'c':
                centre = photometry.centroid()
                
            # Measurement of source
            apt = photometry.aptvalue(centre, aperture)
            # Measurement of background
            sky = photometry.skyvalue2(centre, inner_annulus)
            # measuring source intensity
            intensity = apt[1] - apt[0]*sky[2]
            # column: time, intensity, x, y, aperture size, sky value
            lightcurves.append([index*self.exposuretime, intensity, centre[0], centre[1], apt[0], sky[2]])
        self.f_lightcurves = np.asarray(lightcurves)
        self.aperture = aperture
        
    def psd(self, lc_data, n=1024, save=False):
        flux = lc_data[:,1]
        # check NaN value, and reset it to zero
        nani = np.where(np.isnan(flux) > 0)
        flux[nani] = 0.0
        
        thz = np.fft.fftfreq(n, d=self.exposuretime)
        fre = []
        for index in range(len(flux)/n):
            temp = np.fft.fft(flux[index*n:(index + 1)*n])
            fre.append(temp[1:n/2])
        fre_array = np.asarray(fre)
        avg_fre_array = np.average(fre_array, axis=0)
        power = (np.abs(avg_fre_array)**2)/thz[1:n/2]
        
        plt.clf()
        ps = plt.gca()    
        ps.plot(thz[1:n/2], power, 'k.-')
        ps.set_xscale('log')
        ps.set_yscale('log')
        ps.set_xlabel('Frequency (Hz)')
        ps.set_ylabel(r'PSD (power$^2$/frequency)')
        ps.set_title('PSD: ' + op.basename(self.filename) + r', n=%4d, $\Delta$t=%4.2f sec' % (n, self.exposuretime))
        plt.show()
        
        if save == 1:
            save_filename = 'f' + str(self.fibre) + '_' + op.basename(self.filename)[:-5] + '_psd.png'
            plt.savefig(save_filename, format='png')

    def plotreport(self, lc_data, ext_image=None, save=False):
        time = lc_data[:,0]
        flux = lc_data[:,1]
        xpos = lc_data[:,2]
        ypos = lc_data[:,3]
        pixa = lc_data[:,4]
        msky = lc_data[:,5]

        #searching NaN value in flux
        nani = np.where(np.isnan(lc_data[:,1]) == 1)
        # transferring list to array
        nan_array = np.asarray(nani[0])
        
        #remove NaN elements from data
        vali = np.where(np.isnan(lc_data[:,1]) == 0)

        flux_sans_NaN = lc_data[vali,1]
        m_flux = np.median(flux_sans_NaN)
        std_flux = np.std(flux_sans_NaN)
        
        # check total number of pixels in aperture less than 3-sigam of median value
        tnp_index = np.where(pixa < np.median(pixa)-np.std(pixa)*3.0)
        
        plt.clf()
        plt.figure(1, figsize=(8,11))
        plt.subplots_adjust(hspace=0.2)
                
        win2 = plt.subplot(411)
        win2.plot(time, flux, 'k-', drawstyle='steps-mid')
        win2.plot(time[tnp_index], time[tnp_index]*0.0 + (m_flux+3.0*std_flux), 'rv')
        if nan_array.size > 0:
            for index in range(nan_array.size):
                win2.axvline(x=time[nan_array[index]], color='r', ls='--')
        win2.axhline(y=(m_flux-3.0*std_flux), color='r', ls='-.')
        win2.axhline(y=(m_flux+3.0*std_flux), color='r', ls='-.')
        win2.set_xlim(0, time[-1])
        win2.set_ylim(0, m_flux+5.0*std_flux)
        win2.set_ylabel('Intensity (count)')
        win2.set_title('Light Curves: ' + op.basename(self.filename) + ' [f' + str(self.fibre) + ']')
        
        win1 = plt.subplot(412, sharex=win2)
        win1.plot(time, xpos, 'g.', label='x')
        win1.plot(time, ypos, 'b.', label='y')
        #an = np.linspace(0,2*np.pi,359)        
        #win1.imshow(ext_image[0], cmap=plt.cm.gray, vmin=10, vmax=1000)
        #win1.plot(5*np.sin(an)+xpos[0], 5*np.cos(an)+ypos[0], 'r-')
        #win1.plot(10*np.sin(an)+xpos[0], 10*np.cos(an)+ypos[0], 'r--')
        win1.set_xlim(0, time[-1])
        win1.set_ylim(0, 40)
        win1.set_ylabel('Position')
        win1.legend()
        
        win3 = plt.subplot(413, sharex=win2)
        win3.plot(time, pixa, 'k-', drawstyle='steps-mid')
        espected_pixa = np.pi*self.aperture**2
        win3.axhline(y=espected_pixa, color='r', ls='--')
        if nan_array.size > 0:
            for index in range(nan_array.size):
                win3.axvline(x=time[nan_array[index]], color='r', ls='--')    
        win3.set_xlim(0, time[-1])
        win3.set_ylim(np.mean(pixa)-6.0*np.std(pixa), np.mean(pixa)+6.0*np.std(pixa))
        win3.set_ylabel('Pixel')
        win3.set_title('Total pixels in aperture (%2d)' % (self.aperture))
        
        win4 = plt.subplot(414, sharex=win2)
        if nan_array.size > 0:
            for index in range(nan_array.size):
                win4.axvline(x=time[nan_array[index]], color='r', ls='--')
        win4.plot(time, msky, 'k-', drawstyle='steps-mid')
        win4.set_xlim(0, time[-1])
        win4.set_xlabel('Time (sec)')
        win4.set_ylabel('Sky value (count/pixel)')
        win4.set_title('Median sky value')
        
        xticklabels = win1.get_xticklabels() + win2.get_xticklabels() + win3.get_xticklabels()
        plt.setp(xticklabels, visible=False)

        plt.show()
        
        if save == 1:
            save_filename = 'f' + str(self.fibre) + '_' + op.basename(self.filename)[:-4] + 'png'
            plt.savefig(save_filename, format='png')

def on_press(event):
    mouseevent = event.mouseevent
    artist = event.artist
    data = artist.get_array()
    xind = math.trunc(mouseevent.xdata)
    yind = math.trunc(mouseevent.ydata)
    print 'X: %2d, Y: %2d, Height: %8.2f' % (xind, yind, data[yind,xind])