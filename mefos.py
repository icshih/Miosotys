#!/usr/bin/env python
# MIOSOTYS data convertion and photometric tools.
# I Chun Shih @ Observatoire de Paris, Meudon, 2009-2011
#
# This program consists of several parts: 
# 1. ReadSPEFile() class. This class reads SPE binary data.
# 2. ReadASSFile() class. This class reads fibre assignment file created by FIFI.
# 3. ReadPOSFile() class. This class reads file which records fibre positions in the image.
# 4. converter() function. This function convert SPE to FITS format.
# In this version, the program assumes that the files follow the convention below:
#              00000000001111111111
#              01234567890123456789
#    SPE file: YYYYMMDD_B_MMMNN_XX_$$.SPE
#    ASS file: MMMNN.ASS
#    POS file: YYYYMM_B.POS
#    BIAS    : YYYYMMDD_B_BIAS_XX.SPE
#    FLAT    : YYYYMMDD_B_FLAT_XX.SPE

# update 09/05/2011: correcting jd calculation tojd()
# update 10/01/2011: correction of data types
#                    _xdim: np.int16 -> np.uint16 (WORD)
#                    _ydim: np.int16 -> np.uint16 (WORD)
#                    _zdim: np.int16 -> np.int32 (long)
#                    image: np.int16 -> np.uint16 (WORD)
# For bright images, such as flat-fields, the value of pixels are usually larger than 32768, 
# using 'short integer' will result in wrong negative value. In this version, 
# the data types of variables have been corrected to either unsigned integer (WORD) or long integer (long).  

# short: np.int16
# WORD:  np.uint16
# long:  np.int32
# float: np.float32
# char:  'S','a'

import os
import sys
import os.path as op
import numpy as np
import pyfits as pf
from scipy import optimize

class ReadSPEFile(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._dimension()
    
    def _dimension(self):
        self._xdim = int(self.read_at(42, np.uint16, 1)[0])
        self._ydim = int(self.read_at(656, np.uint16, 1)[0])
        self._zdim = int(self.read_at(1446, np.int32, 1)[0])
        
    def read_at(self, pos, ntype, size):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)
        
    def image(self):
        img = self.read_at(4100, np.uint16, self._xdim * self._ydim * self._zdim)
        return img.reshape((self._zdim, self._ydim, self._xdim))
        
    def avg_red_image(self):
        """Averaging bias and flatfield images."""
        if self._zdim == 1:
            bias_image = self.image()[0,:,:]
        else:
            bias_image = np.mean(self.image(), axis=0)
        return bias_image
            
    def obsdate(self):
        dat = self.read_at(20, 'S10', 1)[0]
        month_list = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        return ('DATE-OBS', dat[5:10] + '-' + month_list[dat[2:5]] + '-' + dat[:2])
        
    def obsutc(self):
        utc = self.read_at(179, 'S7', 1)[0]
        return ('UTC-OBS', utc[:2] + ':' + utc[2:4] + ':' + utc[4:6])
        
    def obslst(self):
        lst = self.read_at(172, 'S7', 1)[0]
        return ('LST-OBS', lst[:2] + ':' + lst[2:4] + ':' + lst[4:6], 'Local time')
        
    def iso8601date(self):
        dat = self.read_at(20, 'S10', 1)[0]    # The date is also in local time
        lst = self.read_at(172, 'S7', 1)[0]
        month_list = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        iso8601 = dat[5:10] + '-' + month_list[dat[2:5]] + '-' + dat[:2] + 'T' + lst[:2] + ':' + lst[2:4] + ':' + lst[4:6]
        return ('DATE-OBS', iso8601, 'Beginning date and time of observation (LT).')
        
    def tojd(self):
        month_list = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        dat = self.read_at(20, 'S10', 1)[0]
        utc = self.read_at(179, 'S7', 1)[0]
        lst = self.read_at(172, 'S7', 1)[0]

        # Always using UTC
        second = int(utc[4:6])
        minute = int(utc[2:4])
        hour = int(utc[:2])
        day = int(dat[:2])
        month = int(month_list[dat[2:5]])
        year = int(dat[5:10])
        
        # Gregorian calendar date to Julian Day Number (Wikipedia)
        a = (14 - month)//12
        y = year + 4800 - a
        m = month + 12*a - 3
        jdn = day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045
        jd = jdn + (hour-12)/24. + minute/1440. + second/86400.
        
        lshour = int(lst[:2])
        if lshour < hour:
            jd = jd - 1
        
        return ('JD', jd, 'Julian Date/Time (still under test!)')
    
    def shutter(self):
        shu = self.read_at(50, np.uint16, 1)[0]
        return ('SHUTTER', shu, 'Shutter gate operation')
            
    def gain(self):
        gai = self.read_at(198, np.uint16, 1)[0]
        return ('GAIN', gai, 'ADC gain')
    
    def exptime(self):
        expt = self.read_at(10, np.float32, 1)[0]
        return ('EXPTIME', expt, 'second')
        
    def readtime(self):
        readt = self.read_at(672, np.float32, 1)[0]
        return ('READOUT', readt, 'millisecond')
    
    def temperature(self):
        ccdtemp = self.read_at(36, np.float32, 1)[0]
        return ('CCDTEMP', ccdtemp, 'Celsius')
        
    def avalancegain(self):
        avgain = self.read_at(4096, np.int16, 1)[0]
        return ('EMGAIN', avgain, 'EM gain stages')
            
    def close(self):
        self._fid.close()


class ReadASSFile(object):

    def __init__(self, fname):
        self.ass = np.loadtxt(fname, skiprows=2, dtype=str)

    def get_armcoor(self):
        armcoor = {}
        for index in range(len(self.ass)):
            temp = self.ass[index]
            ra = temp[2] + ':' + temp[3] + ':' + temp[4]
            dec = temp[5] + ':' + temp[6] + ':' + temp[7]
            arm = int(temp[8])
            mag = float(temp[1])
            if mag == 0.0:
                pass
            else:
                armcoor[arm] = [ra, dec, mag]
        return armcoor


class ReadPOSFile(object):

    def __init__(self, fname):
        self.pos = np.loadtxt(fname, dtype=int)

    def get_position(self):
        poscoor = {}
        for index in range(len(self.pos)):
            temp = self.pos[index]
            poscoor[temp[0]] = [temp[1], temp[2]]
        return poscoor


class Check_Exist(object):
    def __init__(self, path_name):
        self.path_name = path_name
        self.check()
        
    def check(self):
        if op.exists(self.path_name) == 1:
            pass
        else:
            print 'The directory and/or file' + self.path_name + 'are not exist.'
            sys.exit()



# This class performs aperture photometry tasks for MEFOS image.
# No IRAF program is called, so it is much faster!
# 
# The details of how things are done in this code can be found here
# Handbook of CCD Astronomy by Steve B. Howell, 2000, Cambridge University Press
#
# version 0.1: 28/07/2010
#              The reduced image may contain NaN value, the method remove_nan() is excuted, and reset the value to zero in those pixels.
            
class APhot(object):
    """
    Aperture photometery for MEFOS data. 
    """
    
    def __init__(self, raw_image):
        #self.remove_nan(raw_image)
        self.image = raw_image
        self.y_size, self.x_size = raw_image.shape
        
    def remove_nan(self, raw_image):
        nan_index = np.where(np.isnan(raw_image))
        raw_image[nan_index] = 0.0
        self.image = raw_image
        self.y_size, self.x_size = raw_image.shape
    
    # 2d gaussian function
    def gaussian(self, height, centre_x, centre_y, width_x, width_y):
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((centre_x-x)/width_x)**2+((centre_y-y)/width_y)**2)/2)
    
    def moments(self):
        data = self.image
        
        total = np.abs(data).sum()
        Y, X = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y
    
    def fitguassian(self):
        data = self.image
        
        params = self.moments()
        errorfunction = lambda p:np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    
    def gauss_centre(self, bestfit):        
        print 'Target centre:'
        print 'X: %6.2f Y: %6.2f\n' % (bestfit[2], bestfit[1])
        return [bestfit[2], bestfit[1]]

    def centroid(self):
        """
        Searching for rough central position of the star.
        """
        #y_size, x_size = self.image.shape
        
        x_space = np.linspace(1, self.x_size, self.x_size)
        y_space = np.linspace(1, self.y_size, self.y_size)
        
        int_x = np.sum(self.image, axis=0)
        int_y = np.sum(self.image, axis=1)
        red_ix = int_x - np.mean(int_x)
        red_iy = int_y - np.mean(int_y)
        ix_gt_0 = np.where(red_ix > 0)
        iy_gt_0 = np.where(red_iy > 0)
        
        x_center = np.sum(red_ix[ix_gt_0]*x_space[ix_gt_0])/np.sum(red_ix[ix_gt_0])
        y_center = np.sum(red_iy[iy_gt_0]*y_space[iy_gt_0])/np.sum(red_iy[iy_gt_0])
        
        print 'Target centre:'
        print 'X: %6.2f Y: %6.2f\n' % (x_center, y_center)
        return [x_center, y_center]
            
    def aptvalue(self, centre, aperture):
        """
        Measuring aperture value.
        """
        x_centre = centre[0]
        y_centre = centre[1]
        y_index, x_index = np.indices(self.image.shape)
        # offset the indice of image
        x_off = x_index - x_centre
        y_off = y_index - y_centre
        radius_map = np.sqrt(x_off**2 + y_off**2)
        # creating aperture masks
        aperture_mask = radius_map < aperture
        # obtainning photons within the mask
        aperture_photon = self.image*aperture_mask
        # excluding zero value within the mask
        nonzero_aperture = np.extract(aperture_photon > 0.0, aperture_photon)
        
        print 'Aperture:'
        print 'Size*                 : %6d' % (aperture)
        print 'Number of pixel[0]    : %6d' % (len(nonzero_aperture))
        print 'Total photon number[1]: %6d\n' % (np.sum(nonzero_aperture))        
        return [len(nonzero_aperture), np.sum(nonzero_aperture)]
            
    
    def skyvalue(self, centre, inner_annulus, outer_annulus):
        """
        Measuring sky/backgound value.
        """
        x_centre = centre[0]
        y_centre = centre[1]
        y_index, x_index = np.indices(self.image.shape)
        # offset the indice of image
        x_off = x_index - x_centre
        y_off = y_index - y_centre
        radius_map = np.sqrt(x_off**2 + y_off**2)
        # creating sky masks
        sky_mask = np.logical_and(radius_map > inner_annulus, radius_map < outer_annulus)
        # obtainning photons within the mask
        sky_photon = self.image*sky_mask
        # excluding zero value within the mask
        nonzero_sky = np.extract(sky_photon > 0.0, sky_photon)
        # remove extreme values
        sky_median = np.median(nonzero_sky)
        sky_possion = np.sqrt(sky_median)    # backgroung is in Possion distribution!
        good_sky_value_mask = np.logical_and(nonzero_sky > (sky_median - 3.0*sky_possion), nonzero_sky < (sky_median + 3.0*sky_possion))
        good_sky_value = np.extract(nonzero_sky*good_sky_value_mask > 0.0, nonzero_sky*good_sky_value_mask)
        
        print 'Sky/Background:'
        print 'Number of pixel[0]    : %6d' % (len(nonzero_sky))
        print 'Total photon number[1]: %6d' % (np.sum(good_sky_value))
        print 'Median gd sky value[2]: %6.2f\n' % (np.median(good_sky_value))
        return [len(nonzero_sky), np.sum(good_sky_value), np.median(good_sky_value)]
        
    def skyvalue2(self, centre, inner_annulus):
        """
        Using all image to calculate sky/background value.
        """
        #boundary_mask = self.define_mask()
        #masked_image = self.image*boundary_mask
        
        x_centre = centre[0]
        y_centre = centre[1]
        y_index, x_index = np.indices(self.image.shape)
        # offset the indice of image
        x_off = x_index - x_centre
        y_off = y_index - y_centre
        radius_map = np.sqrt(x_off**2 + y_off**2)
        # creating mask outside the aperture
        outside_aperture_mask = radius_map > inner_annulus
        # obtainning photons within the mask
        outside_aperture_photon = self.image*outside_aperture_mask
        # sky value statisitcs
        sky_w_value = np.extract(outside_aperture_photon > 0.0, outside_aperture_photon)
        # remove extreme values
        sky_median = np.median(sky_w_value)
        sky_possion = np.sqrt(sky_median)    # backgroung is in Possion distribution!
        good_sky_value_mask = np.logical_and(sky_w_value > (sky_median - 3.0*sky_possion), sky_w_value < (sky_median + 3.0*sky_possion))
        good_sky_value = np.extract(sky_w_value*good_sky_value_mask > 0.0, sky_w_value*good_sky_value_mask)
        
        print 'Sky/Background:'
        print 'Number of pixel[0]    : %6d' % (len(sky_w_value))
        print 'Total photon number[1]: %6d' % (np.sum(good_sky_value))
        print 'Median gd sky value[2]: %6.2f\n' % (np.median(good_sky_value))
        return [len(sky_w_value), np.sum(good_sky_value), np.median(good_sky_value)]