#!/usr/bin/env python
# MIOSOTYS Data Reduction Processes 2011b
# Please notice: this code is for data taken after December 2011.
#
# Created by I Chun Shih @ Observatoire de Paris, 2011
#
# How the data is reduced?

# Although the class mefos provides functions to correct dark/bias in the images, it is known that the PI EMCCD has pixel-level bias stability problem. Such problem makes conventional data reduction unreliable. As a result we DO NOT apply dark/flat correction to image reduction.

# This converter will only crop the fibre out and save each fibre into a FITS extension.

# From December 2011, a lens has been attached to each fibre to increase the focusing. As a result, we see full sky within the fibre bundle, and outside of the sky is purely the dark of the CCD pixel. Therefore we will use this local dark to subtract each image, rather than using a seperated bias image which is known to be fluctuated.

# external package dependency
# numpy: numerical calculation and array function support. Part of Numpy/Scipy packages.
# pyfits: FITS modules for python.
# mefos: classes for reading SPE file format, Assignment file, and fibre position file, as well as a photometric class for MIOSOTYS-specified photometry functions. This is only useful for the data from OHP.
 
import time
import os
import fnmatch
import os.path as op
import mefos as mf
import numpy as np
import pyfits as pf
import matplotlib.pylab as plt

class Converter(object):
    """ Converting 2011 SPE images to FITS format
    """
    
    def __init__(self, data_id_dir, flat_corr=False):
        # data directories
        # input
        self.obs_id = data_id_dir[-8:]
        self.raw_path = data_id_dir + '/raw/'
        self.calib_path = data_id_dir + '/calib/'
        self.data_id_dir = data_id_dir
    
    def get_dataimage(self, spe_filename):
        """
        """
        self.spe_filename = spe_filename
        self.spefile_fullpath = self.raw_path + '/' + self.spe_filename
        imgf = mf.ReadSPEFile(self.spefile_fullpath)
        self.data_cube = imgf.image()
        self.date = imgf.iso8601date()
        self.lst = imgf.obslst()
        self.utc = imgf.obsutc()
        self.jd = imgf.tojd()
        self.exptime = imgf.exptime()
        self.readout = imgf.readtime()
        self.gain = imgf.gain()
        self.emgain = imgf.avalancegain()
        self.temperature = imgf.temperature()
        imgf.close()

    def get_flatimage(self, flat_filename):
        """
        """
        self.flatfile_fullpath = self.calib_path + '/' + flat_filename
        imgc = mf.ReadSPEFile(self.flatfile_fullpath)
        self.calib_cube = imgc.image()
        imgc.close()
    
    def get_position(self, position_filename):
        """
        """
        get_pos = mf.ReadPOSFile(self.calib_path + '/' + position_filename)
        self.position = get_pos.get_position()
        
    def get_assignment(self, assign_filename):
        """
        """
        get_ass = mf.ReadASSFile(self.calib_path + '/' + assign_filename)
        self.assignment = get_ass.get_armcoor()
        
    def make_spefile_list(self, target_id):
        """make SPE file list"""
        data_id = self.data_id_dir[-8:]
        template = data_id + '*' + target_id + '*.SPE'
    
        file_match = []
        for spe_file in os.listdir(self.raw_path):
            if fnmatch.fnmatch(spe_file, template):
                file_match.append(spe_file)
        return file_match
        
    def convert2fits(self, window_szie, pi_name, output_dir, digi_spe_number, flat_corr=False):
        # save to fits file
        prihdu = pf.PrimaryHDU(self.data_cube[0])    # save a snapshot
        # Important information
        # 1. Basic observation information
        prihdu.header.update('FILENAME', self.spe_filename[:-4], '')
        prihdu.header.update('TELESCOP', 'OHP 1.93m')
        prihdu.header.update('INSTRUME', 'MIOSOTYS')
        prihdu.header.update('OBSERVER', pi_name, 'P.I.')
        # prihdu.header.update('OBJECT', target_id, 'Type of scientific objective')
        # 2. Date and Time
        prihdu.header.update(self.date[0], self.date[1], self.date[2])
        prihdu.header.update(self.lst[0], self.lst[1], self.lst[2])
        prihdu.header.update(self.utc[0], self.utc[1], )
        prihdu.header.update(self.jd[0], self.jd[1], self.jd[2])
        # 3. Calibration
        prihdu.header.update('NUMSHOT', self.data_cube.shape[0], 'Number of shots')
        # prihdu.header.update('BINSIZE', binsize, 'Binning scale')
        prihdu.header.update('BPIXELX', window_szie, 'X axis size of fibre image')
        prihdu.header.update('BPIXELY', window_szie, 'Y axis size of fibre image')
        prihdu.header.update(self.exptime[0], self.exptime[1], self.exptime[2])
        prihdu.header.update(self.readout[0], self.readout[1], self.readout[2])
        prihdu.header.update(self.temperature[0], self.temperature[1], self.temperature[2])
        prihdu.header.update(self.emgain[0], self.emgain[1], self.emgain[2])
        prihdu.header.update('FIBREACT', len(self.assignment.keys()), 'Number of fibres activated')
        ext_index = 1
        imagecontent = []

        for key, value in self.assignment.iteritems():
            prihdu.header.update('EXTEN_'+str(ext_index).zfill(2), str(key), 'fibre arm')
            prihdu.header.update('RA_'+str(ext_index).zfill(2), value[0])
            prihdu.header.update('DEC_'+str(ext_index).zfill(2), value[1])
            prihdu.header.update('MAG_'+str(ext_index).zfill(2), value[2])
            Xstart = self.position[key][0]
            Xend = Xstart + window_szie
            Ystart = self.position[key][1]
            Yend = Ystart + window_szie
            cut_image = self.data_cube[:,Ystart:Yend,Xstart:Xend]
            
            imgtemp = pf.ImageHDU(cut_image)
            imagecontent.append(imgtemp)
            
            ext_index = ext_index + 1
            
        prihdu.header.update('EQUINOX', 2000.0)
        currenttime = time.strftime('%x %X %Z')
        prihdu.header.add_comment('Image status: Converted from WinView SPE to FITS by Shih I Chun')
        prihdu.header.add_comment('Image status: Generated on ' + currenttime)
        
        # Writing newly created fits into assigned directory
        hdulist = pf.HDUList([prihdu] + imagecontent)

        if op.exists(output_dir) == 1:
            pass
        else:
            os.mkdir(output_dir)

        fits_filename = self.spe_filename[:20] + str(self.spe_filename[20:-4]).zfill(digi_spe_number) + '.fits'
        output_path = output_dir + '/' + fits_filename
        hdulist.writeto(output_path)

def convert_seul(main_dir, spefile, assignment, position):
    """
    Converting one SPE image to FITS, only for quick check!
    """
    temp = Converter(main_dir)
    ass_filename = assignment
    pos_filename = position
    temp.get_assignment(ass_filename)    # read assignment file
    temp.get_position(pos_filename)    # read position file
    temp.get_dataimage(spefile)    # read spe image file
    fits_image_dir = '{}/reduced'.format(main_dir)
    temp.convert2fits(40, 'Test O.', fits_image_dir, 1)
    
def convert_tout(main_dir, target_id, pi_name, save_dir='reduced'):
    
    temp = Converter(main_dir)
    data_file_list = temp.make_spefile_list(target_id)
    number_of_file = str(len(data_file_list))
    
    ass_filename = main_dir[-8:-2] + '_' + target_id[:-3] + '.ASS'
    pos_filename = main_dir[-8:-2] + '_2.POS'
    
    temp.get_assignment(ass_filename)
    temp.get_position(pos_filename)
    
    # creating directories
    saved_dir = main_dir + '/' + save_dir
    if op.exists(saved_dir) == 1:
        pass
    else:
        os.mkdir(saved_dir)
        
    #image_dir = saved_dir + '/' + main_dir[-8:] + '_' + target_id + '_images/'
    image_dir = saved_dir + '/' + target_id + '_images/'
    if op.exists(image_dir) == 1:
        pass
    else:
        os.mkdir(image_dir)
    
    for spe_filename in data_file_list:
        print('Processing ' + spe_filename + ' ...')
        temp.get_dataimage(spe_filename)
        temp.convert2fits(40, pi_name, image_dir, len(number_of_file))
        
def flat2mask(flat_image):
    """ Creating mask from flat image
    """
    flat1d = np.ravel(flat_image)
    flat1ds = np.sort(flat1d)
    turncated = flat1ds[304]    # 304 = 40 X 40 - 36 X 36
    mask = flat_image > turncated
    return mask
        
def normalflat(masked_flat_image, degree=2.0):
    """ Normalising the size specified flat image 
    """
    flat_image = masked_flat_image
    i = np.where(flat_image > 0)
    low_limit = np.median(flat_image[i]) - degree*np.std(flat_image[i])
    filtered = flat_image > low_limit
    ffimage = flat_image*filtered
    fmax = np.amax(ffimage)
    #norflat_image = np.float32(ffimage)/np.float32(fmax)
    return fmax