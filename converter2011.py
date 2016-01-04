#!/usr/bin/env python
# MIOSOTYS Data Reduction Processes
# This code is to deal with the data for the mission in March 2011 ONLY!
#
# Created by I Chun Shih @ Observatoire de Paris, 2011
#
# How the data is reduced?
# Raw image are always bias subtracted, the bias image is saved in obsid/calib. Sometimes more than one bias images can be used. One should check the quality before using it.
#
# After subtracted, the full raw data is cropped accroding to the fibre position on the image. The fibre and position information are provided in *.ASS and *.POS. Both info change every mission, so ONLY using them from the same obsid.
# 
# Because the selection window is always slightly larger then the fibre, there is a thin edge which is not part of data. We use flat-field image to create a mask to reset this edge to zero. The quality of mask can be checked by calling IFSample().
#
# Two methods, convert_seul() and convert_tout(), are provied to either convert a single or a group of files, respectively. One can choose whether to correct the image with flat field or not. The correction is not perfect at the moment, and it is uncertain that this step does improve the quality of images. To SWITCH ON the correction, using flat_corr=True. The default option is flat_corr=False.

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
    
    def __init__(self, data_id_dir):
        # data directories
        # input
        self.obs_id = data_id_dir[-8:]
        self.raw_path = data_id_dir + '/raw/'
        self.calib_path = data_id_dir + '/calib/'
    
    def get_dataimage(self, spe_filename):
        """
        """
        spefile_fullpath = self.raw_path + '/' + spe_filename
        imgf = mf.ReadSPEFile(spefile_fullpath)
        data_cube = imgf.image()
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
        return data_cube

    def get_calibimage(self, calib_filename):
        """
        """
        spefile_fullpath = self.calib_path + '/' + calib_filename
        imgc = mf.ReadSPEFile(spefile_fullpath)
        calib_cube = imgc.image()
        cexptime = imgc.exptime()[1]
        imgc.close()
        return calib_cube, cexptime
    
    def get_position(self, position_filename):
        """
        """
        get_pos = mf.ReadPOSFile(self.calib_path + '/' + position_filename)
        position = get_pos.get_position()
        return position
        
    def get_assignment(self, assign_filename):
        """
        """
        get_ass = mf.ReadASSFile(self.calib_path + '/' + assign_filename)
        assignment = get_ass.get_armcoor()
        return assignment
        
    def bin_x_by_2(self, unbin_image):
        """Return a binned image cube
        """
        odd_image = unbin_image[:,:,0::2]
        eve_image = unbin_image[:,:,1::2]
        binned_image = odd_image + eve_image
        return binned_image

class IFSample(object):

    def __init__(self, ata_id_dir, spe_filename, bias_filename, flat_filename, posi_filename):
        test = Converter(ata_id_dir)
        raw = test.get_dataimage(spe_filename)
        raw = np.float32(raw)
        bias, bias_expt = test.get_calibimage(bias_filename)
        bias = np.float32(bias)
        flat, flat_expt = test.get_calibimage(flat_filename)
        flat = np.float32(flat)
        self.raw = test.bin_x_by_2(raw)
        bias = test.bin_x_by_2(bias)
        flat = test.bin_x_by_2(flat)
    
        mean_bias = np.mean(bias, axis=0)
        for index in range(flat.shape[0]):
            flat[index] = flat[index] - mean_bias
        self.sumflat = np.sum(flat, axis=0)
    
        self.position = test.get_position(posi_filename)
        
    def plot_IFS(self, fibre):
        x, y = self.position[fibre]
    
        fb_raw = self.raw[0,y:y+40,x:x+40]
        fb_flat = self.sumflat[y:y+40,x:x+40]
        fb_flat1d = np.ravel(fb_flat)
        fb_flat1ds = np.sort(fb_flat1d)
        turncated = fb_flat1ds[304]
        mask = fb_flat > turncated
    
        fb_masked = fb_raw*mask
    
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(fb_raw)
    
        plt.subplot(2,2,2)
        plt.imshow(fb_flat)
    
        plt.subplot(2,2,3)
        plt.imshow(mask)
    
        plt.subplot(2,2,4)
        plt.imshow(fb_masked)

def convert_seul(data_id_dir, spe_filename, bias_filename, flat_filename, assi_filename, posi_filename, output_dir='', flat_corr=False):
    """Usage: convert_seul('/Volumes/Data/20110324', '20110324_2_TNO02_01_10.SPE', '20110323_2_BIAS_01.SPE', '20110324_2_FLAT_02.SPE', '201103_TNO02.ASS', '201103_2.POS', '/Volumes/Data/test', flat_corr=False)
    """
    # read data image
    test = Converter(data_id_dir)
    
    # get raw
    raw = test.get_dataimage(spe_filename)
    raw = np.float32(raw)
    
    # get calibration
    bias, bias_expt = test.get_calibimage(bias_filename)
    bias = np.float32(bias)
    flat, flat_expt = test.get_calibimage(flat_filename)
    flat = np.float32(flat)
    
    # get assignment and position files
    assign = test.get_assignment(assi_filename)
    position = test.get_position(posi_filename)
    
    # bin x axis by 2
    bias = test.bin_x_by_2(bias)    # 768 X 240 to 384 X 240
    flat = test.bin_x_by_2(flat)    # 768 X 240 to 384 X 240
    mean_bias = np.mean(bias, axis=0)    
    
    # subtract bias from flat image
    for index in range(flat.shape[0]):
        flat[index] = flat[index] - mean_bias
    sumflat = np.sum(flat, axis=0)
    
    raw = test.bin_x_by_2(raw)    # 768 X 240 to 384 X 240
    # subtract bias from raw image
    for index in range(raw.shape[0]):
        raw[index] = raw[index] - mean_bias
    
    if len(output_dir) == 0:
        return raw
    else:
        # save to fits file
        save2fits(spe_filename, bias_filename, flat_filename, raw, sumflat, test, assign, position, output_dir, '1', flat_corr)
        del raw
                    
def convert_tout(data_id_dir, obs_id, bias_filename, flat_filename, assi_filename, posi_filename, fits_data_dir, flat_corr=False):
    """Convert batch SPE files into FITS format.
       Usage: convert_tout('/Volumes/Data/20110324', 'TNO02_01', '20110323_2_BIAS_01.SPE', '20110324_2_FLAT_02.SPE', '201103_TNO02.ASS', '201103_2.POS', '/Volumes/Data/test', flat_corr=False)
    """
    # call Converter class
    test = Converter(data_id_dir)
    
    # get calibration
    bias, bias_expt = test.get_calibimage(bias_filename)
    bias = np.float32(bias)
    flat, flat_expt = test.get_calibimage(flat_filename)
    flat = np.float32(flat)
    
    # get assignment and position files
    assign = test.get_assignment(assi_filename)
    position = test.get_position(posi_filename)
    
    # get raw data list
    data_file_list = make_spefile_list(data_id_dir, obs_id)
    number_of_file = str(len(data_file_list))
    
    # bin x axis by 2
    bias = test.bin_x_by_2(bias)    # 768 X 240 to 384 X 240
    flat = test.bin_x_by_2(flat)    # 768 X 240 to 384 X 240
    mean_bias = np.mean(bias, axis=0)
    
    # subtract bias from flat image
    for index in range(flat.shape[0]):
        flat[index] = flat[index] - mean_bias
    sumflat = np.sum(flat, axis=0)
    
    for spe_filename in data_file_list:
        print('Processing ' + spe_filename + '...')
        raw = test.get_dataimage(spe_filename)
        raw = np.float32(raw)
        raw = test.bin_x_by_2(raw)    # 768 X 240 to 384 X 240
        # subtrac bias from raw image
        for index in range(raw.shape[0]):
            raw[index] = raw[index] - mean_bias

        # save to fits file
        save2fits(spe_filename, bias_filename, flat_filename, raw, sumflat, test, assign, position, fits_data_dir, number_of_file, flat_corr)
        del raw
        
def save2fits(spe_filename, bias_filename, flat_filename, raw_image, flat_image, decl_class, assi_list, posi_list, output_dir, number_of_file, flat_corr):
    """Usage: save2fits(spe_filename, bias_filename, flat_filename, raw_image, flat_image, decl_class, assi_list, posi_list, output_dir, number_of_file)
    """
    # save to fits file
    prihdu = pf.PrimaryHDU(raw_image[0])    # save a snapshot
    # Important information
    # 1. Basic observation information
    prihdu.header.update('FILENAME', spe_filename[:-4], '')
    prihdu.header.update('TELESCOP', 'OHP 1.93m')
    prihdu.header.update('INSTRUME', 'MIOSOTYS')
#    prihdu.header.update('OBSERVER', pi_name, 'P.I.')
#    prihdu.header.update('OBJECT', target_id, 'Type of scientific objective')
    # 2. Date and Time
    prihdu.header.update(decl_class.date[0], decl_class.date[1], decl_class.date[2])
    prihdu.header.update(decl_class.lst[0], decl_class.lst[1], decl_class.lst[2])
    prihdu.header.update(decl_class.utc[0], decl_class.utc[1], )
    prihdu.header.update(decl_class.jd[0], decl_class.jd[1], decl_class.jd[2])
    # 3. Calibration
    prihdu.header.update('NUMSHOT', raw_image.shape[0], 'Number of shots')
#    prihdu.header.update('BINSIZE', binsize, 'Binning scale')
#    prihdu.header.update('BPIXELX', PixelOfImg, 'X axis size of fibre image')
#    prihdu.header.update('BPIXELY', PixelOfImg, 'Y axis size of fibre image')
    prihdu.header.update(decl_class.exptime[0], decl_class.exptime[1], decl_class.exptime[2])
    prihdu.header.update(decl_class.readout[0], decl_class.readout[1], decl_class.readout[2])
    prihdu.header.update(decl_class.temperature[0], decl_class.temperature[1], decl_class.temperature[2])
    # Th values below assumed readout rate at 10 MHz, with conversion gain at level 3.
    # See Princeton Instruments manual for more information
#    prihdu.header.update('READNOI', 39.52, 'readout noise (e- rms)')
#    prihdu.header.update('GAIN', 2.47, 'Conversion gain (e-/ADU)')
#    prihdu.header.update('DARK', 0.0074, 'e-/pixel/sec at -55 degree Celsius')
    prihdu.header.update(decl_class.emgain[0], decl_class.emgain[1], decl_class.emgain[2])
    prihdu.header.update('FIBREACT', len(assi_list.keys()), 'Number of fibres activated')
    ext_index = 1
    imagecontent = []
    for key, value in assi_list.iteritems():
        ra = value[0]
        dec = value[1]
        mag = value[2]
        x_start = posi_list[key][0]
        y_start = posi_list[key][1]
        prihdu.header.update('EXTEN_'+str(ext_index).zfill(2), str(key), 'fibre arm')
        prihdu.header.update('RA_'+str(ext_index).zfill(2), ra)
        prihdu.header.update('DEC_'+str(ext_index).zfill(2), dec)
        prihdu.header.update('MAG_'+str(ext_index).zfill(2), mag)
        # Correcting flatfield or just apply mask to raw image
        crop_flat = flat_image[y_start:y_start+40,x_start:x_start+40]
        crop_image = raw_image[:,y_start:y_start+40,x_start:x_start+40]
        if flat_corr == 0:
            mask = flat2mask(crop_flat)
            final_image = crop_image*mask
        if flat_corr == 1:
            mask = flat2mask(crop_flat)
            mask_flat = crop_flat*mask
            max_in_mask_flat = normalflat(mask_flat, 2.0)
            nflat = crop_flat/max_in_mask_flat
            final_image = (crop_image/nflat)*mask
        #
        imgtemp = pf.ImageHDU(final_image)
        imagecontent.append(imgtemp)
        info_a = '%2d  %3d  %3s  %3.2f  ' % (key, x_start, y_start, mag)
        print(info_a + ra + '  ' + dec)
        ext_index = ext_index + 1
        
    prihdu.header.update('EQUINOX', 2000.0)
    currenttime = time.strftime('%x %X %Z')
    prihdu.header.add_comment('Image status: Converted from WinView SPE to FITS by Shih I Chun')
    prihdu.header.add_comment('Image status: Using Bias file: ' + bias_filename)
    if flat_corr == 0:
        prihdu.header.add_comment('Image status: Using Flat file: None.')
    if flat_corr == 1:
        prihdu.header.add_comment('Image status: Using Flat file: ' + flat_filename)
    prihdu.header.add_comment('Image status: Generated on ' + currenttime)
    
    
    # Writing newly created fits into assigned directory
    hdulist = pf.HDUList([prihdu] + imagecontent)

    if op.exists(output_dir) == 1:
        pass
    else:
        os.mkdir(output_dir)

    fits_filename = spe_filename[:20] + str(spe_filename[20:-4]).zfill(len(number_of_file)) + '.fits'
    output_path = output_dir + '/' + fits_filename
    hdulist.writeto(output_path)

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
    
def make_spefile_list(data_id_dir, target_id):
    """make SPE file list"""
    spedata_dir = data_id_dir + '/raw/'
    data_id = data_id_dir[-8:]
    template = data_id + '*' + target_id + '*.SPE'
    
    file_match = []
    for spe_file in os.listdir(spedata_dir):
        if fnmatch.fnmatch(spe_file, template):
            file_match.append(spe_file)
    return file_match
