#!/usr/bin/env python
# MIOSOTYS Data Reduction Processes (MioDataRedProc)
# This is an alternative method to deal with abnormal bias level currently generated by the ProEM camera.
#
# Created by I Chun Shih @ Observatoire de Paris, 2011

import time
import os
import math
import fnmatch
import os.path as op
import mefos as mf
import numpy as np
import pyfits as pf
import matplotlib.pylab as plt
from scipy.ndimage import morphology

version = 2.0

class MioDataRedProc(object):

    def __init__(self, imagefile):
        imgf = mf.ReadSPEFile(imagefile)
        self.image_cube = imgf.image()
        self.date = imgf.iso8601date()
        self.lst = imgf.obslst()
        self.utc = imgf.obsutc()
        self.jd = imgf.tojd()
        self.exptime = imgf.exptime()
        self.readout = imgf.readtime()
        self.gain = imgf.gain()
        self.emgain = imgf.avalancegain()
        self.tempc = imgf.temperature()

        imgf.close()
    
    def FibreImage(self, x, y):
        image_frame = self.image_cube[:, y:y+40, x:x+40]
        return image_frame
        
    def BR6Image(self):
        """Under current configuration, there are 6 regions of ccd pixels which are not exposed to light source. They are used to determine the instant bias level.
           At this version, we always assume that the readout image size is 240X240 pixels. 
        """
        br1eme = self.image_cube[:,0:40,0:40]
        br2eme = self.image_cube[:,0:40,160:200]
        br3eme = self.image_cube[:,0:40,200:]
        br4eme = self.image_cube[:,200:,0:40]
        br5eme = self.image_cube[:,200:,40:80]
        br6eme = self.image_cube[:,200:,200:]

        contain_br6 = []
        for index in range(self.image_cube.shape[0]):
            avg_br6 = np.mean([br1eme[index],br2eme[index],br3eme[index],br4eme[index],br5eme[index],br6eme[index]],axis=0)
            contain_br6.append(avg_br6)
        return np.asarray(contain_br6)

class Converter(object):
    """ Alternative SPE to FITS converter
    Using this program to convert data obtained from all missions in 2010
    """

    def __init__(self, main_dir):
        self.main_dir = main_dir
        
    def make_spefile_list(self, target_id):
        """make SPE file list"""
        spedata_dir = self.main_dir + '/raw/'
        data_id = self.main_dir[-8:]
        template = data_id + '*' + target_id + '*.SPE'
    
        file_match = []
        for spe_file in os.listdir(spedata_dir):
            if fnmatch.fnmatch(spe_file, template):
                file_match.append(spe_file)
        return file_match
        
    def get_correct_flat_and_mask(self, flat_filename, ass_filename, pos_filename):
        """usage: get_correct_flat_and_mask(flat_filename, ass_filename, pos_filename)
        """
        calib_dir = self.main_dir + '/calib/'
    
        # Reading flat, assign, and position files
        image_flat = MioDataRedProc(calib_dir + flat_filename)
        print('Reading ' + flat_filename + ' completed!')
    
        assf = mf.ReadASSFile(calib_dir + ass_filename)
        self.arm_assign_list = assf.get_armcoor()
    
        posf = mf.ReadPOSFile(calib_dir + pos_filename)
        self.fibre_pos_list = posf.get_position()
        
        # creating two new dictionary with the same keys of "arm_assign_list"                   
        self.active_fibre = self.arm_assign_list.keys()
        fibre_mask = self.arm_assign_list.fromkeys(self.active_fibre)    # fibre mask
        correct_flat = self.arm_assign_list.fromkeys(self.active_fibre)

        # calling Bias Rigions 6 images. Just need to call once from the same flat or data images.
        print('and obtaining flat bias corners')
        br6_flat = image_flat.BR6Image()
    
        # correcting flat images and create an averaged flat template
        for fibre_number in self.active_fibre:
            x, y = self.fibre_pos_list[fibre_number]    # x, y positions of fibre image
            # subtracting instead bias level to obtain a corrected flat iamge
            correct_flatimage = image_flat.FibreImage(x,y) - br6_flat
            average_flat = np.mean(correct_flatimage, axis=0)    # creating an averaged flat
            # normalising flat image
            correct_flat[fibre_number] = average_flat / np.amax(average_flat)
            # creating a fibre mask to remove the non_data edge pixel 
            mask_erosion = morphology.grey_erosion(average_flat, size=(1,1))
            fibre_mask[fibre_number] = mask_erosion > np.median(mask_erosion) - 1.5*np.std(mask_erosion)
        return correct_flat, fibre_mask
    
    def get_correct_data(self, data_filename, correct_flat, fibre_mask):
        """usage: get_correct_data(data_spefile, correct_flat*, fibre_mask*)
        * these are generated by get_correct_flat_and_mask()
        """
        image_dir = self.main_dir + '/raw/'
    
        # Reading image
        image_data = MioDataRedProc(image_dir + data_filename) 
        print('Reading ' + data_filename + ' completed!')

        print('and obtaining data bias corners')
        br6_data = image_data.BR6Image()

        correct_data = self.arm_assign_list.fromkeys(self.active_fibre)
    
        # correcting data image
        # subtracting instant bias level in the data image
        for fibre_number in self.active_fibre:
            print('Processing fibre %3d...' % (fibre_number))
            x, y = self.fibre_pos_list[fibre_number]    # x, y positions of fibre image
            correct_dataimage = image_data.FibreImage(x,y) - br6_data
            corrected = (correct_dataimage/correct_flat[fibre_number])*fibre_mask[fibre_number]
            correct_data[fibre_number] = np.float32(corrected)
        
        self.data_filename = data_filename
        self.image_data = image_data
        return correct_data
    
    def save_to_fits(self, correct_data, target_id, pi_name, fits_filename, output_dir):
        """usage: save_to_fits(correct_data*, target_id, pi_name, fits_filename, output_dir)
        * this is generated by get_correct_data()
        example:
        save_to_fits(correct_data*, 'TNO01_01', 'IC Shih', fits_filename, 'reduced/')
        """
        hdinfo = self.image_data
        
        prihdu = pf.PrimaryHDU(self.image_data.image_cube[0])    # snapshot
        # Important information
        # 1. Basic observation information
        prihdu.header.update('FILENAME', self.data_filename[:-4], '')
        prihdu.header.update('TELESCOP', 'OHP 1.93m')
        prihdu.header.update('INSTRUME', 'MIOSOTYS')
        prihdu.header.update('OBSERVER', pi_name, 'P.I.')
        prihdu.header.update('OBJECT', target_id, 'Type of scientific objective')
        # 2. Date and Time
        prihdu.header.update(hdinfo.date[0], hdinfo.date[1], hdinfo.date[2])
        prihdu.header.update(hdinfo.lst[0], hdinfo.lst[1], hdinfo.lst[2])
        prihdu.header.update(hdinfo.utc[0], hdinfo.utc[1], )
        prihdu.header.update(hdinfo.jd[0], hdinfo.jd[1], hdinfo.jd[2])
        # 3. Calibration
        prihdu.header.update('NUMSHOT', hdinfo.image_cube.shape[0], 'Number of shots')
        #prihdu.header.update('BINSIZE', binsize, 'Binning scale')
        #prihdu.header.update('BPIXELX', PixelOfImg, 'X axis size of fibre image')
        #prihdu.header.update('BPIXELY', PixelOfImg, 'Y axis size of fibre image')
        prihdu.header.update(hdinfo.exptime[0], hdinfo.exptime[1], hdinfo.exptime[2])
        prihdu.header.update(hdinfo.readout[0], hdinfo.readout[1], hdinfo.readout[2])
        prihdu.header.update(hdinfo.tempc[0], hdinfo.tempc[1], hdinfo.tempc[2])
        # Th values below assumed readout rate at 10 MHz, with conversion gain at level 3.
        # See Princeton Instruments manual for more information
        prihdu.header.update('READNOI', 39.52, 'readout noise (e- rms)')
        prihdu.header.update('GAIN', 2.47, 'Conversion gain (e-/ADU)')
        prihdu.header.update('DARK', 0.0074, 'e-/pixel/sec at -55 degree Celsius')
        prihdu.header.update(hdinfo.emgain[0], hdinfo.emgain[1], hdinfo.emgain[2])
        prihdu.header.update('FIBREACT', len(correct_data.keys()), 'Number of active fibres')
        
        ext_index = 1
        imagecontent = []
        for key, values in self.arm_assign_list.iteritems():
            prihdu.header.update('EXTEN_'+str(ext_index).zfill(2), str(key), 'fibre arm')
            prihdu.header.update('RA_'+str(ext_index).zfill(2), values[0])
            prihdu.header.update('DEC_'+str(ext_index).zfill(2), values[1])
            prihdu.header.update('MAG_'+str(ext_index).zfill(2), values[2])
            imgtemp = pf.ImageHDU(correct_data[key])
            imagecontent.append(imgtemp)
            ext_index = ext_index + 1
        
        prihdu.header.update('EQUINOX', 2000.0)
        # 5. Comments
        currenttime = time.strftime('%x %X %Z')
        prihdu.header.add_comment('Image status: Converted from WinView SPE to FITS by Shih I Chun')
        prihdu.header.add_comment('Image status: Converter version: %2.1f' % (version))
        prihdu.header.add_comment('Image status: Generated on ' + currenttime)
        
        # Writing newly created fits into assigned directory
        hdulist = pf.HDUList([prihdu] + imagecontent)
        hdulist.writeto(output_dir + fits_filename)

def seul_convert(main_dir, spe_filename, flat_filename, ass_filename, pos_filename, save_dir='seul'):
    """ Converting single SPE image
    usage: seul_converter(main_dir, spe_filename, flat_filename, ass_filename, pos_filename, save_dir='seul')
    """
    saved_dir = main_dir + '/' + save_dir + '/'
    if op.exists(saved_dir) == 1:
        pass
    else:
        os.mkdir(saved_dir)
    
    temp = Converter(main_dir)
    correct_flat, fibre_mask = temp.get_correct_flat_and_mask(flat_filename, ass_filename, pos_filename)
    final_data = temp.get_correct_data(spe_filename, correct_flat, fibre_mask)
    temp.save_to_fits(final_data, 'n/a', 'n/a', spe_filename[:-3]+'fits', saved_dir)
    
def batch_convert(main_dir, flat_filename, target_id, pi_name, save_dir='reduced'):
    """ Converting all SPE images in the same target_id
    usage: batch_convert(main_dir, flat_filename, target_id, pi_name, save_dir='reduced')
    """
    temp = Converter(main_dir)
    
    data_file_list = temp.make_spefile_list(target_id)
    number_of_file = str(len(data_file_list))

    ass_filename = main_dir[-8:-2] + '_' + target_id[:-3] + '.ASS'
    pos_filename = main_dir[-8:-2] + '_2.POS'
    
    # creating directories
    saved_dir = main_dir + '/' + save_dir
    if op.exists(saved_dir) == 1:
        pass
    else:
        os.mkdir(saved_dir)
    
    image_dir = saved_dir + '/' + target_id + '_images/'
    if op.exists(image_dir) == 1:
        pass
    else:
        os.mkdir(image_dir)
        
    correct_flat, fibre_mask = temp.get_correct_flat_and_mask(flat_filename, ass_filename, pos_filename)
                
    for spe_filename in data_file_list:
        final_data = temp.get_correct_data(spe_filename, correct_flat, fibre_mask)
        fits_filename = spe_filename[:20] + str(spe_filename[20:-4]).zfill(len(number_of_file)) + '.fits'
        temp.save_to_fits(final_data, target_id, pi_name, fits_filename, image_dir)

def preview_speimage(spe_filename):
    """Preview SPE image
    """
    temp = mf.ReadSPEFile(spe_filename)

    Exptime = temp.exptime()[1]
    EMgain = temp.avalancegain()[1]
    
    temp_im = temp.avg_red_image()
    
    plt.clf()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,1,1)
    vmax = np.amax(temp_im)
    ax1.matshow(temp_im, vmin=0, vmax=vmax, picker=True)
    ax1.set_title('ExpTime: %6.2f, EM Gain: %6d\n' % (Exptime, EMgain))
    fig.canvas.mpl_connect('pick_event', on_press)
    plt.show()
    
    temp.close()
     
def plot_flat_and_mask(correct_flat, fibre_mask, fibre_number):
    """ Plot corrected average flatfield image and its mask.
    (Only for internal check usage!)
    """
    plt.clf()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(correct_flat[fibre_number], picker=True)
    ax1.set_title('Corrected avg. flatfield image from fibre %2d' % (fibre_number))
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(fibre_mask[fibre_number], picker=True)
    ax2.set_title('Mask image from the same fibre')

    fig.canvas.mpl_connect('pick_event', on_press)
     
    plt.show()
    
def on_press(event):
    mouseevent = event.mouseevent
    artist = event.artist
    data = artist.get_array()
    xind = math.trunc(mouseevent.xdata)
    yind = math.trunc(mouseevent.ydata)
    print 'X: %2d, Y: %2d, Height: %8.2f' % (xind, yind, data[yind,xind])