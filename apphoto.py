#!/usr/bin/env python
# Aperture Photometry for MIOSOTYS FITS image data
# Last update: 18/01/2012

import pyfits as pf
import mefos as mf    # aperture photometry scripts is here.
import numpy as np
import os
import os.path as op
import fnmatch as fm

def Lightcurve_seul(image_dir, fits_filename, output_dir, aperture=8, inner_annulus=13, outer_annulus=0, centroid='c'):
    """
    aperture photometry for all the firbres in the FITS images
    The results are saved in the FITS table format. 
    """
    # getting numbers of active fibres and other header information
    temp = pf.open(image_dir + '/' + fits_filename)
    priheader = temp[0].header
    num_of_fibre = priheader['FIBREACT']
    exposure_time = priheader['EXPTIME']
    tzero = 0.0
    snapshot = temp[0].data
    fstheader = temp[1].header
    length_of_cube = fstheader['NAXIS3']

    cols_ext = []
    for index in range(num_of_fibre):
        ext_image = temp[index+1].data
        temp_lcs = []
        for numimage in range(length_of_cube):
            temp_image = ext_image[numimage]
    
            photo_temp = mf.APhot(temp_image)
            # Image centring
            if centroid == 'g':
                best_gfit = photo_temp.fitguassian()
                centre = photo_temp.gauss_centre(best_gfit)
            if centroid == 'c':
                centre = photo_temp.centroid()       
            # Measurement of source
            apt = photo_temp.aptvalue(centre, aperture)
            # Measurement of background
            if (outer_annulus <= inner_annulus):
                sky = photo_temp.skyvalue2(centre, inner_annulus)
            else:
                sky = photo_temp.skyvalue(centre, inner_annulus, outer_annulus)
            # measuring source intensity
            intensity = apt[1] - apt[0]*sky[2]
            time_stamp = tzero + numimage*exposure_time
            temp_lcs.append([time_stamp, intensity, centre[0], centre[1], apt[0], sky[2]])
        temp_lcs = np.asarray(temp_lcs)
            
        # Creating FITS table
        time = np.array(temp_lcs[:,0])
        src_flux = np.array(temp_lcs[:,1])
        x_centre = np.array(temp_lcs[:,2])
        y_centre = np.array(temp_lcs[:,3])
        src_pixl = np.array(temp_lcs[:,4])
        avg_skyv = np.array(temp_lcs[:,5])
                        
        col1 = pf.Column(name='Time (sec)', format='E', array=time)
        col2 = pf.Column(name='Src_Flux (count)', format='E', array=src_flux)
        col3 = pf.Column(name='X_Centre (pixel)', format='E', array=x_centre)
        col4 = pf.Column(name='Y_Centre (pixel)', format='E', array=y_centre)
        col5 = pf.Column(name='Src_Area (pixel)', format='E', array=src_pixl)
        col6 = pf.Column(name='Median_Sky (count)', format='E', array=avg_skyv)
            
        cols = pf.ColDefs([col1, col2, col3, col4, col5, col6])
        tbhdu = pf.new_table(cols)
            
        cols_ext.append(tbhdu)
        
    hdu = pf.PrimaryHDU(snapshot, priheader)
    thdulist = pf.HDUList([hdu] + cols_ext)
    
    # Update extension header    
    for itable in range(num_of_fibre):
        ext_table = thdulist[itable+1]
        ext_table.header.update('FIBRE', priheader['EXTEN_'+str(itable+1).zfill(2)])
        ext_table.header.update('RA', priheader['RA_'+str(itable+1).zfill(2)])
        ext_table.header.update('DEC', priheader['DEC_'+str(itable+1).zfill(2)])
        ext_table.header.update('MAG', priheader['MAG_'+str(itable+1).zfill(2)])
        ext_table.header.update('EQUINOX', priheader['EQUINOX'])
        ext_table.header.update('APERTURE', aperture, 'aperture radius in pixels')
        ext_table.header.update('SKYRADIN', inner_annulus, 'inner sky radius in pixels')
        if (outer_annulus > inner_annulus):
            ext_table.header.update('SKYRADOU', outer_annulus, 'outer sky radius in pixels')
    
    thdulist.writeto(output_dir + '/' + fits_filename, output_verify='fix')    # pyfits 3.0.3 changes the behaviour of verify (default is 'exception'), but is reversed to fix in 3.0.4
    
    temp.close()

def Lightcurve_tout(main_dir, target_id, aperture=8, inner_annulus=13, outer_annulus=0, centroid='c', save_dir='reduced'):
    """
    creating lightcurves en mass
    """
    image_dir = main_dir + '/' + save_dir + '/' + target_id + '_images'
    lc_dir = main_dir + '/' + save_dir + '/' + target_id + '_ap' + str(aperture).zfill(2) + '_lcs'

    if op.exists(lc_dir) == 1:
        pass
    else:
        os.mkdir(lc_dir)
    
    # creating an fits image list
    for fits_filename in os.listdir(image_dir):
        if fm.fnmatch(fits_filename, '*.fits'):
            Lightcurve_seul(image_dir, fits_filename, lc_dir, aperture, inner_annulus, outer_annulus, centroid)