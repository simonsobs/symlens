from __future__ import print_function
from orphics import maps,cosmology
from pixell import enmap
import numpy as np
import os,sys



from symlens import qe
from enlib import bench

# Say we want analytic RDN0 for the TTTE estimator
XY='TE'
UV='TE'

# example geometry, you can use your own map's geometry
shape,wcs = maps.rect_geometry(width_deg=25.,px_res_arcmin=2.0)
modlmap = enmap.modlmap(shape,wcs)

# symlens QEs always need you to specify 2d Fourier space masks
# for the CMB, and also for the final lensing k-mask
# For the CMB I use our typical ranges
ellmin = 500 ; ellmax = 3000
# and create a 2d mask (you can pass lxcut, lycut etc. if you want
# or use the 2d masks you already have for your analysis)
cmb_kmask = maps.mask_kspace(shape,wcs,lmin=ellmin,lmax=ellmax)
# Similarly, I create a final lensing k-mask which can go to lower L
Lmin = 100 ; Lmax = 3000
lens_kmask = maps.mask_kspace(shape,wcs,lmin=Lmin,lmax=Lmax)

# You always need a feed_dict to specify various 2d power spectra
feed_dict = {}
# We need some theory spectra. You'll have your own way to get them,
# but I'll load them with orphics
theory = cosmology.default_theory()

# The first set of 2d spectra you need are the "uC" spectra. In the original
# papers, these were unlensed spectra, but now we know to use the lensed
# (or better yet for TT, the TgradT) spectra. These are the CMB signal
# only spectra that appear in numerators of the norm/response and filters
feed_dict['uC_T_T'] = theory.lCl('TT',modlmap) # interpolate on to modlmap
feed_dict['uC_T_E'] = theory.lCl('TE',modlmap)

# The next set are the "tC" spectra. These are the total theory spectra
# that appear in the denominators of (diagonal) filters. In this example,
# I am lazily leaving these as the lensed spectra, BUT PLEASE REPLACE
# THESE with the theory total power spectra you use in filters!
feed_dict['tC_T_T'] = theory.lCl('TT',modlmap)
feed_dict['tC_T_E'] = theory.lCl('TE',modlmap)
feed_dict['tC_E_E'] = theory.lCl('EE',modlmap)

# The next set are the "dC" spectra. These are the realized data spectra
# (e.g of the current sim, if using these in an ensemble of sims). In this example,
# I am lazily leaving these as the lensed spectra, BUT PLEASE REPLACE
# THESE with the realized data spectra you calculate!
feed_dict['dC_T_T'] = theory.lCl('TT',modlmap)
feed_dict['dC_T_E'] = theory.lCl('TE',modlmap)
feed_dict['dC_E_E'] = theory.lCl('EE',modlmap)

# The next set are the "nC" spectra. These are the analytic/expected power
# spectra. Typically, these are identical to "tC", but in some cases you
# might want to filter sub-optimally. In this example,
# I am lazily leaving these as the lensed spectra, BUT PLEASE REPLACE
# THESE with the expected theory total power spectrum!
feed_dict['nC_T_T'] = theory.lCl('TT',modlmap)
feed_dict['nC_T_E'] = theory.lCl('TE',modlmap)
feed_dict['nC_E_E'] = theory.lCl('EE',modlmap)

# Here's how to get A_L. It might be different from other codes by L^2 and/or 4 factors!
with bench.show("Al"):
    AlXY = qe.A_l(shape,wcs,feed_dict,'hu_ok',XY,xmask=cmb_kmask,ymask=cmb_kmask,kmask=lens_kmask)
with bench.show("Al"):
    AlUV = qe.A_l(shape,wcs,feed_dict,'hu_ok',UV,xmask=cmb_kmask,ymask=cmb_kmask,kmask=lens_kmask)


# Now we do a call to get the 2d analytic RDN0
with bench.show("rdn0"):
    rdn0_2d = qe.RDN0_analytic(shape,wcs,feed_dict,'hu_ok',XY,'hu_ok',UV,
                               xmask=cmb_kmask,ymask=cmb_kmask,kmask=lens_kmask,Aalpha=AlXY,Abeta=AlUV)
    
# I have speed this up by providing pre-calculated normalizations in AlXY
# and AlUV arguments, but (note to Alex) make sure you calculate that using symlens itself
# to avoid problems due to differences in conventions between symlens
# and Falcon.

# Note on w-factors: if the data spectra have been corrected by a w2 factor
# the final rdn0 you get should be correct (and directly comparable to 
# a standard N0 curve).

