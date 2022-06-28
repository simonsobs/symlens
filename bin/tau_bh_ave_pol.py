from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from symlens import qe
from symlens.factorize import e
import symlens as s

import matplotlib.pyplot as plt

shape,wcs = maps.rect_geometry(width_deg=20.,px_res_arcmin=1.5,proj='plain')
theory = cosmology.default_theory()
modlmap = enmap.modlmap(shape,wcs)

#hardening = 'src'

noise_level = 1.

# Lmax = 2000
# cluster = False

Lmax = 5100
beam_arcmin = 1.0

delensing_amp_factor = 0.2 #assumed reduction factor for delensing, sometimes called A_L


cluster = False

if cluster:
    xmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=2000)
    ymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=6000)
    lxmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=2000)
    lymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=6000)
    estimator = 'hdv'
else:
    lmax = 5000
    lmin = 30
    xmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    ymask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    lxmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    lymask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    estimator = 'hu_ok'

    


bin_edges = np.arange(20,Lmax,20)
ells = np.arange(20,Lmax,1)
binner = stats.bin2D(modlmap,bin_edges)
clkk = theory.gCl('kk',ells)

# XYs = ['TT', 'EB']
XYs = ['TT', 'EB']

kmask = maps.mask_kspace(shape,wcs,lmin=20,lmax=Lmax)

feed_dict = {}
feed_dict['uC_T_T'] = theory.uCl('TT',modlmap)
feed_dict['tC_T_T'] = theory.lCl('TT',modlmap) + (noise_level*np.pi/180/60)**2./maps.gauss_beam(modlmap,beam_arcmin)**2.

feed_dict['uC_E_E'] = theory.uCl('EE',modlmap)
feed_dict['tC_E_E'] = theory.lCl('EE',modlmap) + 2 * (noise_level*np.pi/180/60)**2./maps.gauss_beam(modlmap,beam_arcmin)**2.


feed_dict['uC_B_B'] = theory.uCl('BB',modlmap) #this should just be zero
feed_dict['tC_B_B'] = theory.lCl('BB',modlmap) * delensing_amp_factor \
                      + 2 * (noise_level*np.pi/180/60)**2./maps.gauss_beam(modlmap,beam_arcmin)**2.

feed_dict['pc_T_T'] = 1


do_all = True
if do_all:
    A_L_phi_predef_1d = {}

    A_L_mask_predef_1d = {}

    A_L_tau_predef_1d = {}

    N_L_phi_1d  = {}
    N_L_phi_srcbh_1d = {}

    N_L_mask_1d  = {}
    N_L_tau_1d  = {}

    N_L_mask_phibh_1d = {}
    N_L_tau_phibh_1d = {}




    for XY in XYs:

        A_L_phi_predef = qe.A_l(shape,wcs,feed_dict,'hu_ok',XY,xmask=xmask,ymask=ymask)


        # A_L_mask_predef = qe.A_l(shape, wcs, feed_dict, 'mask', XY, xmask=xmask,ymask=ymask)

        A_L_tau_predef = qe.A_l(shape, wcs, feed_dict, 'tau', XY, xmask=xmask,ymask=ymask)

        N_l_phi = qe.N_l_optimal(shape,wcs,feed_dict,'hu_ok',XY,
                                 xmask=xmask,ymask=ymask,field_names=None,kmask=kmask)



        # h_phi_srcbh = qe.HardenedXY(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
        #                             Al=None,estimator=estimator,hardening=hardening, XY=XY)
        # N_l_phi_srcbh = h_phi_srcbh.get_Nl()


        # h_phi_maskbh = qe.HardenedXY(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
        #                              Al=None,estimator='hu_ok', hardening='mask', XY=XY)



        # N_l_mask = qe.N_l_optimal(shape,wcs,feed_dict,'mask','TT',xmask=xmask,ymask=ymask,
        #                           field_names=None,kmask=kmask)

        N_l_tau = qe.N_l_optimal(shape,wcs,feed_dict,'tau',XY,xmask=xmask,ymask=ymask,
                                 field_names=None,kmask=kmask)


        # h_mask_phibh = qe.HardenedXY(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
        #                              Al=None,estimator='mask', hardening='phi', target = 'mask',XY=XY)

        # import pdb as PDB
        # PDB.set_trace()
        h_tau_phibh = qe.HardenedXY(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
                                    Al=None,estimator='tau', hardening='phi', target = 'tau',XY=XY)



        # N_l_mask_phibh = h_mask_phibh.get_Nl()
        N_l_tau_phibh = h_tau_phibh.get_Nl()

        cents, A_L_phi_predef_1d[XY] = binner.bin(A_L_phi_predef)

        # cents, A_L_mask_predef_1d[XY] = binner.bin(A_L_mask_predef)

        cents, A_L_tau_predef_1d[XY] = binner.bin(A_L_tau_predef)


        cents, N_L_phi_1d[XY] = binner.bin(N_l_phi)
        # cents, N_L_phi_srcbh_1d[XY] = binner.bin(N_l_phi_srcbh)

        # cents, N_L_mask_1d[XY] = binner.bin(N_l_mask)
        cents, N_L_tau_1d[XY] = binner.bin(N_l_tau)

        # cents, N_L_mask_phibh_1d[XY] = binner.bin(N_l_mask_phibh)
        cents, N_L_tau_phibh_1d[XY] = binner.bin(N_l_tau_phibh)






for ii, XY in enumerate(XYs):
    plt.figure(figsize=(5,5))

    prefac = 1/cents**2
    plt.semilogy(cents,prefac * A_L_tau_predef_1d[XY] , label = 'A_L_tau_predef_1d', color = 'b', ls = 'dashed')


    plt.semilogy(cents,prefac * N_L_tau_1d[XY] * 4 / cents**2, label = 'N_L_tau_1d' , color = 'b', ls = 'dashdot')
    plt.semilogy(cents,prefac * N_L_tau_phibh_1d[XY] * 4 / cents**2, label = 'N_L_tau_phibh_1d' , color = 'b', ls = 'dotted')

    plt.legend(loc = 'best')
    # pl.hline(y=1)
    plt.ylim([1e-14,1e0])
    plt.xlim([0,5e3])
    plt.savefig('../output/A_L_tau_%s_lmax%i.png' % (XY, lmax))

    np.savetxt('../output/nl_tau_%s_lmax%i.txt' % (XY, lmax), np.transpose([cents, N_L_tau_1d[XY], N_L_tau_phibh_1d[XY]]), 
               header = " ell, N_L_tau, N_L_tau_bh")
stop

