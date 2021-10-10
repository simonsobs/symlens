from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from symlens import qe
from symlens.factorize import e

shape,wcs = maps.rect_geometry(width_deg=20.,px_res_arcmin=1.5,proj='plain')
theory = cosmology.default_theory()
modlmap = enmap.modlmap(shape,wcs)

hardening = 'src'

noise_level = 10.

# Lmax = 2000
# cluster = False

Lmax = 6000
cluster = False

if cluster:
    xmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=2000)
    ymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=6000)
    lxmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=2000)
    lymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=6000)
    estimator = 'hdv'
else:
    xmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=3500)
    ymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=3500)
    lxmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=3500)
    lymask = maps.mask_kspace(shape,wcs,lmin=100,lmax=3500)
    estimator = 'hu_ok'

kmask = maps.mask_kspace(shape,wcs,lmin=20,lmax=Lmax)

feed_dict = {}
feed_dict['uC_T_T'] = theory.uCl('TT',modlmap)
feed_dict['tC_T_T'] = theory.lCl('TT',modlmap) + (noise_level*np.pi/180/60)**2./maps.gauss_beam(modlmap,1.4)**2.
feed_dict['pc_T_T'] = 1

N_l = qe.N_l_optimal(shape,wcs,feed_dict,'hu_ok','TT',xmask=lxmask,ymask=lymask,field_names=None,kmask=kmask)
h = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,Al=None,estimator=estimator,hardening=hardening)
N_l_bh = h.get_Nl()
    
bin_edges = np.arange(20,Lmax,20)
ells = np.arange(20,Lmax,1)
binner = stats.bin2D(modlmap,bin_edges)
clkk = theory.gCl('kk',ells)

#print(binner.bin(anull)[1])

cents,nl1d = binner.bin(N_l)
cents,nlbh1d = binner.bin(N_l_bh)
pl = io.Plotter('CL',xyscale='loglog')
pl.add(ells,clkk,color='k')
pl.add(cents,nl1d,ls='--')
pl.add(cents,nlbh1d,ls=':')
pl._ax.set_ylim(1e-8,5e-4)
pl.done('../output/bh.png')


pl = io.Plotter('rCL',xyscale='linlin')
pl.add(cents,nlbh1d/nl1d)
pl.hline(y=1)
pl.done('../output/bhdiff.png')

print('hi')
cents, ALsrc1d = binner.bin(h.fdict['Asrc_src_L'] )
pl = io.Plotter('CL',xyscale='linlog')
pl.add(cents,ALsrc1d / cents**2)
# pl.hline(y=1)
pl.done('../output/A_L_src.png')

