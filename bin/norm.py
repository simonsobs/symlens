"""
Script to look at how the lensing response varies with power in the data maps
"""

from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import symlens as s


# Make a flat sky geometry
width_deg = 20.0
px_res_arcmin = 2.0
shape,wcs = maps.rect_geometry(width_deg=width_deg,px_res_arcmin=px_res_arcmin,proj='plain')
modlmap = enmap.modlmap(shape,wcs)

# Make the CMB and kappa Fourier masks
lmin = 600
lmax = 3000
kmin = modlmap[modlmap>0].min()
kmax = 1000
tmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
kmask = maps.mask_kspace(shape,wcs,lmin=kmin,lmax=kmax)

# Response part of estimator
fresp = s.Ldl1 * s.e('rC_T_T_l1') + s.Ldl2 * s.e('rC_T_T_l2')
# Filter part of estimator
ffilt = s.Ldl1 * s.e('uC_T_T_l1') + s.Ldl2 * s.e('uC_T_T_l2')
F = ffilt / 2 / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')

theory = cosmology.default_theory()

def get_spec_1d(exp,lmax=None):
    ls,cltt = np.loadtxt(f'{exp}_cls.txt',usecols=[0,1],unpack=True)
    cltt[ls<2] = 0
    if not(lmax is None):
        cltt = cltt[ls<lmax]
        ls = ls[ls<lmax]
    return ls,cltt

def get_spec(exp):
    ls,cltt = get_spec_1d(exp)
    return maps.interp(ls,cltt)(modlmap)

ls,cl_d = get_spec_1d('default',3000)
ls,cl_p = get_spec_1d('planck',3000)
ls,cl_a = get_spec_1d('act',3000)

pl = io.Plotter('rCl')
pl.add(ls,(cl_p-cl_d)/cl_d,label='Planck 2018 TT vs. default')
pl.add(ls,(cl_a-cl_d)/cl_d,label='ACT DR4+WMAP TT vs. default')
pl.hline(y=0)
pl._ax.set_ylim(-0.01,0.035)
pl.done('clttdiff.png')
sys.exit()



def Nl(exp=None):

    # ACTish noise
    noise = 12.0
    beam = 1.4
    
    feed_dict = {}

    # Default to using same cosmology in response
    if exp is None:
        feed_dict['rC_T_T'] = get_spec('default')
    else:
        # or use TT Cls for response from a saved file
        feed_dict['rC_T_T'] = get_spec(exp)
    feed_dict['uC_T_T'] = get_spec('default')
    feed_dict['tC_T_T'] = get_spec('default') + (noise*(np.pi/180./60.) / maps.gauss_beam(modlmap,beam))**2.

    # A_L
    Al2d = s.A_l_custom(shape,wcs,feed_dict,fresp,F,xmask=tmask,ymask=tmask,groups=None,kmask=kmask)
    # N_L for easy comparison with C_L
    Nl2d = s.N_l_from_A_l_optimal(shape,wcs,Al2d)

    # Bin it
    bin_edges = np.linspace(kmin,kmax,30)
    binner = stats.bin2D(modlmap,bin_edges)
    cents,N1d = binner.bin(Nl2d)
    return cents,N1d

# Get the various N_L curves assuming different cosmologies in the response
cents,N1d = Nl()
cents,N1d_p = Nl(exp='planck')
cents,N1d_a = Nl(exp='act')


# Plot
ls = np.arange(40,kmax)
pl = io.Plotter('CL',xyscale='loglog')
pl.add(ls,theory.gCl('kk',ls))
pl.add(cents,N1d,ls='--')
pl.add(cents,N1d_p,ls=':')
pl.add(cents,N1d_a,ls='-')
pl.done('c_L_n_L.png')

pl = io.Plotter('rCL',xyscale='loglin')
pl.add(cents,2.*(N1d_p-N1d)/N1d,ls='--',label='Planck 2018 vs. default')
pl.add(cents,2.*(N1d_a-N1d)/N1d,ls='--',label='ACT DR4+WMAP vs. default')
pl.hline(y=0)
pl.done('Cl_diff.png',dpi=250)
