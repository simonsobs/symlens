from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from symlens import qe
from symlens.factorize import e
import symlens as s

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


bin_edges = np.arange(20,Lmax,20)
ells = np.arange(20,Lmax,1)
binner = stats.bin2D(modlmap,bin_edges)
clkk = theory.gCl('kk',ells)


kmask = maps.mask_kspace(shape,wcs,lmin=20,lmax=Lmax)

f_phi = s.Ldl1 * s.e('uC_T_T_l1') + s.Ldl2 * s.e('uC_T_T_l2')
F_phi = f_phi / 2 / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')

expr1_phi = f_phi * F_phi

f_tau = s.e('uC_T_T_l1') +  s.e('uC_T_T_l2')
F_tau = f_tau / 2 / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')

expr1_tau = f_tau * F_tau

feed_dict = {}
feed_dict['uC_T_T'] = theory.uCl('TT',modlmap)
feed_dict['tC_T_T'] = theory.lCl('TT',modlmap) + (noise_level*np.pi/180/60)**2./maps.gauss_beam(modlmap,1.4)**2.
feed_dict['pc_T_T'] = 1


# res_phi = s.integrate(shape,wcs,feed_dict,expr1_phi,xmask=xmask,ymask=xmask).real

# res_tau = s.integrate(shape,wcs,feed_dict,expr1_tau,xmask=xmask,ymask=xmask).real

A_L_phi = qe.A_l_custom(shape, wcs, feed_dict, f_phi, F_phi, xmask=xmask, ymask=ymask)
print( feed_dict.keys())
A_L_phi_predef = qe.A_l(shape,wcs,feed_dict,'hu_ok','TT',xmask=xmask,ymask=ymask)
print(feed_dict.keys())
A_L_tau = qe.A_l_custom(shape, wcs, feed_dict, f_tau, F_tau, xmask=xmask, ymask=ymask)
print(feed_dict.keys())
A_L_mask_predef = qe.A_l(shape, wcs, feed_dict, 'mask', 'TT', xmask=xmask,ymask=ymask)
print( feed_dict.keys())
# my_qe_phi = qe.QE(shape, wcs, feed_dict, 'hu_ok', XY = 'TT', 
#                   f = f_phi, F = F_phi, xmask = xmask, ymask = ymask)


N_l_phi = qe.N_l_optimal(shape,wcs,feed_dict,'hu_ok','TT',xmask=lxmask,ymask=lymask,field_names=None,kmask=kmask)
print (feed_dict.keys())


h_phi_srcbh = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
                            Al=None,estimator=estimator,hardening=hardening)
N_l_phi_srcbh = h_phi_srcbh.get_Nl()
print (feed_dict.keys())

h_phi_taubh = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
                            Al=None,estimator='hu_ok', hardening='mask')


# N_l = qe.N_l_optimal(shape,wcs,feed_dict,'hu_ok','TT',xmask=lxmask,ymask=lymask,field_names=None,kmask=kmask)
N_l_mask = qe.N_l_optimal(shape,wcs,feed_dict,'mask','TT',xmask=lxmask,ymask=lymask,field_names=None,kmask=kmask)

print('check, ', feed_dict.keys())


h_mask_phibh = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
                            Al=None,estimator='mask', hardening='phi', target = 'mask')

N_l_mask_phibh = h_mask_phibh.get_Nl()





# h_phi_srcbh = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,
#                             Al=None,estimator=estimator,hardening=hardening)


# h_tau_phibh = qe.HardenedTT

#print(binner.bin(anull)[1])


cents, A_L_phi_1d = binner.bin(A_L_phi )
cents, A_L_phi_predef_1d = binner.bin(A_L_phi_predef)

cents, A_L_tau_1d = binner.bin(A_L_tau)
cents, A_L_mask_predef_1d = binner.bin(A_L_mask_predef)

cents, N_L_phi_1d = binner.bin(N_l_phi)
cents, N_L_phi_srcbh_1d = binner.bin(N_l_phi_srcbh)

cents, N_L_mask_1d = binner.bin(N_l_mask)
cents, N_L_mask_phibh_1d = binner.bin(N_l_mask_phibh)

pl = io.Plotter('CL',xyscale='loglog')
pl.add(cents,A_L_phi_1d * cents**2)
pl.add(cents,A_L_phi_predef_1d * cents**2, ls='--')
# pl.hline(y=1)
pl.done('../output/A_L_phi.png')



pl = io.Plotter('L',xyscale='loglog')
# pl.add(cents,A_L_tau_1d , label = 'A_L_tau_1d')
# pl.add(cents,A_L_mask_predef_1d, label = 'A_L_mask_predef_1d')
pl.add(cents,N_L_mask_1d * 4 / cents**2, label = 'N_L_mask_1d')
pl.add(cents,N_L_mask_phibh_1d * 4 / cents**2, label = 'N_L_mask_phibh_1d', ls = '--')


pl.legend(loc = 'best')
# pl.hline(y=1)
pl.done('../output/A_L_tau.png')




pl = io.Plotter('CL',xyscale='loglog')
pl.add(ells,clkk,color='k')
pl.add(cents,N_L_phi_1d,ls='--')
pl.add(cents,N_L_phi_srcbh_1d,ls=':')
pl._ax.set_ylim(1e-8,5e-4)
pl.done('../output/phi_bh.png')


# pl = io.Plotter('rCL',xyscale='liN_Lin')
# pl.add(cents,N_Lbh1d/N_L1d)
# pl.hline(y=1)
# pl.done('../output/bhdiff.png')


cents, ALsrc1d = binner.bin(h_phi_srcbh.fdict['Asrc_src_L'] )
pl = io.Plotter('CL',xyscale='linlog')
pl.add(cents,ALsrc1d / cents**2)
# pl.hline(y=1)
pl.done('../output/A_L_src.png')


