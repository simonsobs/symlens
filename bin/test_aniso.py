from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import symlens


nsims = 40
deg = 25.
px = 2.0

shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px,proj='plain')


modlmap = enmap.modlmap(shape,wcs)
ymap,xmap = enmap.posmap(shape,wcs)

omap = np.sin(ymap/np.pi*100) + np.cos(xmap/np.pi*100)
mfact = 10
afact = 20
rms = (omap - omap.min())*mfact + afact
# io.hplot(rms,colorbar=True)

pmap = enmap.pixsizemap(shape,wcs)

ivar = maps.ivar(shape,wcs,rms,ipsizemap=pmap)
# io.hplot(ivar,colorbar=True)

my_tasks = range(nsims)

theory = cosmology.default_theory()
cov = theory.lCl('TT',modlmap)
mgen = maps.MapGen((1,)+shape,wcs,cov=cov[None,None])

fwhm = 1.5
wnoise = 40.
kbeam = maps.gauss_beam(modlmap,fwhm)

feed_dict = {}
lmin = 200
lmax = 3000
Lmin = 40
Lmax = 3000
xmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
ymask = xmask
kmask = maps.mask_kspace(shape,wcs,lmin=Lmin,lmax=Lmax)
feed_dict['uC_T_T'] = cov
feed_dict['tC_T_T'] = cov + (wnoise*np.pi/180/60)**2 / kbeam**2
qe = symlens.QE(shape,wcs,feed_dict,'hu_ok','TT',xmask=xmask,ymask=ymask,kmask=kmask)

s = stats.Stats()

for task in my_tasks:

    cmb = maps.filter_map(mgen.get_map(seed=(1,task))[0],kbeam)
    nseed = (2,task)
    nmap = maps.white_noise(shape,wcs,noise_muK_arcmin=None,seed=nseed,ipsizemap=pmap,div=ivar)

    obs = cmb + nmap
    kobs = enmap.fft(obs,normalize='phys')/kbeam
    kobs[~np.isfinite(kobs)] = 0

    feed_dict['X'] = kobs
    feed_dict['Y'] = kobs
    krecon = qe.reconstruct(feed_dict)
    
    print(cmb.shape,nmap.shape,krecon.shape)
    s.add_to_stack('kreal',krecon.real)
    s.add_to_stack('kimag',krecon.imag)

s.get_stacks()

mf = enmap.ifft(s.stacks['kreal'] + 1j*s.stacks['kimag'],normalize='phys')
io.hplot(mf,'mf.png')
    
