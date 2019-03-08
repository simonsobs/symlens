import symlens as s
from pixell import enmap, utils as putils, powspec
import os,sys
from scipy.interpolate import interp1d
import numpy as np
import pytest


"""
Only tests marked @pytest.mark.webtest
will be run on Travis
"""

@pytest.mark.webtest
def test_trivial():
    assert 1==1


# The following tests need orphics, and so are not meant to be run on Travis
def test_hdv_huok_planck():
    from orphics import lensing,io,cosmology,maps

    shape,wcs = enmap.geometry(shape=(512,512),res=2.0*putils.arcmin,pos=(0,0))
    modlmap = enmap.modlmap(shape,wcs)
    theory = cosmology.default_theory()
    ells = np.arange(0,3000,1)
    ctt = theory.lCl('TT',ells)
    # ps,_ = powspec.read_camb_scalar("tests/Aug6_highAcc_CDM_scalCls.dat")
    # ells = range(ps.shape[-1])

    ## Build HuOk TT estimator
    f = s.Ldl1 * s.e('uC_T_T_l1') + s.Ldl2 * s.e('uC_T_T_l2')
    F = f / 2 / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')
    expr1 = f * F
    feed_dict = {}
    feed_dict['uC_T_T'] = s.interp(ells,ctt)(modlmap)
    feed_dict['tC_T_T'] = s.interp(ells,ctt)(modlmap)+(33.*np.pi/180./60.)**2./s.gauss_beam(modlmap,7.0)**2.
    tellmin = 10 ; tellmax = 3000
    xmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    integral = s.integrate(shape,wcs,feed_dict,expr1,xmask=xmask,ymask=xmask).real
    Nl = modlmap**4./integral/4.
    bin_edges = np.arange(10,3000,40)
    binner = s.bin2D(modlmap,bin_edges)
    cents,nl1d = binner.bin(Nl)
    

    
    ## Build HDV TT estimator
    F = s.Ldl1 * s.e('uC_T_T_l1') / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')
    expr1 = f*F
    integral = s.integrate(shape,wcs,feed_dict,expr1,xmask=xmask,ymask=xmask).real
    Nl = modlmap**4./integral/4.

    cents,nl1d2 = binner.bin(Nl)

    cents,nl1d3 = binner.bin(s.N_l_cross(shape,wcs,feed_dict,"hu_ok","TT","hu_ok","TT",
                                         xmask=xmask,ymask=xmask))
    cents,nl1d4 = binner.bin(s.N_l_cross(shape,wcs,feed_dict,"hdv","TT","hdv","TT",
                                       xmask=xmask,ymask=xmask))
    cents,nl1d5 = binner.bin(s.N_l(shape,wcs,feed_dict,"hu_ok","TT",xmask=xmask,ymask=xmask))
    cents,nl1d6 = binner.bin(s.N_l(shape,wcs,feed_dict,"hdv","TT",xmask=xmask,ymask=xmask))

    clkk = theory.gCl('kk',ells)
    pl = io.Plotter(xyscale='linlog')
    pl.add(cents,nl1d)
    pl.add(cents,nl1d2)
    # pl.add(cents,nl1d3)
    pl.add(cents,nl1d4)
    pl.add(cents,nl1d5)
    pl.add(cents,nl1d6)
    pl.add(ells,clkk)
    pl.done("plcomp.png")
    

def test_lens_recon():
    from orphics import lensing,io,cosmology,maps
    from enlib import bench

    deg = 10.
    px = 2.0
    tellmin = 100
    tellmax = 3000
    kellmin = 40
    kellmax = 3000
    grad_cut = None
    bin_width = 80
    beam_arcmin = 0.01
    noise_uk_arcmin = 0.01
    
    theory = cosmology.default_theory(lpad=30000)
    shape,wcs = s.rect_geometry(width_deg=deg,px_res_arcmin=px)
    flsims = lensing.FlatLensingSims(shape,wcs,theory,beam_arcmin,noise_uk_arcmin)
    kbeam = flsims.kbeam
    modlmap = enmap.modlmap(shape,wcs)
    fc = maps.FourierCalc(shape,wcs)
    n2d = (noise_uk_arcmin*np.pi/180./60.)**2./flsims.kbeam**2.
    tmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    kmask = s.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
    with bench.show("orphics init"):
        qest = lensing.qest(shape,wcs,theory,
                            noise2d=n2d,kmask=tmask,kmask_K=kmask,
                            pol=False,grad_cut=grad_cut,
                            unlensed_equals_lensed=True,bigell=30000)
    bin_edges = np.arange(kellmin,kellmax,bin_width)
    binner = s.bin2D(modlmap,bin_edges)
    i = 0
    unlensed,kappa,lensed,beamed,noise_map,observed = flsims.get_sim(seed_cmb=(i,1),
                                                                     seed_kappa=(i,2),
                                                                     seed_noise=(i,3),
                                                                     lens_order=5,
                                                                     return_intermediate=True)

    kmap = enmap.fft(observed,normalize="phys")
    # _,kmap,_ = fc.power2d(observed)
    with bench.show("orphics"):
        kkappa = qest.kappa_from_map("TT",kmap/kbeam,alreadyFTed=True,returnFt=True)    
    pir2d,kinput = fc.f1power(kappa,kkappa)
    pii2d = fc.f2power(kinput,kinput)
    prr2d = fc.f2power(kkappa,kkappa)
    cents,pir1d = binner.bin(pir2d)
    cents,pii1d = binner.bin(pii2d)
    cents,prr1d = binner.bin(prr2d)

    feed_dict = {}
    cltt = theory.lCl('TT',modlmap)
    feed_dict['uC_T_T'] = theory.lCl('TT',modlmap)
    feed_dict['tC_T_T'] = cltt+n2d
    feed_dict['X'] = kmap/kbeam
    feed_dict['Y'] = kmap/kbeam

    with bench.show("symlens init"):
        Al = s.A_l(shape,wcs,feed_dict,"hdv","TT",xmask=tmask,ymask=tmask)
    Nl = s.N_l_from_A_l_optimal(shape,wcs,Al)
    with bench.show("symlens"):
        ukappa = s.unnormalized_quadratic_estimator(shape,wcs,feed_dict,"hdv","TT",xmask=tmask,ymask=tmask)
    nkappa = Al * ukappa

    pir2d2 = fc.f2power(nkappa,kinput)
    cents,pir1d2 = binner.bin(pir2d2)


    cents,Nlkk = binner.bin(qest.N.Nlkk['TT'])
    cents,Nlkk2 = binner.bin(Nl)

    pl = io.Plotter(xyscale='linlog')
    pl.add(cents,pii1d,color='k',lw=3)
    pl.add(cents,pir1d,label='orphics')
    pl.add(cents,pir1d2,label='hdv symlens')
    pl.add(cents,Nlkk,ls="--",label='orphics')
    pl.add(cents,Nlkk2,ls="-.",label='symlens')
    pl.done("ncomp.png")

def test_shear():
    from orphics import lensing,io,cosmology,maps

    deg = 20.
    px = 2.0
    tellmin = 30
    tellmax = 3500
    kellmin = 10
    kellmax = 3000
    bin_width = 20
    beam_arcmin = 1.4
    noise_uk_arcmin = 7.0
    
    theory = cosmology.default_theory(lpad=30000)
    shape,wcs = s.rect_geometry(width_deg=deg,px_res_arcmin=px)
    flsims = lensing.FlatLensingSims(shape,wcs,theory,beam_arcmin,noise_uk_arcmin)
    kbeam = flsims.kbeam
    modlmap = enmap.modlmap(shape,wcs)
    fc = maps.FourierCalc(shape,wcs)
    n2d = (noise_uk_arcmin*np.pi/180./60.)**2./flsims.kbeam**2.
    tmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    kmask = s.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
    bin_edges = np.arange(kellmin,kellmax,bin_width)
    binner = s.bin2D(modlmap,bin_edges)
    i = 0
    unlensed,kappa,lensed,beamed,noise_map,observed = flsims.get_sim(seed_cmb=(i,1),
                                                                     seed_kappa=(i,2),
                                                                     seed_noise=(i,3),
                                                                     lens_order=5,
                                                                     return_intermediate=True)
    _,kmap,_ = fc.power2d(observed)
    pii2d,kinput,_ = fc.power2d(kappa)

    feed_dict = {}
    cltt = theory.lCl('TT',modlmap)
    feed_dict['uC_T_T'] = theory.lCl('TT',modlmap)
    feed_dict['tC_T_T'] = (cltt+n2d)
    feed_dict['X'] = kmap/kbeam
    feed_dict['Y'] = kmap/kbeam

    ells = np.arange(0,10000,1)
    ucltt = theory.lCl('TT',ells)
    feed_dict['duC_T_T'] = s.interp(ells,np.gradient(np.log(ucltt),np.log(ells)))(modlmap)
    sAl = s.A_l(shape,wcs,feed_dict,"shear","TT",xmask=tmask,ymask=tmask)
    sNl = s.N_l(shape,wcs,feed_dict,"shear","TT",
              xmask=tmask,ymask=tmask,
              Al=sAl)
    sukappa = s.unnormalized_quadratic_estimator(shape,wcs,feed_dict,"shear","TT",xmask=tmask,ymask=tmask)
    snkappa = sAl * sukappa
    

    pir2d3 = fc.f2power(snkappa,kinput)
    cents,pir1d3 = binner.bin(pir2d3)
    cents,pii1d = binner.bin(pii2d)
    cents,prr1d = binner.bin(fc.f2power(snkappa,snkappa))
    
    cents,Nlkk3 = binner.bin(sNl)

    pl = io.Plotter(xyscale='loglog')
    pl.add(ells,theory.gCl('kk',ells))
    pl.add(cents,pii1d,color='k',lw=3)
    pl.add(cents,pir1d3,label='shear')
    pl.add(cents,prr1d)
    pl.add(cents,Nlkk3,ls=":")
    pl._ax.set_xlim(10,3500)
    pl.done("ncomp.png")
    
def test_pol():
    from orphics import lensing,io,cosmology,maps

    est = "hu_ok"
    pols = ['TT','EE','TE','EB','TB']
    # est = "hdv"
    # pols = ['TT','EE','TE','ET','EB','TB']

    deg = 5.
    px = 2.0
    tellmin = 30
    tellmax = 3000
    pellmin = 30
    pellmax = 5000
    kellmin = 10
    kellmax = 5000
    bin_width = 40
    
    beam_arcmin = 1.5
    noise_uk_arcmin = 10.0
    
    theory = cosmology.default_theory(lpad=30000)
    shape,wcs = s.rect_geometry(width_deg=deg,px_res_arcmin=px)
    modlmap = enmap.modlmap(shape,wcs)
    kbeam = s.gauss_beam(modlmap,beam_arcmin)
    n2d = (noise_uk_arcmin*np.pi/180./60.)**2./kbeam**2.
    tmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    pmask = s.mask_kspace(shape,wcs,lmin=pellmin,lmax=pellmax)
    kmask = s.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
    bin_edges = np.arange(kellmin,kellmax,bin_width)
    binner = s.bin2D(modlmap,bin_edges)
    
    feed_dict = {}
    cltt = theory.lCl('TT',modlmap)
    clee = theory.lCl('EE',modlmap)
    clbb = theory.lCl('BB',modlmap)
    clte = theory.lCl('TE',modlmap)
    feed_dict['uC_T_T'] = cltt
    feed_dict['tC_T_T'] = (cltt+n2d)
    feed_dict['uC_E_E'] = clee
    feed_dict['tC_E_E'] = (clee+n2d*2.)
    feed_dict['uC_B_B'] = clbb
    feed_dict['tC_B_B'] = (clbb+n2d*2.)
    feed_dict['uC_T_E'] = clte
    feed_dict['tC_T_E'] = clte

    ells = np.arange(0,10000,1)
    pl = io.Plotter(xyscale='loglog')
    pl.add(ells,theory.gCl('kk',ells))
    imask = {'T':tmask, 'E':pmask,'B': pmask}
    for pol in pols:
        print(pol)
        X,Y = pol
        cents,Nl = binner.bin(s.N_l(shape,wcs,feed_dict,est,pol,xmask=imask[X],ymask=imask[Y]))
        pl.add(cents,Nl,label=pol)
    pl._ax.set_xlim(10,kellmax)
    pl.done("nls.png")
    
    
