import camb
import os,sys
import numpy as np
from camb import model
import matplotlib.pyplot as plt

def save_cols(filename,tuple_of_vectors,**kwargs):
    tuple_of_vectors = np.asarray(tuple_of_vectors)
    save_mat = np.vstack(tuple_of_vectors).T
    np.savetxt(filename,save_mat,**kwargs)

for exp in ['planck','act','default']:
    print(exp)

    # These are reasonable accuracy settings for this ellmax
    AccuracyBoost = 1.0
    lSampleBoost = 1.0
    lAccuracyBoost = 1.0
    lens_potential_accuracy = 4
    lens_margin = 1250
    lmax = 4000

    if exp=='planck':
        #Planck 2018 SET PARAMETERS HERE
        As = np.exp(3.087)/10**10
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.51, ombh2=0.02241 , omch2=0.1197, mnu=0.06, omk=0, tau=0.076)
        pars.InitPower.set_params(As=As, ns=0.9668, r=0)
    elif exp=='act':
        # ACT DR4 SET PARAMETERS HERE
        As = np.exp(3.068)/10**10
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.1, ombh2=0.0223 , omch2=0.1212, mnu=0.06, omk=0, tau=0.061)
        pars.InitPower.set_params(As=As, ns=0.9714, r=0)
    elif exp=='default':
        # Calabrese et al Planck 2015 SET PARAMETERS HERE
        As = 2.15086031154146e-9
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.02393, ombh2=0.02219218 , omch2=0.1203058, mnu=0.06, omk=0, tau=0.06574325)
        pars.InitPower.set_params(As=As, ns=0.9625356, r=0)


    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy, lens_margin=lens_margin)
    pars.set_accuracy(AccuracyBoost=AccuracyBoost, lSampleBoost=lSampleBoost, lAccuracyBoost=lAccuracyBoost,
                      DoLateRadTruncation=False)
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params('mead2020')

    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    #powers_grads = results.get_lensed_gradient_cls(CMB_unit='muK')

    l = np.arange(lmax)
    l_fact = (l*(l+1)/(2*np.pi))
    #ps_planck2018_theo = np.zeros((3,3,lmax))
    cltt = powers["total"][:lmax,0]/(l*(l+1)/(2*np.pi))
    clte = powers["total"][:lmax,3]/(l*(l+1)/(2*np.pi))
    clee = powers["total"][:lmax,1]/(l*(l+1)/(2*np.pi))
    clbb = powers["total"][:lmax,2]/(l*(l+1)/(2*np.pi))

    save_cols(f'{exp}_cls.txt',(l,cltt,clte,clee,clbb))

def get_spec_1d(exp,lmax=None):
    ls,cltt = np.loadtxt(f'{exp}_cls.txt',usecols=[0,1],unpack=True)
    cltt[ls<2] = 0
    if not(lmax is None):
        cltt = cltt[ls<lmax]
        ls = ls[ls<lmax]
    return ls,cltt

ls,cl_d = get_spec_1d('default',3000)
ls,cl_p = get_spec_1d('planck',3000)
ls,cl_a = get_spec_1d('act',3000)

plt.plot(ls,(cl_p-cl_d)/cl_d,label='Planck 2018 TT vs. default')
plt.plot(ls,(cl_a-cl_d)/cl_d,label='ACT DR4+WMAP TT vs. default')
plt.axhline(y=0)
plt.ylim(-0.01,0.035)
plt.ylabel('$\\Delta C_{\\ell} / C_{\\ell}$')
plt.xlabel('$\\ell$')
plt.legend()
plt.savefig('clttdiff.png',dpi=250)


