from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from pixell import fft as efft, enmap
import os,sys
from .factorize import Ldl1,Ldl2,l1,l2,cos2t12,sin2t12,l1x,l2x,l1y,l2y,e,L,Lx,Ly,Lxl1,Lxl2,integrate
import warnings

def cross_names(x,y,fn1,fn2,dstr="t"):
    if fn1 is None: e1 = ""
    else:
        assert "_" not in fn1, "Field names cannot have underscores. Sorry! Use a hyphen instead."
        e1 = fn1+"_"
    if fn2 is None: e2 = ""
    else:
        assert "_" not in fn2, "Field names cannot have underscores. Sorry! Use a hyphen instead."
        e2 = fn2+"_"
    return '%sC_%s%s_%s%s' % (dstr,e1,x,e2,y)

def _get_groups(e1,e2=None,noise=True):
    if e2 is None: e2 = e1
    if e1 != e2: return None
    if e1 in ['hu_ok','hdv']:
        if noise:
            return [Lx*Lx,Ly*Ly,Lx*Ly]
        else:
            return [Ly,Lx]
    else:
        return None
    
class HardenedTT(object):
    def __init__(self,shape,wcs,feed_dict,xmask=None,ymask=None,kmask=None,Al=None,hardening='src',estimator='hu_ok'):
        h = hardening
        f_bias,F_bias,_ = get_mc_expressions(hardening,'TT')
        f_phi,F_phi,_ = get_mc_expressions(estimator,'TT')
        f_bh,F_bh,_ = get_mc_expressions(f'{h}-hardened','TT',estimator_to_harden=estimator)
        self.fdict = feed_dict
        # 1 / Response of the biasing agent to the biasing agent
        self.fdict[f'A{h}_{h}_L'] = A_l_custom(shape,wcs,feed_dict,f_bias,F_bias,xmask=xmask,ymask=ymask,groups=None,kmask=kmask)
        # 1 / Response of the biasing agent to CMB lensing
        self.fdict[f'Aphi_{h}_L'] = A_l_custom(shape,wcs,feed_dict,f_phi,F_bias,xmask=xmask,ymask=ymask,groups=None,kmask=kmask)
        self.Al = A_l_custom(shape,wcs,feed_dict,f_bh,F_bh,xmask=xmask,ymask=ymask,groups=None,kmask=kmask) if Al is None else Al
        self.F_bh = F_bh
        self.xmask = xmask
        self.ymask = ymask
        self.kmask = kmask
        self.shape,self.wcs = shape,wcs
    def get_Nl(self):
        return N_l_cross_custom(self.shape,self.wcs,self.fdict,"TT","TT",self.F_bh,self.F_bh,self.F_bh,
                                     xmask=self.xmask,ymask=self.ymask,
                                     Aalpha=self.Al,Abeta=self.Al,groups=None,kmask=self.kmask)
    def reconstruct(self,feed_dict,xname='X_l1',yname='Y_l2',groups=None,physical_units=True):
        uqe = unnormalized_quadratic_estimator_custom(self.shape,self.wcs,feed_dict,
                                                      self.F_bh,xname=xname,yname=yname,
                                                      xmask=self.xmask,ymask=self.ymask,
                                                      groups=groups,physical_units=physical_units)
        return self.Al * uqe * self.kmask
        
        
    

class QE(object):
    """Construct a quadratic estimator such that the normalization is pre-calculated
    and reused.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization 
        calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below).
    estimator: str,optional
        The name of a pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    XY: str,optional
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    f: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. If this is specified, the argument F is required, and the arguments
        estimator, XY and field_names are ignored.
    F: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter. See the Usage guide
        for details. 
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    groups: list,optional 
        Group all terms in the normalization such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.


    """
    def __init__(self,shape,wcs,feed_dict,estimator=None,XY=None,
                 f=None,F=None,xmask=None,ymask=None,
                 field_names=None,groups=None,kmask=None):
        """
        """
        if f is not None:
            assert F is not None
            assert estimator is None
            assert XY is None
            self.Al = A_l_custom(shape,wcs,feed_dict,f,F,xmask=xmask,ymask=ymask,groups=groups,kmask=kmask)
            self.F = F
            self.custom = True
        else:
            assert F is None
            self.Al = A_l(shape,wcs,feed_dict,estimator,XY,xmask=xmask,ymask=ymask,field_names=field_names,kmask=kmask)
            self.estimator = estimator
            self.XY = XY
            self.field_names = field_names
            self.custom = False
        self.shape = shape
        self.wcs = wcs
        self.xmask = xmask
        self.ymask = ymask
        kmask = 1 if kmask is None else kmask
        self.kmask = kmask
        
    def reconstruct(self,feed_dict,xname='X_l1',yname='Y_l2',groups=None,physical_units=True):
        """
        Returns a normalized reconstruction corresponding to the initialized
        mode-coupling estimator.

        Parameters
        ----------

        feed_dict: dict
            Mapping from names of custom symbols to numpy arrays used in the reconstruction 
            calculation. When using pre-defined mode-coupling estimators, typical
            keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
            depend on the requested estimator XY (see below). This feed_dict must
            also contain the keys with name xname and yname (see below), which 
            contain the 2D maps X and Y for the data from which the quadratic estimate
            is constructed.
        xname: str,optional
            The name of the key in feed_dict where the X map in the XY quadratic estimator
            is stored. Defaults to X_l1.
        yname: str,optional
            The name of the key in feed_dict where the Y map in the XY quadratic estimator
            is stored. Defaults to Y_l2.
        groups: list,optional 
            Group all terms in the reconstruction calclulation such that they have common factors 
            of the provided list of expressions to reduce the number of FFTs.

        Returns
        -------

        krecon : (Ny,Nx) ndarray
            The normalized Fourier space reconstruction in physical units (not pixel units).
        """
        if self.custom:
            uqe = unnormalized_quadratic_estimator_custom(self.shape,self.wcs,feed_dict,
                                                           self.F,xname=xname,yname=yname,
                                                           xmask=self.xmask,ymask=self.ymask,
                                                           groups=groups,physical_units=physical_units)
        else:
            uqe = unnormalized_quadratic_estimator(self.shape,self.wcs,feed_dict,
                                                   self.estimator,self.XY,
                                                   xname=xname,yname=yname,
                                                   field_names=self.field_names,
                                                   xmask=self.xmask,ymask=self.ymask,physical_units=physical_units)
        return self.Al * uqe * self.kmask

def cross_integral_custom(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                           xmask=None,ymask=None,
                           field_names_alpha=None,field_names_beta=None,groups=None,power_name="t"):
    """
    Calculates the integral

    .. math::
        \\int \\frac{d^2 \\vec{l}_1 }{ (2\\pi)^2 }  F_{\\alpha}(\\vec{l}_1,\\vec{l}_2) (F_{\\beta}(\\vec{l}_1,\\vec{l}_2) C^{ac}_{l_1} C^{bd}_{l_2}+ F_{\\beta}(\\vec{l}_2,\\vec{l}_1) C^{ad}_{l_1} C^{bc}_{l_2})

    where
    alpha_XY = "ab"
    beta_XY = "cd"


    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    alpha_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    beta_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    Falpha: :obj:`sympy.core.symbol.Symbol`
        A sympy expression containing the first alpha estimator filter. See the Usage guide
        for details. 
    Fbeta: :obj:`sympy.core.symbol.Symbol` 
        A sympy expression containing the second beta estimator filter. See the Usage guide
        for details. 
    Fbeta_rev: :obj:`sympy.core.symbol.Symbol` 
        Same as above but with l1 and l2 swapped.
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names_alpha: 2 element list, optional
        Providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    field_names_beta: 2 element list, optional
        As above, but for the second beta estimator.
    groups: list,optional 
        Group all terms in the normalization calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    integral : (Ny,Nx) ndarray
        Returns the integral described above.


    """
    a,b = alpha_XY
    c,d = beta_XY
    fnalpha1,fnalpha2 = field_names_alpha if field_names_alpha is not None else (None,None)
    fnbeta1,fnbeta2 = field_names_beta if field_names_beta is not None else (None,None)
    tCac_l1 = e(cross_names(a,c,fnalpha1,fnbeta1,power_name) + "_l1")
    tCbd_l2 = e(cross_names(b,d,fnalpha2,fnbeta2,power_name) + "_l2")
    tCad_l1 = e(cross_names(a,d,fnalpha1,fnbeta2,power_name) + "_l1")
    tCbc_l2 = e(cross_names(b,c,fnalpha2,fnbeta1,power_name) + "_l2")
    Dexpr1 = tCac_l1*tCbd_l2
    Dexpr2 = tCad_l1*tCbc_l2
    return generic_cross_integral(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,Dexpr1,Dexpr2,
                           xmask=xmask,ymask=ymask,
                           field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,groups=groups)

def generic_cross_integral(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,Dexpr1,Dexpr2,
                           xmask=None,ymask=None,
                           field_names_alpha=None,field_names_beta=None,groups=None):
    """
    Calculates the integral

    .. math::
        \\int \\frac{d^2 \\vec{l}_1 }{ (2\\pi)^2 }  F_{\\alpha}(\\vec{l}_1,\\vec{l}_2) (F_{\\beta}(\\vec{l}_1,\\vec{l}_2) D_1({l_1},{l_2})+ F_{\\beta}(\\vec{l}_2,\\vec{l}_1) D_2({l_1},{l_2}))

    where
    alpha_XY = "ab"
    beta_XY = "cd"


    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    alpha_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    beta_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    Falpha: :obj:`sympy.core.symbol.Symbol`
        A sympy expression containing the first alpha estimator filter. See the Usage guide
        for details. 
    Fbeta: :obj:`sympy.core.symbol.Symbol` 
        A sympy expression containing the second beta estimator filter. See the Usage guide
        for details. 
    Fbeta_rev: :obj:`sympy.core.symbol.Symbol` 
        Same as above but with l1 and l2 swapped.
    Dexpr1: :obj:`sympy.core.symbol.Symbol`  
        A sympy expression entering in the generic integral.
    Dexpr2: :obj:`sympy.core.symbol.Symbol`  
        A second sympy expression entering in the generic integral.
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names_alpha: 2 element list, optional
        Providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    field_names_beta: 2 element list, optional
        As above, but for the second beta estimator.
    groups: list,optional 
        Group all terms in the normalization calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    integral : (Ny,Nx) ndarray
        Returns the integral described above.


    """
    a,b = alpha_XY
    c,d = beta_XY
    fnalpha1,fnalpha2 = field_names_alpha if field_names_alpha is not None else (None,None)
    fnbeta1,fnbeta2 = field_names_beta if field_names_beta is not None else (None,None)
    expr = Falpha*(Fbeta*Dexpr1+Fbeta_rev*Dexpr2)
    integral = integrate(shape,wcs,feed_dict,expr,xmask=xmask,ymask=ymask,groups=groups,
                         physical_units=False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    assert np.all(np.isfinite(integral))
    return integral

def N_l_cross_custom(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                      xmask=None,ymask=None,
                      field_names_alpha=None,field_names_beta=None,
                      falpha=None,fbeta=None,Aalpha=None,Abeta=None,
                     groups=None,kmask=None,power_name="t"):
    """
    Returns the 2D cross-covariance between two custom mode-coupling estimators. 
    This involves 3 integrals, unless pre-calculated normalizations Al are provided.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    alpha_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    beta_XY: str
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    falpha: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response for the first estimator alpha. 
        See the Usage guide
        for details. 
    fbeta: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response for the second estimator beta. 
        See the Usage guide for details. 
    Falpha: :obj:`sympy.core.symbol.Symbol`
        A sympy expression containing the first alpha estimator filter. See the Usage guide
        for details. 
    Fbeta: :obj:`sympy.core.symbol.Symbol` 
        A sympy expression containing the second beta estimator filter. See the Usage guide
        for details. 
    Fbeta_rev: :obj:`sympy.core.symbol.Symbol` 
        Same as above but with l1 and l2 swapped.
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names_alpha: 2 element list, optional
        Providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    field_names_beta: 2 element list, optional
        As above, but for the second beta estimator.
    groups: list,optional 
        Group all terms in the normalization calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.
    Aalpha: (Ny,Nx) ndarray, optional
        Pre-calculated normalization for the first estimator. This is calculated if
        not provided
    Abeta: (Ny,Nx) ndarray, optional
        Pre-calculated normalization for the second estimator. This is calculated if
        not provided

    Returns
    -------

    Nl : (Ny,Nx) ndarray
        The requested 2D cross-covariance.

    """
    cross_integral = cross_integral_custom(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                                           xmask=xmask,ymask=ymask,
                                           field_names_alpha=field_names_alpha,
                                           field_names_beta=field_names_beta,
                                           groups=groups,power_name=power_name)
    return generic_noise_expression(cross_integral,shape,wcs,feed_dict,falpha,fbeta,Falpha,Fbeta, \
                                    xmask,ymask,kmask,Aalpha,Abeta)

def generic_noise_expression(cross_integral,shape,wcs,feed_dict,falpha,fbeta,Falpha,Fbeta,xmask=None,ymask=None,kmask=None,
                             Aalpha=None,Abeta=None):
    """
    returns (1/4) A_alpha * A_beta * cross_integral
    """
    if Aalpha is None: Aalpha = A_l_custom(shape,wcs,feed_dict,falpha,Falpha,xmask=xmask,ymask=ymask,kmask=kmask)
    if Abeta is None: Abeta = A_l_custom(shape,wcs,feed_dict,fbeta,Fbeta,xmask=xmask,ymask=ymask,kmask=kmask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 0.25 * Aalpha * Abeta * cross_integral


def RDN0_analytic(shape,wcs,feed_dict,alpha_estimator,alpha_XY,beta_estimator,beta_XY,
                  split_estimator=False,Aalpha=None,Abeta=None,xmask=None,ymask=None,kmask=None,
                  field_names_alpha=None,field_names_beta=None,skip_filter_field_names=False):
    """
    Often lovingly called the `dumb' N0 by ACT lensers, this is the analytic expression
    for the realization-dependependent N0 correction when the noise is isotropic and
    no mask is present.

    feed_dict should have
    dC_T_T, etc. the realized total data power spectrum.
    tC_T_T, etc. should be the total coadd power spectrum used in filters
    uC_T_T, etc. the usual theory spectra for the CMB signal.
    nC_T_T etc. should be the expected total theory power spectrum
    But if split_estimator is true:
    dC_T_T, etc. should be the realized cross-power average.
    nC_T_T etc. should be the expected cross-power, which is usually nC without the instrument noise.
    """
    falpha,Falpha,Falpha_rev = get_mc_expressions(alpha_estimator,alpha_XY,field_names=field_names_alpha if not(skip_filter_field_names) else None)
    fbeta,Fbeta,Fbeta_rev = get_mc_expressions(beta_estimator,beta_XY,field_names=field_names_beta  if not(skip_filter_field_names) else None)
    return RDN0_analytic_generic(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                                 falpha=falpha,fbeta=fbeta,Aalpha=Aalpha,Abeta=Abeta,xmask=xmask,ymask=ymask,kmask=kmask,
                                 field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,
                                 split_estimator=split_estimator)

def RDN0_analytic_generic(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                          falpha=None,fbeta=None,Aalpha=None,Abeta=None,xmask=None,ymask=None,kmask=None,
                          field_names_alpha=None,field_names_beta=None,split_estimator=False,groups=None):
    """
    Often lovingly called the `dumb' N0 by ACT lensers, this is the analytic expression
    for the realization-dependependent N0 correction when the noise is isotropic and
    no mask is present.

    feed_dict should have
    dC_T_T, etc. the realized total data power spectrum.
    tC_T_T, etc. should be the total coadd power spectrum used in filters
    uC_T_T, etc. the usual theory spectra for the CMB signal.
    nC_T_T etc. should be the expected total theory power spectrum
    But if split_estimator is true:
    dC_T_T, etc. should be the realized cross-power average.
    nC_T_T etc. should be the expected cross-power, which is usually nC without the instrument noise.
    """
    a,b = alpha_XY
    c,d = beta_XY
    fnalpha1,fnalpha2 = field_names_alpha if field_names_alpha is not None else (None,None)
    fnbeta1,fnbeta2 = field_names_beta if field_names_beta is not None else (None,None)
    tCac_l1 = e(cross_names(a,c,fnalpha1,fnbeta1,'n') + "_l1")
    tCbd_l2 = e(cross_names(b,d,fnalpha2,fnbeta2,'n') + "_l2")
    tCad_l1 = e(cross_names(a,d,fnalpha1,fnbeta2,'n') + "_l1")
    tCbc_l2 = e(cross_names(b,c,fnalpha2,fnbeta1,'n') + "_l2")
    dCac_l1 = e(cross_names(a,c,fnalpha1,fnbeta1,'d') + "_l1")
    dCbd_l2 = e(cross_names(b,d,fnalpha2,fnbeta2,'d') + "_l2")
    dCad_l1 = e(cross_names(a,d,fnalpha1,fnbeta2,'d') + "_l1")
    dCbc_l2 = e(cross_names(b,c,fnalpha2,fnbeta1,'d') + "_l2")
    Dexpr1 = dCac_l1*tCbd_l2 + tCac_l1*dCbd_l2 - tCac_l1*tCbd_l2
    Dexpr2 = dCad_l1*tCbc_l2 + tCad_l1*dCbc_l2 - tCad_l1*tCbc_l2
    gint = generic_cross_integral(shape,wcs,feed_dict,alpha_XY,alpha_XY,Falpha,Fbeta,Fbeta_rev,Dexpr1,Dexpr2,
                                  xmask=xmask,ymask=ymask,
                                  field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,groups=groups)
    return generic_noise_expression(gint,shape,wcs,feed_dict,falpha,fbeta,Falpha,Fbeta, \
                                    xmask,ymask,kmask,Aalpha,Abeta)

    
def N_l_cross(shape,wcs,feed_dict,alpha_estimator,alpha_XY,beta_estimator,beta_XY,
              xmask=None,ymask=None,
              Aalpha=None,Abeta=None,field_names_alpha=None,field_names_beta=None,kmask=None,
              skip_filter_field_names=False,power_name="t"):
    """
    Returns the 2D cross-covariance between two pre-defined mode-coupling estimators. 
    This involves 3 integrals, unless pre-calculated normalizations Al are provided.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    alpha_estimator: str
        The name of the first pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    beta_estimator: str,optional
        The name of the second pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    alpha_XY: str,optional
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    beta_XY: str,optional
        The XY pair for the first estimator. Typical examples include "TT" and "EB".
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    Aalpha: (Ny,Nx) ndarray, optional
        Pre-calculated normalization for the first estimator. This is calculated if
        not provided
    Abeta: (Ny,Nx) ndarray, optional
        Pre-calculated normalization for the second estimator. This is calculated if
        not provided
    field_names_alpha: 2 element list, optional
        Providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    field_names_beta: 2 element list, optional
        As above, but for the second beta estimator.


    Returns
    -------

    Nl : (Ny,Nx) ndarray
        The requested 2D cross-covariance.

    """
    falpha,Falpha,Falpha_rev = get_mc_expressions(alpha_estimator,alpha_XY,field_names=field_names_alpha if not(skip_filter_field_names) else None)
    fbeta,Fbeta,Fbeta_rev = get_mc_expressions(beta_estimator,beta_XY,field_names=field_names_beta  if not(skip_filter_field_names) else None)
    return N_l_cross_custom(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                            xmask=xmask,ymask=ymask,
                            field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,
                            falpha=falpha,fbeta=fbeta,Aalpha=Aalpha,Abeta=Abeta,
                            groups=_get_groups(alpha_estimator,beta_estimator),kmask=kmask,
                            power_name=power_name)

def N_l(shape,wcs,feed_dict,estimator,XY,
        xmask=None,ymask=None,
        Al=None,field_names=None,kmask=None,power_name="t"):
    """
    Returns the 2D noise corresponding to a pre-defined mode-coupling estimator
    NOT assuming that it is optimal. This involves 2 integrals, unless a pre-calculated
    normalization Al is provided, in which case only 1 integral needs to be evaluated.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    estimator: str
        The name of a pre-defined mode-coupling estimator. 
        e.g. "hu_ok", "hdv" and "shear".
    XY: str
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    Al: (Ny,Nx) ndarray, optional
        Pre-calculated normalization for the estimator. Reduces the number of integrals 
        calculated to 1 if provided, else calculates 2 integrals.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.

    Returns
    -------

    Nl : (Ny,Nx) ndarray
        The 2D noise for the estimator.

    """
    falpha,Falpha,Falpha_rev = get_mc_expressions(estimator,XY,field_names=field_names)
    return N_l_cross_custom(shape,wcs,feed_dict,XY,XY,Falpha,Falpha,Falpha_rev,
                            xmask=xmask,ymask=ymask,
                            field_names_alpha=field_names,field_names_beta=field_names,
                            falpha=falpha,fbeta=falpha,Aalpha=Al,Abeta=Al,
                            groups=_get_groups(estimator),kmask=kmask,power_name=power_name)

    
def A_l_custom(shape,wcs,feed_dict,f,F,xmask=None,ymask=None,groups=None,kmask=None):
    """
    Returns the 2D normalization corresponding to a custom
    mode-coupling estimator.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    f: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. If this is specified, the argument F is required, and the arguments
        estimator, XY and field_names are ignored.
    F: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter. See the Usage guide
        for details. 
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    xname: str,optional
        The name of the key in feed_dict where the X map in the XY quadratic estimator
        is stored. Defaults to X_l1.
    yname: str,optional
        The name of the key in feed_dict where the Y map in the XY quadratic estimator
        is stored. Defaults to Y_l2.
    groups: list,optional 
        Group all terms in the normalization calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    Al : (Ny,Nx) ndarray
        The 2D normalization for the estimator.


    """
    integral = integrate(shape,wcs,feed_dict,f*F/L/L,
                         xmask=xmask,ymask=ymask,groups=groups,
                         physical_units=False).real * enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    modlmap = enmap.modlmap(shape,wcs)
    assert np.all(np.isfinite(integral[modlmap>0]))
    kmask = 1 if kmask is None else kmask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num(1/integral)*kmask
    
def A_l(shape,wcs,feed_dict,estimator,XY,xmask=None,ymask=None,field_names=None,kmask=None):
    """
    Returns the normalization corresponding to a pre-defined mode-coupling estimator.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    estimator: str
        The name of a pre-defined mode-coupling estimator. 
        e.g. "hu_ok", "hdv" and "shear".
    XY: str
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.

    Returns
    -------

    Al : (Ny,Nx) ndarray
        The 2D normalization for the estimator.

    """
    
    f,F,Fr = get_mc_expressions(estimator,XY,field_names=field_names)
    return A_l_custom(shape,wcs,feed_dict,f,F,xmask=xmask,ymask=ymask,groups=_get_groups(estimator),kmask=kmask)

def N_l_from_A_l_optimal(shape,wcs,Al):
    modlmap = enmap.modlmap(shape,wcs)
    return Al * modlmap*(modlmap+1.)/4.    

def N_l_optimal(shape,wcs,feed_dict,estimator,XY,xmask=None,ymask=None,field_names=None,kmask=None):
    """
    Returns the 2D noise corresponding to a pre-defined mode-coupling estimator
    but assuming that it is optimal, i.e.
    Nl = A_L L^2 / 4

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    estimator: str
        The name of a pre-defined mode-coupling estimator. 
        e.g. "hu_ok", "hdv" and "shear".
    XY: str
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.

    Returns
    -------

    Nl : (Ny,Nx) ndarray
        The 2D noise for the estimator.

    """
    
    Al = A_l(shape,wcs,feed_dict,estimator,XY,xmask,ymask,field_names=field_names,kmask=kmask)
    modlmap = enmap.modlmap(shape,wcs)
    return N_l_from_A_l_optimal(shape,wcs,Al)

def N_l_optimal_custom(shape,wcs,feed_dict,f,F,xmask=None,ymask=None,groups=None,kmask=None):
    """
    Returns the 2D noise corresponding to a custom
    mode-coupling estimator but assuming that it is optimal, i.e.
    Nl = A_L L^2 / 4

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    f: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. If this is specified, the argument F is required, and the arguments
        estimator, XY and field_names are ignored.
    F: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter. See the Usage guide
        for details. 
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    xname: str,optional
        The name of the key in feed_dict where the X map in the XY quadratic estimator
        is stored. Defaults to X_l1.
    yname: str,optional
        The name of the key in feed_dict where the Y map in the XY quadratic estimator
        is stored. Defaults to Y_l2.
    groups: list,optional 
        Group all terms in the normalization calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    Nl : (Ny,Nx) ndarray
        The 2D noise for the estimator.


    """
    
    Al = A_l_custom(shape,wcs,feed_dict,f,F,xmask,ymask,groups=groups,kmask=kmask)
    return N_l_from_A_l_optimal(shape,wcs,Al)

def unnormalized_quadratic_estimator_custom(shape,wcs,feed_dict,F,xname='X_l1',yname='Y_l2',
                                            xmask=None,ymask=None,groups=None,physical_units=True):
    """
    Returns a normalized reconstruction corresponding to a custom
    mode-coupling estimator.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    F: :obj:`sympy.core.symbol.Symbol`
        A sympy expression containing the estimator filter. See the Usage guide
        for details. 
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    xname: str,optional
        The name of the key in feed_dict where the X map in the XY quadratic estimator
        is stored. Defaults to X_l1.
    yname: str,optional
        The name of the key in feed_dict where the Y map in the XY quadratic estimator
        is stored. Defaults to Y_l2.
    groups: list,optional 
        Group all terms in the reconstruction calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    krecon : (Ny,Nx) ndarray
        The normalized Fourier space reconstruction in physical units (not pixel units).

    See Also
    --------
    QE
        A class with which the slow normalization can be pre-calculated and repeated estimation
        can be performed on similar datasets.

    """
    
    res = integrate(shape,wcs,feed_dict,e(xname)*e(yname)*F/2,xmask=xmask,ymask=ymask,groups=groups,physical_units=physical_units)
    assert np.all(np.isfinite(res))
    return res

def unnormalized_quadratic_estimator(shape,wcs,feed_dict,estimator,XY,
                                     xname='X_l1',yname='Y_l2',field_names=None,xmask=None,ymask=None,physical_units=True):
    """
    Returns a normalized reconstruction corresponding to specified pre-defined
    mode-coupling estimator.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    estimator: str,optional
        The name of a pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    XY: str,optional
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    xname: str,optional
        The name of the key in feed_dict where the X map in the XY quadratic estimator
        is stored. Defaults to X_l1.
    yname: str,optional
        The name of the key in feed_dict where the Y map in the XY quadratic estimator
        is stored. Defaults to Y_l2.

    Returns
    -------

    krecon : (Ny,Nx) ndarray
        The normalized Fourier space reconstruction in physical units (not pixel units).

    See Also
    --------
    reconstruct
        Get the properly normalized quadratic estimator reconstruction.
    QE
        A class with which the slow normalization can be pre-calculated and repeated estimation
        can be performed on similar datasets.

    """

    
    f,F,Fr = get_mc_expressions(estimator,XY,field_names=field_names)
    return unnormalized_quadratic_estimator_custom(shape,wcs,feed_dict,F,xname=xname,yname=yname,xmask=xmask,ymask=ymask,groups=_get_groups(estimator,noise=False),physical_units=physical_units)


def reconstruct(shape,wcs,feed_dict,estimator=None,XY=None,
                f=None,F=None,xmask=None,ymask=None,
                field_names=None,norm_groups=None,est_groups=None,
                xname='X_l1',yname='Y_l2',kmask=None,physical_units=True):


    """
    Returns a normalized reconstruction corresponding to specified
    mode-coupling estimator.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays used in the normalization and
        reconstruction  calculation. When using pre-defined mode-coupling estimators, typical
        keys that must be present are 'uC_X_Y' and 'tC_X_Y', where X and Y
        depend on the requested estimator XY (see below). This feed_dict must
        also contain the keys with name xname and yname (see below), which 
        contain the 2D maps X and Y for the data from which the quadratic estimate
        is constructed.
    estimator: str,optional
        The name of a pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    XY: str,optional
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    f: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. If this is specified, the argument F is required, and the arguments
        estimator, XY and field_names are ignored.
    F: :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter. See the Usage guide
        for details. 
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more custom noise correlations.
    norm_groups: list,optional 
        Group all terms in the normalization such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.
    xname: str,optional
        The name of the key in feed_dict where the X map in the XY quadratic estimator
        is stored. Defaults to X_l1.
    yname: str,optional
        The name of the key in feed_dict where the Y map in the XY quadratic estimator
        is stored. Defaults to Y_l2.
    est_groups: list,optional 
        Group all terms in the reconstruction calclulation such that they have common factors 
        of the provided list of expressions to reduce the number of FFTs.

    Returns
    -------

    krecon : (Ny,Nx) ndarray
        The normalized Fourier space reconstruction in physical units (not pixel units).

    See Also
    --------
    QE
        A class with which the slow normalization can be pre-calculated and repeated estimation
        can be performed on similar datasets.

    """
    
    qe = QE(shape,wcs,feed_dict,estimator=estimator,XY=XY,
            f=f,F=F,xmask=xmask,ymask=ymask,
            field_names=field_names,groups=norm_groups,kmask=kmask)
    return qe.reconstruct(feed_dict,xname=xname,yname=yname,groups=est_groups,physical_units=physical_units)
    



def u1(ab):
    a,b = ab
    return e('uC_%s_%s_l1' % (a,b))
def u2(ab):
    a,b = ab
    return e('uC_%s_%s_l2' % (a,b))
def du1(ab):
    a,b = ab
    return e('duC_%s_%s_l1' % (a,b))
def du2(ab):
    a,b = ab
    return e('duC_%s_%s_l2' % (a,b))


def lensing_response_f(XY,rev=False,curl=False):

    """
    Returns the mode-coupling response f(l1,l2) for CMB lensing.

    Parameters
    ----------

    XY: str
        The XY pair for the requested estimator. This must belong to one
        of TT, EE, TE, ET, EB or TB.
    rev: boolean, optional
        Whether to swap l1 and l2. Defaults to False.

    Returns
    -------

    f : :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. 

    """
    
    cLdl1 = Lxl1 if curl else Ldl1
    cLdl2 = Lxl2 if curl else Ldl2
    
    iLdl1 = cLdl2 if rev else cLdl1
    iLdl2 = cLdl1 if rev else cLdl2
    def iu1(ab): return u2(ab) if rev else u1(ab)
    def iu2(ab): return u1(ab) if rev else u2(ab)
    if XY=='TT':
        f = iLdl1*iu1('TT')+iLdl2*iu2('TT')
    elif XY=='TE':
        f = iLdl1*cos2t12*iu1('TE')+iLdl2*iu2('TE')
    elif XY=='ET':
        f = iLdl2*iu2('TE')*cos2t12+iLdl1*iu1('TE')
    elif XY=='TB':
        f = iu1('TE')*sin2t12*iLdl1
    elif XY=='EE':
        f = (iLdl1*iu1('EE')+iLdl2*iu2('EE'))*cos2t12
    elif XY=='EB':
        f = iLdl1*iu1('EE')*sin2t12
    else:
        print(XY)
        raise ValueError
    return f


def rotation_response_f(XY,rev=False):

    """
    Returns the mode-coupling response f(l1,l2) for CMB rotation.

    Parameters
    ----------

    XY: str
        The XY pair for the requested estimator. This must belong to one
        of EE, TE, ET, EB or TB.
    rev: boolean, optional
        Whether to swap l1 and l2. Defaults to False.

    Returns
    -------

    f : :obj:`sympy.core.symbol.Symbol`
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details.

    """

    def iu1(ab): return u2(ab) if rev else u1(ab)
    def iu2(ab): return u1(ab) if rev else u2(ab)

    if XY=='TE':
        f = 2*iu1('TE')*sin2t12
    elif XY=='TB':
        f = 2*iu1('TE')*cos2t12
    elif XY=='EE':
        f = 2*(iu1('EE')-iu2('EE'))*sin2t12
    elif XY=='EB':
        f = 2*(iu1('EE')-iu2('BB'))*cos2t12
    elif XY=='BB':
        f = (iu1('BB')+iu2('BB'))*sin2t12
    else:
        print(XY)
        raise ValueError
    return f

def get_mc_expressions(estimator,XY,field_names=None,estimator_to_harden='hu_ok'):
    """
    Pre-defined mode coupling expressions.
    Returns f(l1,l2), F(l1,l2), F(l2,l1).
    If a list field_names is provided containing two strings,
    then "total power" spectra are customized to potentially
    be different and feed_dict will need to have more values.

    Parameters
    ----------

    estimator: str
        The name of a pre-defined mode-coupling estimator. If this is provided,
        the argument XY is required and the arguments f, F, and groups are ignored.
        e.g. "hu_ok", "hdv" and "shear".
    XY: str
        The XY pair for the requested estimator. Typical examples include "TT" and "EB".
    field_names: 2 element list, optional
        When a pre-defined mode-coupling estimator is used, providing a list field_names
        modifies the total power spectra variable names that feed_dict expects. 
        Typically, names like "tC_T_T" and "tC_T_E" are expected. But if field_names
        is ["E1","E2"] for example, variable names like ``tC_E1_T_E1_T``, ``tC_E2_T_E2_T``,
        ``tC_E1_T_E2_T``, ``tC_E2_T_E1_T`` are expected to be present in feed_dict. This
        allows for more general noise correlations.

    Returns
    -------

    f : :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the mode-coupling response. See the Usage guide
        for details. 
    F : :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter. 
    Fr : :obj:`sympy.core.symbol.Symbol` , optional
        A sympy expression containing the estimator filter but with l1 and l2 swapped. 


    """
    
    estimator = estimator.lower()
    X,Y = XY
    XX = X+X
    YY = Y+Y
    f1,f2 = field_names if field_names is not None else (None,None)

    # NOTE: previously, this used cross(f1,f2) for t1 and t2
    # I've switched it to t1 = cross(f1,f1) and t2 = cross(f2,f2)
    # to allow for different filters for the two fields.
    def t1(ab):
        a,b = ab
        return e(cross_names(a,b,f1,f1)+"_l1")
    def t2(ab):
        a,b = ab
        return e(cross_names(a,b,f2,f2)+"_l2")
    
    hus = ['hu_ok_curl','hdv_curl','hu_ok','hdv']
    curls = ['hu_ok_curl','hdv_curl']
    curl = estimator in curls
    cLdl1 = Lxl1 if curl else Ldl1
    cLdl2 = Lxl2 if curl else Ldl2
    if estimator in hus: # Hu, Okamoto 2001 # Hu, DeDeo, Vale 2007
        f = lensing_response_f(XY,rev=False,curl=curl)
        if estimator in ['hu_ok','hu_ok_curl']:
            fr = lensing_response_f(XY,rev=True,curl=curl)
            if XY in ['TT','EE']:
                F = f / 2 / t1(XY) / t2(XY)
                Fr = fr / 2 / t2(XY) / t1(XY)
            elif XY in ['TB','EB']:
                F = f  / t1(XX) / t2(YY)
                Fr = fr  / t2(XX) / t1(YY)
            elif XY=='TE':
                # this filter is not separable
                #(tCl1['EE']*tCl2['TT']*f - tCl1['TE']*tCl2['TE']*frev)/(tCl1['TT']*tCl2['EE']*tCl1['EE']*tCl2['TT']-(tCl1['TE']*tCl2['TE'])**2.)
                # this approximation is
                F = (t1('EE')*t2('TT')*f - t1('TE')*t2('TE')*fr)/(t1('TT')*t2('EE')*t1('EE')*t2('TT'))
                Fr = (t2('EE')*t1('TT')*fr - t2('TE')*t1('TE')*f)/(t2('TT')*t1('EE')*t2('EE')*t1('TT'))
        elif estimator in ['hdv','hdv_curl']:
            Fp = cLdl1/t1(X+X)/t2(Y+Y)
            Fpr = cLdl2/t2(X+X)/t1(Y+Y)
            if Y=='T':
                F = Fp * u1(X+Y)
                Fr = Fpr * u2(X+Y)
            if Y=='E':
                F = Fp * u1(X+Y) * cos2t12
                Fr = Fpr * u2(X+Y) * cos2t12
            if Y=='B':
                F = Fp * u1(X+'E') * sin2t12
                Fr = Fpr * u2(X+'E') * sin2t12

    elif estimator=='shear': # Schaan, Ferraro 2018
        assert XY=="TT", "Shear estimator only implemented for TT."
        f = cLdl2*u2('TT') + cLdl1*u1('TT')
        fr = cLdl1*u1('TT')
        cos2theta = ((2*(cLdl1)**2)/L**2/l1**2) - 1
        cos2theta_rev = ((2*(cLdl2)**2)/L**2/l2**2) - 1
        F = cos2theta * u1('TT') * du1('TT')/2/t1('TT')/t1('TT')
        Fr = cos2theta_rev * u2('TT') * du2('TT')/2/t2('TT')/t2('TT')

    elif estimator=='src':
        f = e('pc_T_T_l1')*e('pc_T_T_l2')
        F = f / t1(XY) / t2(XY) / 2
        fr = f
        Fr = F
    elif estimator=='src-hardened':
        """ Osborne et. al. point source hardening
        This gives you the MC expressions for the source
        hardened lensing estimator.
        You have to provide the following during the
        calculation apart from the usual spectra: 
        (1) pc_T_T, the Fourier space profile
        of the source, which is 1 for a point source.
        (2) Asrc_src_L : 1 / the source estimator response to sources
        (3) Aphi_src_L : 1 / the lens estimator response to sources
        """
        assert XY=="TT", "BH only implemented for TT."
        f_phi,F_phi,_ = get_mc_expressions(estimator_to_harden,XY,field_names=field_names)
        f_src,_,_ = get_mc_expressions('src',XY,field_names=field_names)
        A_src_src = e('Asrc_src_L')
        A_phi_src = e('Aphi_src_L')
        f = f_phi - A_src_src / A_phi_src * f_src
        F = f / t1(XY) / t2(XY) / 2
        fr = f
        Fr = F
    elif estimator=='mask': # Namikawa et. al. mask bias hardening
        f = - t1('TT') - t2('TT')
        fr = f
        F = f / t1(XY) / t2(XY) / 2
        fr = f
        Fr = F
    elif estimator=='mask-hardened':
        """ Namikawa et. al. mask hardening
        This gives you the MC expressions for the mask
        hardened lensing estimator.
        You have to provide the following during the
        calculation apart from the usual spectra: 
        (1) Amsk_msk_L : 1 / the mask estimator response to masks
        (2) Aphi_msk_L : 1 / the lens estimator response to masks
        """
        assert XY=="TT", "BH only implemented for TT."
        f_phi,F_phi,_ = get_mc_expressions(estimator_to_harden,XY,field_names=field_names)
        f_msk,_,_ = get_mc_expressions('mask',XY,field_names=field_names)
        A_msk_msk = e('Amask_mask_L')
        A_phi_msk = e('Aphi_mask_L')
        f = f_phi - A_msk_msk / A_phi_msk * f_msk
        F = f / t1(XY) / t2(XY) / 2
        fr = f
        Fr = F
    elif estimator=='rot':  # Yadav et. al. 2009
        f = rotation_response_f(XY,rev=False)
        fr = rotation_response_f(XY,rev=True)
        if XY in ['EE','BB']:
            F = f / 2 / t1(XY) / t2(XY)
            Fr = fr / 2 / t2(XY) / t1(XY)
        elif XY in ['TB','EB']:
            F = f / t1(XX) / t2(YY)
            Fr = fr / t2(XX) / t1(YY)
        elif XY=='TE':
            F = (t1('EE')*t2('TT')*f - t1('TE')*t2('TE')*fr)/(t1('TT')*t2('EE')*t1('EE')*t2('TT'))
            Fr = (t2('EE')*t1('TT')*fr - t2('TE')*t1('TE')*f)/(t2('TT')*t1('EE')*t2('EE')*t1('TT'))

    return f,F,Fr
