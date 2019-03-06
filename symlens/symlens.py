from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from pixell import fft as efft, enmap
import os,sys
from . import _core

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

# Built-in special symbols
l1x = Symbol('l1x')
l1y = Symbol('l1y')
l2x = Symbol('l2x')
l2y = Symbol('l2y')
l1 = Symbol('l1')
l2 = Symbol('l2')
Lx = Symbol('Lx')
Ly = Symbol('Ly')
L = Symbol('L')
Ldl1 = (Lx*l1x+Ly*l1y)
Ldl2 = (Lx*l2x+Ly*l2y)
l1dl2 = (l1x*l2x+l2x*l2y)

# More built-in special symbols
cos2t12,sin2t12 = _core.substitute_trig(l1x,l1y,l2x,l2y,l1,l2)

# Custom symbol wrapper
def e(symbol):
    # TODO: add exceptions if symbol doesn't correspond to key structure
    return Symbol(symbol)

ifft = lambda x: efft.ifft(x,axes=[-2,-1],normalize=True)
fft = lambda x: efft.fft(x,axes=[-2,-1])
evaluate = _core.evaluate

def factorize_2d_convolution_integral(expr,l1funcs=None,l2funcs=None,groups=None,validate=True):
    """Reduce a sympy expression of variables l1x,l1y,l2x,l2y,l1,l2 into a sum of 
    products of factors that depend only on vec(l1) and vec(l2) and neither, each. If the expression
    appeared as the integrand in an integral over vec(l1), where 
    vec(l2) = vec(L) - vec(l1) then this reduction allows one to evaluate the 
    integral as a function of vec(L) using FFTs instead of as a convolution.

    expr: The full Sympy expression to reduce to sum of products of functions of l1 and l2
    l1funcs: list of symbols that are functions of l1
    l2funcs: list of symbols that are functions of l2
    groups: (optional) group all terms such that they have common factors of the provided list of 
    expressions to reduce the number of FFTs
    validate: Whether to make sure the final expression and the original agree    
    """

    # Generic message if validation fails
    val_fail_message = "Validation failed. This expression is likely not reducible to FFT form."
    # Get the 2D convolution cartesian variables
    # l1x,l1y,l2x,l2y,l1,l2 = get_ells()
    if l1funcs is None: l1funcs = []
    if l2funcs is None: l2funcs = []
    if l1x not in l1funcs: l1funcs.append(l1x)
    if l1y not in l1funcs: l1funcs.append(l1y) 
    if l1 not in l1funcs: l1funcs.append(l1)
    if l2x not in l2funcs: l2funcs.append(l2x)
    if l2y not in l2funcs: l2funcs.append(l2y) 
    if l2 not in l2funcs: l2funcs.append(l2)
    Lx = Symbol('Lx')
    Ly = Symbol('Ly')
    L = Symbol('L')
    ofuncs1 = set(l1funcs) - set([l1x,l1y,l1])
    ofuncs2 = set(l2funcs) - set([l2x,l2y,l2])

   
    # List to collect terms in
    terms = []
    if validate: prodterms = []
    # We must expand the expression so that the top-level operation is Add, i.e. it looks like
    # A + B + C + ...
    expr = sympy.expand( expr )
    # What is the top-level operation?
    op = expr.func
    if op is sympy.Add:
        arguments = expr.args # If Add, then we have multiple terms
    else:
        arguments = [expr] # If not Add, then we have a single term
    # Let's factorize each term
    unique_l1s = []
    unique_l2s = []
    
    def homogenize(inexp):
        outexp = inexp.subs([[l1x,Lx],[l2x,Lx],[l1y,Ly],[l2y,Ly],[l1,L],[l2,L]])
        ofuncs = ofuncs1.union(ofuncs2)
        for ofunc in ofuncs:
            nfunc = Symbol(str(ofunc)[:-3])
            outexp = outexp.subs(ofunc,nfunc)
        return outexp

    
    def get_group(inexp):
        if groups is None: return 0
        found = False
        d = Symbol('dummy')
        for i,group in enumerate(groups):
            s = inexp.subs(group,d)
            if not((s/d).has(d)):
                if found:
                    print(s,group)
                    raise ValueError("Groups don't seem to be mutually exclusive.")
                index = i
                found = True
        if not(found):
            raise ValueError("Couldn't associate a group")
        return index


    ogroups = [] if not(groups is None) else None
    ogroup_weights = [] if not(groups is None) else None
    ogroup_symbols = sympy.ones(len(groups),1) if not(groups is None) else None
    for k,arg in enumerate(arguments):
        temp, ll1terms = arg.as_independent(*l1funcs, as_Mul=True)
        loterms, ll2terms = temp.as_independent(*l2funcs, as_Mul=True)

        
        if any([x==0 for x in [ll1terms,ll2terms,loterms]]): continue
        # Group ffts
        if groups is not None:
            gindex = get_group(loterms)
            ogroups.append(gindex)
            fsyms = loterms.free_symbols
            ocoeff = loterms.evalf(subs=dict(zip(fsyms,[1]*len(fsyms))))
            ogroup_weights.append( float(ocoeff) )
            if ogroup_symbols[gindex]==1:
                ogroup_symbols[gindex] = loterms/ocoeff
            else:
                assert ogroup_symbols[gindex]==loterms/ocoeff, "Error validating group membership"
        
        vdict = {}
        vdict['l1'] = ll1terms
        vdict['l2'] = ll2terms
        tdict = {}
        tdict['l1'] = homogenize(vdict['l1'])
        tdict['l2'] = homogenize(vdict['l2'])

        if not(tdict['l1'] in unique_l1s):
            unique_l1s.append(tdict['l1'])
        tdict['l1index'] = unique_l1s.index(tdict['l1'])

        if not(tdict['l2'] in unique_l2s):
            unique_l2s.append(tdict['l2'])
        tdict['l2index'] = unique_l2s.index(tdict['l2'])

        
        
        vdict['other'] = loterms
        tdict['other'] = loterms
        terms.append(tdict)
        # Validate!
        if validate:
            # Check that all the factors of this term do give back the original term
            products = sympy.Mul(vdict['l1'])*sympy.Mul(vdict['l2'])*sympy.Mul(vdict['other'])
            assert sympy.simplify(products-arg)==0, val_fail_message
            prodterms.append(products)
            # Check that the factors don't include symbols they shouldn't
            assert all([not(vdict['l1'].has(x)) for x in l2funcs]), val_fail_message
            assert all([not(vdict['l2'].has(x)) for x in l1funcs]), val_fail_message
            assert all([not(vdict['other'].has(x)) for x in l1funcs]), val_fail_message
            assert all([not(vdict['other'].has(x)) for x in l2funcs]), val_fail_message
    # Check that the sum of products of final form matches original expression
    if validate:
        fexpr = sympy.Add(*prodterms)
        assert sympy.simplify(expr-fexpr)==0, val_fail_message
    return terms,unique_l1s,unique_l2s,ogroups,ogroup_weights,ogroup_symbols



def integrate(shape,wcs,feed_dict,expr,xmask=None,ymask=None,cache=True,validate=True,groups=None,pixel_units=False):
    """
    Integrate an arbitrary expression after factorizing it.

    shape: Shape of numpy array (geometry of map)
    wcs: WCS (geometry of map)
    feed_dict: mapping from names of custom symbols to numpy arrays
    expr: a sympy expression containing recognized symbols (see docs)
    xmask: Fourier space 2D mask for the l1 part of the integral
    ymask: Fourier space 2D mask for the l2 part of the integral
    cache: Whether to store in memory and reuse repeated terms
    validate: Whether to make sure the final expression and the original agree    
    groups: (optional) group all terms such that they have common factors of the provided list of 
    expressions to reduce the number of FFTs
    pixel_units: whether the input is in pixel units or not

    """
    # Geometry
    modlmap = enmap.modlmap(shape,wcs)
    lymap,lxmap = enmap.lmap(shape,wcs)
    pixarea = np.prod(enmap.pixshape(shape,wcs))
    feed_dict['L'] = modlmap
    feed_dict['Ly'] = lymap
    feed_dict['Lx'] = lxmap
    shape = shape[-2:]
    ones = np.ones(shape,dtype=np.float32)
    val = 0.
    if xmask is None: xmask = ones
    if ymask is None: ymask = ones

    # Expression
    syms = expr.free_symbols
    l1funcs = []
    l2funcs = []
    for sym in syms:
        strsym = str(sym)
        if   strsym[-3:]=="_l1": l1funcs.append(sym)
        elif strsym[-3:]=="_l2": l2funcs.append(sym)

    integrands,ul1s,ul2s, \
        ogroups,ogroup_weights, \
        ogroup_symbols = factorize_2d_convolution_integral(expr,l1funcs=l1funcs,l2funcs=l2funcs,
                                                                         validate=validate,groups=groups)

    def _fft(x): return fft(x+0j)
    def _ifft(x): return ifft(x+0j)

    if cache:
        cached_u1s = []
        cached_u2s = []
        for u1 in ul1s:
            l12d = evaluate(u1,feed_dict)*ones
            cached_u1s.append(_ifft(l12d*xmask))
        for u2 in ul2s:
            l22d = evaluate(u2,feed_dict)*ones
            cached_u2s.append(_ifft(l22d*ymask))


    # For each term, the index of which group it belongs to  

    def get_l1l2(term):
        if cache:
            ifft1 = cached_u1s[term['l1index']]
            ifft2 = cached_u2s[term['l2index']]
        else:
            l12d = evaluate(term['l1'],feed_dict)*ones
            ifft1 = _ifft(l12d*xmask)
            l22d = evaluate(term['l2'],feed_dict)*ones
            ifft2 = _ifft(l22d*ymask)
        return ifft1,ifft2


    if ogroups is None:    
        for i,term in enumerate(integrands):
            ifft1,ifft2 = get_l1l2(term)
            ot2d = evaluate(term['other'],feed_dict)*ones
            ffft = _fft(ifft1*ifft2)
            val += ot2d*ffft
    else:
        vals = np.zeros((len(ogroup_symbols),)+shape,dtype=np.float32)+0j
        for i,term in enumerate(integrands):
            ifft1,ifft2 = get_l1l2(term)
            gindex = ogroups[i]
            vals[gindex,...] += ifft1*ifft2 *ogroup_weights[i]
        for i,group in enumerate(ogroup_symbols):
            ot2d = evaluate(ogroup_symbols[i],feed_dict)*ones            
            ffft = _fft(vals[i,...])
            val += ot2d*ffft
     
    mul = 1 if pixel_units else 1./pixarea
    return val * mul


class QE(object):
    def __init__(self,shape,wcs,feed_dict,estimator=None,XY=None,
                 f=None,F=None,xmask=None,ymask=None,
                 field_names=None,groups=None):
        if f is not None:
            assert F is not None
            assert estimator is None
            assert XY is None
            self.Al = A_l_general(shape,wcs,feed_dict,f,F,xmask=xmask,ymask=ymask,groups=groups)
            self.F = F
            self.general = True
        else:
            assert F is None
            self.Al = A_l(shape,wcs,feed_dict,estimator,XY,xmask=xmask,ymask=ymask,field_names=field_names)
            self.estimator = estimator
            self.XY = XY
            self.field_names = field_names
            self.general = False
        self.shape = shape
        self.wcs = wcs
        self.xmask = xmask
        self.ymask = ymask
        
    def reconstruct(self,feed_dict,xname='X_l1',yname='Y_l2',groups=None):
        if self.general:
            uqe = unnormalized_quadratic_estimator_general(self.shape,self.wcs,feed_dict,
                                                           self.F,xname=xname,yname=yname,
                                                           xmask=self.xmask,ymask=self.ymask,
                                                           groups=groups)
        else:
            uqe = unnormalized_quadratic_estimator(self.shape,self.wcs,feed_dict,
                                                   self.estimator,self.XY,
                                                   xname=xname,yname=yname,
                                                   field_names=self.field_names,
                                                   xmask=self.xmask,ymask=self.ymask)
        return self.Al * uqe

def _cross_names(x,y,fn1,fn2):
    if fn1 is None: e1 = ""
    else:
        assert "_" not in fn1, "Field names cannot have underscores. Sorry! Use a hyphen instead."
        e1 = fn1+"_"
    if fn2 is None: e2 = ""
    else:
        assert "_" not in fn2, "Field names cannot have underscores. Sorry! Use a hyphen instead."
        e2 = fn2+"_"
    return 'tC_%s%s_%s%s' % (e1,x,e2,y)

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
    
    
def cross_integral_general(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                           xmask=None,ymask=None,
                           field_names_alpha=None,field_names_beta=None,groups=None):
    """
    est_alpha = ab
    est_beta = cd
    """
    a,b = alpha_XY
    c,d = beta_XY
    fnalpha1,fnalpha2 = field_names_alpha if field_names_alpha is not None else (None,None)
    fnbeta1,fnbeta2 = field_names_beta if field_names_beta is not None else (None,None)
    tCac_l1 = e(_cross_names(a,c,fnalpha1,fnbeta1) + "_l1")
    tCbd_l2 = e(_cross_names(b,d,fnalpha2,fnbeta2) + "_l2")
    tCad_l1 = e(_cross_names(a,d,fnalpha1,fnbeta2) + "_l1")
    tCbc_l2 = e(_cross_names(b,c,fnalpha2,fnbeta1) + "_l2")
    expr = Falpha*(Fbeta*tCac_l1*tCbd_l2+Fbeta_rev*tCad_l1*tCbc_l2)
    integral = integrate(shape,wcs,feed_dict,expr,xmask=xmask,ymask=xmask,groups=groups).real
    return integral

def N_l_cross_general(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                      xmask=None,ymask=None,
                      field_names_alpha=None,field_names_beta=None,
                      falpha=None,fbeta=None,Aalpha=None,Abeta=None,groups=None):
    cross_integral = cross_integral_general(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                                            xmask=xmask,ymask=xmask,
                                            field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,groups=groups)
    if Aalpha is None: Aalpha = A_l_general(shape,wcs,feed_dict,falpha,Falpha,xmask=xmask,ymask=ymask)
    if Abeta is None: Abeta = A_l_general(shape,wcs,feed_dict,fbeta,Fbeta,xmask=xmask,ymask=ymask)
    return 0.25 * Aalpha * Abeta * cross_integral
    
def N_l_cross(shape,wcs,feed_dict,alpha_estimator,alpha_XY,beta_estimator,beta_XY,
              xmask=None,ymask=None,
              Aalpha=None,Abeta=None,field_names_alpha=None,field_names_beta=None):
    falpha,Falpha,Falpha_rev = get_mc_expressions(alpha_estimator,alpha_XY,field_names=field_names_alpha)
    fbeta,Fbeta,Fbeta_rev = get_mc_expressions(alpha_estimator,alpha_XY,field_names=field_names_beta)
    return N_l_cross_general(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,
                             xmask=xmask,ymask=ymask,
                             field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,
                             falpha=falpha,fbeta=fbeta,Aalpha=Aalpha,Abeta=Abeta,groups=_get_groups(alpha_estimator,beta_estimator))

def N_l(shape,wcs,feed_dict,estimator,XY,
        xmask=None,ymask=None,
        Al=None,field_names=None):
    falpha,Falpha,Falpha_rev = get_mc_expressions(estimator,XY,field_names=field_names)
    return N_l_cross_general(shape,wcs,feed_dict,XY,XY,Falpha,Falpha,Falpha_rev,
                             xmask=xmask,ymask=ymask,
                             field_names_alpha=field_names,field_names_beta=field_names,
                             falpha=falpha,fbeta=falpha,Aalpha=Al,Abeta=Al,groups=_get_groups(estimator))

    
def A_l_general(shape,wcs,feed_dict,f,F,xmask=None,ymask=None,groups=None):
    integral = integrate(shape,wcs,feed_dict,f*F/L/L,xmask=xmask,ymask=xmask,groups=groups).real
    return 1/integral
    
def A_l(shape,wcs,feed_dict,estimator,XY,xmask=None,ymask=None,field_names=None):
    f,F,Fr = get_mc_expressions(estimator,XY,field_names=field_names)
    return A_l_general(shape,wcs,feed_dict,f,F,xmask=xmask,ymask=ymask,groups=_get_groups(estimator))

def N_l_from_A_l_optimal(shape,wcs,Al):
    modlmap = enmap.modlmap(shape,wcs)
    return A_l * modlmap**2./4.    

def N_l_optimal(shape,wcs,feed_dict,estimator,XY,xmask=None,ymask=None,field_names=None):
    Al = A_l(shape,wcs,feed_dict,estimator,XY,xmask,ymask,field_names=field_names)
    modlmap = enmap.modlmap(shape,wcs)
    return N_l_from_A_l_optimal(shape,wcs,Al)

def N_l_optimal_general(shape,wcs,feed_dict,f,F,xmask=None,ymask=None,groups=None):
    Al = A_l_general(shape,wcs,feed_dict,f,F,xmask,ymask,groups=groups)
    return N_l_from_A_l_optimal(shape,wcs,Al)

def unnormalized_quadratic_estimator_general(shape,wcs,feed_dict,F,xname='X_l1',yname='Y_l2',xmask=None,ymask=None,groups=None):
    return integrate(shape,wcs,feed_dict,e(xname)*e(yname)*F/2,xmask=xmask,ymask=xmask,groups=groups,pixel_units=True)

def unnormalized_quadratic_estimator(shape,wcs,feed_dict,estimator,XY,xname='X_l1',yname='Y_l2',field_names=None,xmask=None,ymask=None):
    f,F,Fr = get_mc_expressions(estimator,XY,field_names=field_names)
    return unnormalized_quadratic_estimator_general(shape,wcs,feed_dict,F,xname=xname,yname=yname,xmask=xmask,ymask=ymask,groups=_get_groups(estimator,noise=False))


def reconstruct(shape,wcs,feed_dict,estimator=None,XY=None,
                f=None,F=None,xmask=None,ymask=None,
                field_names=None,norm_groups=None,est_groups=None,
                xname='X_l1',yname='Y_l2'):
    qe = QE(shape,wcs,feed_dict,estimator=estimator,XY=XY,
            f=f,F=F,xmask=xmask,ymask=ymask,
            field_names=field_names,groups=norm_groups)
    return qe.reconstruct(feed_dict,xname=xname,yname=yname,groups=est_groups)
    


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


def lensing_response_f(XY,rev=False):
    iLdl1 = Ldl2 if rev else Ldl1
    iLdl2 = Ldl1 if rev else Ldl2
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
    return f

def get_mc_expressions(estimator,XY,field_names=None):
    """
    Pre-defined mode coupling expressions.
    Returns f(l1,l2), F(l1,l2), F(l2,l1)
    If a list field_names is provided containing two strings,
    then "total power" spectra are generalized to potentially
    be different and feed_dict will need to have more values.
    """
    
    estimator = estimator.lower()
    X,Y = XY
    XX = X+X
    YY = Y+Y
    f1,f2 = field_names if field_names is not None else (None,None)

    def t1(ab):
        a,b = ab
        return e(_cross_names(a,b,f1,f2)+"_l1")
    def t2(ab):
        a,b = ab
        return e(_cross_names(a,b,f1,f2)+"_l2")
    
    
    if estimator=='hu_ok' or estimator=='hdv': # Hu, Okamoto 2001 # Hu, DeDeo, Vale 2007
        f = lensing_response_f(XY,rev=False)
        if estimator=='hu_ok':
            fr = lensing_response_f(XY,rev=True)
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
        elif estimator=='hdv':
            Fp = Ldl1/t1(X+X)/t2(Y+Y)
            Fpr = Ldl2/t2(X+X)/t1(Y+Y)
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
        f = Ldl2*u2('TT')
        fr = Ldl1*u1('TT')
        cos2theta = ((2*(Ldl1)**2)/L**2/l1**2) - 1
        cos2theta_rev = ((2*(Ldl2)**2)/L**2/l2**2) - 1
        F = cos2theta * u1('TT') * du1('TT')/2/t1('TT')/t1('TT') 
        Fr = cos2theta_rev * u2('TT') * du2('TT')/2/t2('TT')/t2('TT')  

    return f,F,Fr
        
