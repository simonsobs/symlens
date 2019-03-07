from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy
from pixell import fft as efft, enmap
import os,sys
from . import _helpers

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

# Built-in special symbols
l1x = Symbol('l1x') # \vec{l}_1x
l1y = Symbol('l1y') # \vec{l}_1y
l2x = Symbol('l2x') # \vec{l}_2x
l2y = Symbol('l2y') # \vec{l}_2y
l1 = Symbol('l1') # |\vec{l}_1|
l2 = Symbol('l2') # |\vec{l}_2|
Lx = Symbol('Lx') # \vec{L}_x
Ly = Symbol('Ly') # \vec{L}_y
L = Symbol('L') # |\vec{L}|
Ldl1 = (Lx*l1x+Ly*l1y) # \vec{L}.\vec{l}_1
Ldl2 = (Lx*l2x+Ly*l2y) # \vec{L}.\vec{l}_2
Lxl1 = (Ly*l1x-Lx*l1y) # \vec{L} x \vec{l}_1 for curl
Lxl2 = (Ly*l2x-Lx*l2y) # \vec{L} x \vec{l}_2 for curl

# More built-in special symbols
# cos(2\theta_{12}), sin(2\theta_{12}) for polarization
cos2t12,sin2t12 = _helpers.substitute_trig(l1x,l1y,l2x,l2y,l1,l2)

# Custom symbol wrapper
def e(symbol):
    # TODO: add exceptions if symbol doesn't correspond to key structure
    return Symbol(symbol)

ifft = lambda x: efft.ifft(x,axes=[-2,-1],normalize=True)
fft = lambda x: efft.fft(x,axes=[-2,-1])
evaluate = _helpers.evaluate

def factorize_2d_convolution_integral(expr,l1funcs=None,l2funcs=None,groups=None,validate=True):
    """Reduce a sympy expression of variables l1x,l1y,l2x,l2y,l1,l2 into a sum of 
    products of factors that depend only on vec(l1) and vec(l2) and neither, each. If the expression
    appeared as the integrand in an integral over vec(l1), where 
    vec(l2) = vec(L) - vec(l1) then this reduction allows one to evaluate the 
    integral as a function of vec(L) using FFTs instead of as a convolution.

    Parameters
    ----------

    expr: :obj:`sympy.core.symbol.Symbol` 
        The full Sympy expression to reduce to sum of products of functions of l1 and l2.

    l1funcs: list
        List of symbols that are functions of l1
    l2funcs: list
        List of symbols that are functions of l2
    groups: list,optional 
        Group all terms such that they have common factors of the provided list of 
        expressions to reduce the number of FFTs.
    validate: boolean,optional
        Whether to check that the final expression and the original agree. Defaults to True.

    Returns
    -------

    terms
    unique_l1s
    unique_l2s
    ogroups
    ogroup_weights
    ogroup_symbols

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

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    feed_dict: dict
        Mapping from names of custom symbols to numpy arrays.
    expr: :obj:`sympy.core.symbol.Symbol` 
        A sympy expression containing recognized symbols (see docs)
    xmask: (Ny,Nx) ndarray,optional
        Fourier space 2D mask for the l1 part of the integral. Defaults to ones.
    ymask:  (Ny,Nx) ndarray, optional
        Fourier space 2D mask for the l2 part of the integral. Defaults to ones.
    cache: boolean, optional
        Whether to store in memory and reuse repeated terms. Defaults to true.
    validate: boolean,optional
        Whether to check that the final expression and the original agree. Defaults to True.
    groups: list,optional 
        Group all terms such that they have common factors of the provided list of 
        expressions to reduce the number of FFTs.
    pixel_units: boolean,optional
        Whether the input is in pixel units or not.

    Returns
    -------

    result : (Ny,Nx) ndarray
        The numerical result of the integration of the expression after factorization.

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



