import sympy
from sympy import Symbol
import numpy as np
import warnings
import contextlib
import os,sys
import warnings

known_zeros = [('E','B'), ('B','E'), ('T','B'), ('B','T')]
def _handle_missing_keys(t1,t2,comp1,comp2):
    if (comp1,comp2) in known_zeros:
        warnings.warn("Assuming " + t1 + " is zero. Provide a value for it in feed_dict if not!")
        return 0
    else:
        raise KeyError('Neither ',t1, ' nor ', t2, ' were found in feed_dict.')


def get_feed(feed_dict,key):
    # Instead of
    # return feed_dict[k]
    # this allows symmetrized keys.
    
    if "_" not in key: return feed_dict[key]
    
    # Handle l1 and l2 symbols
    if key[-3:]=='_l1' or key[-3:]=='_l2':
        skey = key[:-3]
        suff = key[-3:]
    else:
        skey = key
        suff = ""

    # Split into components
    components = skey.split('_')
    fcomp = components[0]
    if len(components)==3:
        # Non-field-name-case
        comp1 = components[1]
        comp2 = components[2]
    elif len(components)==5:
        # Field-name-case
        comp1 = components[1] + "_" + components[2]
        comp2 = components[3] + "_" + components[4]
    else:
        raise KeyError('This key ',key, ' has an unsupported number of components. Excluding the l1/l2 tag, there should be either three (no field-name case) or five (field-name case).')

    t1 = fcomp + "_" + comp1 + "_" + comp2 + suff
    t2 = fcomp + "_" + comp2 + "_" + comp1 + suff

    try:
        return feed_dict[t1]
    except:
        try:
            return feed_dict[t2]
        except:
            return _handle_missing_keys(t1,t2,comp1,comp2)
    

def evaluate(symbolic_term,feed_dict):
    """
    Convert a symbolic term into a numerical result by using values for
    the symbols from a dictionary.

    symbolic_term: sympy expression
    feed_dict: dictionary mapping names of symbols to numpy arrays
    """
    symbols = list(symbolic_term.free_symbols)
    func_term = sympy.lambdify(symbols,symbolic_term,dummify=False)
    # func_term accepts as keyword arguments strings that are in symbols
    # We need to extract a dict from feed_dict that only has the keywords
    # in symbols
    varstrs = [str(x) for x in symbols]
    edict = {k: get_feed(feed_dict,k) for k in varstrs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaled = np.nan_to_num(func_term(**edict))
    return evaled

def substitute_trig(l1x,l1y,l2x,l2y,l1,l2):
    # Expand cos(2theta_12) and sin(2theta_12) in terms of l1x,l2x,l1y,l2y,l1,l2
    phi1 = Symbol('phi1')
    phi2 = Symbol('phi2')
    cos2t12 = sympy.cos(2*(phi1-phi2))
    sin2t12 = sympy.sin(2*(phi1-phi2))
    simpcos = sympy.expand_trig(cos2t12)
    simpsin = sympy.expand_trig(sin2t12)
    cos2t12 = sympy.expand(sympy.simplify(simpcos.subs([(sympy.cos(phi1),l1x/l1),(sympy.cos(phi2),l2x/l2),
                            (sympy.sin(phi1),l1y/l1),(sympy.sin(phi2),l2y/l2)])))
    sin2t12 = sympy.expand(sympy.simplify(simpsin.subs([(sympy.cos(phi1),l1x/l1),(sympy.cos(phi2),l2x/l2),
                            (sympy.sin(phi1),l1y/l1),(sympy.sin(phi2),l2y/l2)])))
    return cos2t12,sin2t12

