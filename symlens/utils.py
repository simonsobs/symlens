import numpy as np
from scipy.interpolate import interp1d
from pixell import enmap,utils

def mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = None):
    """Produce a Fourier space mask.

    Parameters
    ----------

    shape : tuple
        The shape of the array for the geometry of the footprint. Typically 
        (...,Ny,Nx) for Ny pixels in the y-direction and Nx in the x-direction.
    wcs : :obj:`astropy.wcs.wcs.WCS`
        The wcs object completing the specification of the geometry of the footprint.
    lxcut : int, optional
        The width of a band in number of Fourier pixels to be masked in the lx direction.
        Default is no masking in this band.
    lycut : int, optional
        The width of a band in number of Fourier pixels to be masked in the ly direction.
        Default is no masking in this band.
    lmin : int, optional
        The radial distance in Fourier space below which all Fourier pixels are masked.
        Default is no masking.
    lmax : int, optional
        The radial distance in Fourier space above which all Fourier pixels are masked.
        Default is no masking.

    Returns
    -------

    output : (Ny,Nx) ndarray
        A 2D array containing the Fourier space mask.

    """
    output = np.ones(shape[-2:], dtype = int)
    if (lmin is not None) or (lmax is not None): modlmap = enmap.modlmap(shape, wcs)
    if (lxcut is not None) or (lycut is not None): ly, lx = enmap.laxes(shape, wcs, oversample=1)
    if lmin is not None:
        output[np.where(modlmap <= lmin)] = 0
    if lmax is not None:
        output[np.where(modlmap >= lmax)] = 0
    if lxcut is not None:
        output[:,np.where(np.abs(lx) < lxcut)] = 0
    if lycut is not None:
        output[np.where(np.abs(ly) < lycut),:] = 0
    return output


def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    """Return a function that interpolates (x,y). This wraps around 
    scipy.interpolate.interp1d but by defaulting to zero filling outside bounds.

    Docstring copied from scipy. Interpolate a 1-D function.
    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.
    Note that calling `interp1d` with NaNs present in input values results in
    undefined behaviour.
    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
        refer to a spline interpolation of zeroth, first, second or third
        order; 'previous' and 'next' simply return the previous or next value
        of the point) or as an integer specifying the order of the spline
        interpolator to use.
        Default is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        False by default.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is zero. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``.
          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.
          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.
    Methods
    -------
    __call__
    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)
    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()

    """
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)

def gauss_beam(ells,fwhm):
    """Return a Gaussian beam transfer function for the given ells.

    Parameters
    ----------

    ells : ndarray
        Any numpy array containing the multipoles at which the beam transfer function
        is requested.
    fwhm : float
        The beam FWHM in arcminutes.

    Returns
    -------

    output : ndarray
        An array of the same shape as ells containing the Gaussian beam transfer function
        for those multipoles.

    """
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ells**2.) / (16.*np.log(2.)))


class bin2D(object):
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(np.ndarray.flatten(modrmap), bin_edges,right=True)
        self.bin_edges = bin_edges
    def bin(self,data2d,weights=None):
        
        if weights is None:
            res = np.bincount(self.digitized,(data2d).reshape(-1))[1:-1]/np.bincount(self.digitized)[1:-1]
        else:
            res = np.bincount(self.digitized,(data2d*weights).reshape(-1))[1:-1]/np.bincount(self.digitized,weights.reshape(-1))[1:-1]
        return self.centers,res

def rect_geometry(width_arcmin=None,width_deg=None,px_res_arcmin=0.5,proj="car",pol=False,height_deg=None,height_arcmin=None,xoffset_degree=0.,yoffset_degree=0.,extra=False,**kwargs):
    """
    Get shape and wcs for a rectangular patch of specified size and coordinate center
    """

    if width_deg is not None:
        width_arcmin = 60.*width_deg
    if height_deg is not None:
        height_arcmin = 60.*height_deg
    
    hwidth = width_arcmin/2.
    if height_arcmin is None:
        vwidth = hwidth
    else:
        vwidth = height_arcmin/2.
    arcmin =  utils.arcmin
    degree =  utils.degree
    pos = [[-vwidth*arcmin+yoffset_degree*degree,-hwidth*arcmin+xoffset_degree*degree],[vwidth*arcmin+yoffset_degree*degree,hwidth*arcmin+xoffset_degree*degree]]
    shape, wcs = enmap.geometry(pos=pos, res=px_res_arcmin*arcmin, proj=proj,**kwargs)
    if pol: shape = (3,)+shape
    if extra:
        modlmap = enmap.modlmap(shape,wcs)
        lmax = modlmap.max()
        ells = np.arange(0,lmax,1.)
        return shape,wcs,modlmap,ells
    else:
        return shape, wcs
    
