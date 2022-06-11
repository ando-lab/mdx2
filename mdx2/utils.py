
import importlib

import numpy as np
from scipy.ndimage import map_coordinates
from nexusformat.nexus import nxload

# FUNCTIONS FOR LOADING AND SAVING MDX2 CLASSES TO NEXUS FILES

def loadobj(filename,objectname,verbose=True):
    #simple wrapper to load mdx2 objects from nxs files
    #handles import using the mdx2_module and mdx2_class attributes
    #does a from_nexus() call to instantiate the class
    if verbose: print(f'Reading {objectname} from {filename}')
    nxs = nxload(filename,'r')['/entry/' + objectname]
    mod = nxs.attrs['mdx2_module']
    cls = nxs.attrs['mdx2_class']
    _tmp = importlib.__import__(mod,fromlist=[cls])
    Class = getattr(_tmp,cls)
    if verbose: print(f'  importing as {cls} from {mod}')
    return Class.from_nexus(nxs)

def saveobj(obj,filename,name=None,append=False,verbose=True,mode='w'):
    # simple wrapper to save mdx2 objects as nxs files
    #
    if verbose: print(f'Exporting {type(obj)} to nexus object')
    nxsobj = obj.to_nexus()
    if name is not None:
        nxsobj.rename(name)
    if verbose: print(f'  writing {nxsobj.nxname} to {filename}')
    nxsobj.attrs['mdx2_module'] = type(obj).__module__
    nxsobj.attrs['mdx2_class'] = type(obj).__name__
    if append:
        root = nxload(filename,'r+')
        root['entry/' + nxsobj.nxname] = nxsobj
    else:
        nxsobj.save(filename,mode=mode)
    return nxsobj

# FUNCTIONS FOR EFFICIENT LINEAR INTERPOLATION

def _bilinear_g2g(d0,b,f):

    def clip_data(d0,b):
        bx,by = b
        # just get the grid points I need
        d0 = d0[bx[0]:(bx[-1]+2),by[0]:(by[-1]+2),...]
        bx = bx-bx[0]
        by = by-by[0]
        b = (bx,by)
        return d0, b

    d0, b = clip_data(d0,b)

    bx,by = b
    fx,fy = f
    dx,dy,_ = np.meshgrid(fx,fy,[1],indexing='ij',sparse=True)

    if len(d0.shape) == 2:
        d0 = d0[:,:,np.newaxis]

    c0 = d0[bx,:-1,...]*(1-dx) +d0[bx+1,:-1,...]*dx
    c1 = d0[bx,1:,...]*(1-dx) +d0[bx+1,1:,...]*dx
    c = c0[:,by,...]*(1-dy) + c1[:,by,...]*dy

    if c.shape[2]==1:
        c = np.squeeze(c,axis=2)

    return c

def _quick_trilinear_g2g(d0,b,f):
    # Faster implementation of trilinear interpolation than available in scipy.

    # sort axes so that shortest one is first
    # this has performance gains when one axis is very small
    ord = np.argsort([v.size for v in b])
    unsort = np.empty_like(ord)
    unsort[ord] = np.arange(3)
    sz = d0.shape
    d0 = np.transpose(d0,axes=(ord[0],ord[1],ord[2]) + sz[3:])
    b = (b[ord[0]], b[ord[1]], b[ord[2]])
    f = (f[ord[0]], f[ord[1]], f[ord[2]])

    c = _trilinear_g2g(d0,b,f)

    # put back in order
    c = np.transpose(c,axes=(unsort[0],unsort[1],unsort[2]) + sz[3:])
    return c

def _trilinear_g2g(d0,b,f):
    # perform trilinear interp from one grid to another

    def clip_data(d0,b):
        bx,by,bz = b
        # just get the grid points I need
        d0 = d0[bx[0]:(bx[-1]+2),by[0]:(by[-1]+2),bz[0]:(bz[-1]+2),...]
        bx = bx-bx[0]
        by = by-by[0]
        bz = bz-bz[0]
        b = (bx,by,bz)
        return d0, b

    d0, b = clip_data(d0,b)

    bx,by,bz = b
    fx,fy,fz = f
    dx,dy,dz,_ = np.meshgrid(fx,fy,fz,[1],indexing='ij',sparse=True)

    if len(d0.shape) == 3:
        d0 = d0[:,:,:,np.newaxis]

    c00 = d0[bx,:-1,:-1,...]*(1-dx) +d0[bx+1,:-1,:-1,...]*dx
    c01 = d0[bx,:-1,1:,...]*(1-dx) +d0[bx+1,:-1,1:,...]*dx
    c10 = d0[bx,1:,:-1,...]*(1-dx) +d0[bx+1,1:,:-1,...]*dx
    c11 = d0[bx,1:,1:,...]*(1-dx) +d0[bx+1,1:,1:,...]*dx
    c0 = c00[:,by,:,...]*(1-dy) + c10[:,by,:,...]*dy
    c1 = c01[:,by,:,...]*(1-dy) + c11[:,by,:,...]*dy
    c = c0[:,:,bz,...]*(1-dz) + c1[:,:,bz,...]*dz

    if c.shape[3]==1:
        c = np.squeeze(c,axis=3)

    return c

def _fraction(x,axis=None):
    fx = np.interp(x.astype(float),axis.astype(float),np.arange(axis.size))
    return fx

def _base_fraction(x,axis=None):
    fx = _fraction(x,axis=axis)
    bx = fx.astype(int)
    fx = fx - bx
    return bx, fx

def slice_sections(Ntotal,Nsections):
    """Slices that divide an array of length Ntotal into Nsections. See np.split_array"""
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                 extras * [Neach_section+1] +
                 (Nsections-extras) * [Neach_section])
    div_points = np.array(section_sizes, dtype=int).cumsum()
    slices = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        slices.append(slice(st,end))
    return tuple(slices)

# def bin_2d(x0,y0,v0,Nx,Ny):
#     x_slices = _slice_sections(x0.size,Nx)
#     y_slices = _slice_sections(y0.size,Ny)
#     x = np.array([x0[sl].mean() for sl in x_slices])
#     y = np.array([y0[sl].mean() for sl in y_slices])
#     v = np.empty_like(v0,shape=(y.size,x.size))
#     for ix,slx in enumerate(x_slices):
#         for iy,sly in enumerate(y_slices):
#             v[iy,ix] = v0[sly,slx].mean()
#     return x,y,v
#
# def bin_3d(x0,y0,z0,v0,Nx,Ny,Nz):
#     x_slices = _slice_sections(x0.size,Nx)
#     y_slices = _slice_sections(y0.size,Ny)
#     z_slices = _slice_sections(z0.size,Nz)
#     x = np.array([x0[sl].mean() for sl in x_slices])
#     y = np.array([y0[sl].mean() for sl in y_slices])
#     z = np.array([z0[sl].mean() for sl in z_slices])
#     v = np.empty_like(v0,shape=(z.size,y.size,x.size))
#     for ix,slx in enumerate(x_slices):
#         for iy,sly in enumerate(y_slices):
#             for iz,slz in enumerate(z_slices):
#                 v[iz,iy,ix] = v0[sly,slx].mean()
#     return x,y,z,v


def interp_g2g_bilinear(x0,y0,v0,x,y):
    """bilinear interpolation from one grid to another"""
    bx,fx = _base_fraction(x,axis=x0)
    by,fy = _base_fraction(y,axis=y0)
    return _bilinear_g2g(v0,(bx,by),(fx,fy))

def interp_g2g_trilinear(x0,y0,z0,v0,x,y,z):
    """trilinear interpolation from one grid to another"""
    # bounds are not checked.
    # weird behavior expected if target grid exceed bounds of the source grid...
    bx,fx = _base_fraction(x,axis=x0)
    by,fy = _base_fraction(y,axis=y0)
    bz,fz = _base_fraction(z,axis=z0)
    return _quick_trilinear_g2g(v0,(bx,by,bz),(fx,fy,fz))

def interp3(x0,y0,z0,v0,x,y,z,order=1):
    """interpolate from data on a 3D grid to a set of points"""
    # faster version of scipy.interpolate.interpn using map_coordinates
    fx = _fraction(x,axis=x0)
    fy = _fraction(y,axis=y0)
    fz = _fraction(z,axis=z0)
    coordinates = np.stack((fx,fy,fz))
    return map_coordinates(v0,coordinates,order=order)

def interp2(x0,y0,v0,x,y,order=1):
    """interpolate from data on a 2D grid to a set of points"""
    # faster version of scipy.interpolate.interpn using map_coordinates
    fx = _fraction(x,axis=x0)
    fy = _fraction(y,axis=y0)
    coordinates = np.stack((fx,fy))
    return map_coordinates(v0,coordinates,order=order)
