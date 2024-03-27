
import copy

import numpy as np
from scipy import sparse

from nexusformat.nexus import NXfield, NXdata, NXgroup

try:
    import xarray as xr
except ImportError:
    pass # fail silently... xarray features still in development

class InterpLin3:
    def __init__(self,x,y,z,wx,wy,wz):
        self.x = x # points to interpolate at in the x direction
        self.y = y # points to interpolate at in the y direction
        self.z = z # points to interpolate at in the z direction
        self.wx = wx # the control points in the x direction, increasing
        self.wy = wy # the control points in the y direction, increasing
        self.wz = wz # the control points in the z direction, increasing

    @property
    def shape(self):
        return self.wx.size, self.wy.size, self.wz.size

    def _map(self):
        Nx,Ny,Nz = self.shape
        isinside = (self.x >= self.wx[0]) & (self.x <= self.wx[-1]) & \
                   (self.y >= self.wy[0]) & (self.y <= self.wy[-1]) & \
                   (self.z >= self.wz[0]) & (self.z <= self.wz[-1])
        ind = np.nonzero(isinside)[0]
        x = self.x[ind]
        y = self.y[ind]
        z = self.z[ind]
        fx = np.interp(x,self.wx,np.arange(Nx))
        fy = np.interp(y,self.wy,np.arange(Ny))
        fz = np.interp(z,self.wz,np.arange(Nz))

        xbin = fx.astype(int)
        ybin = fy.astype(int)
        zbin = fz.astype(int)
        xbin[xbin==(Nx-1)] -= 1 # fix edge cases
        ybin[ybin==(Ny-1)] -= 1
        zbin[zbin==(Nz-1)] -= 1
        xfrac = fx - xbin
        yfrac = fy - ybin
        zfrac = fz - zbin
        return ind, xbin, ybin, zbin, xfrac, yfrac, zfrac

    def _sub2ind(self,ix,iy,iz,Nx=None,Ny=None,Nz=None):
        if Nx is None:
            Nx = self.wx.size
        if Ny is None:
            Ny = self.wy.size
        if Nz is None:
            Nz = self.wz.size
        #ind = ix + Nx*(iy + Ny*iz)
        ind = np.ravel_multi_index((ix,iy,iz),(Nx,Ny,Nz),order='F')
        return ind

    def interp(self,p):
        ind, xbin, ybin, zbin, xfrac, yfrac, zfrac = self._map()

        vals = np.zeros_like(self.x)
        p = p.flatten(order='F')
        vals[ind] = \
            p[self._sub2ind(xbin,  ybin,  zbin  )]*(1-zfrac)*(1-xfrac)*(1-yfrac) + \
            p[self._sub2ind(xbin+1,ybin,  zbin  )]*(1-zfrac)*xfrac*(1-yfrac) + \
            p[self._sub2ind(xbin,  ybin+1,zbin  )]*(1-zfrac)*(1-xfrac)*yfrac + \
            p[self._sub2ind(xbin+1,ybin+1,zbin  )]*(1-zfrac)*xfrac*yfrac + \
            p[self._sub2ind(xbin,  ybin,  zbin+1)]*zfrac*(1-xfrac)*(1-yfrac) + \
            p[self._sub2ind(xbin+1,ybin,  zbin+1)]*zfrac*xfrac*(1-yfrac) + \
            p[self._sub2ind(xbin,  ybin+1,zbin+1)]*zfrac*(1-xfrac)*yfrac + \
            p[self._sub2ind(xbin+1,ybin+1,zbin+1)]*zfrac*xfrac*yfrac

        return vals

    @property
    def A(self):
        ind, xbin, ybin, zbin, xfrac, yfrac, zfrac = self._map()
        vals = np.concatenate((
            (1-zfrac)*(1-xfrac)*(1-yfrac),
            (1-zfrac)*xfrac*(1-yfrac),
            (1-zfrac)*(1-xfrac)*yfrac,
            (1-zfrac)*xfrac*yfrac,
            zfrac*(1-xfrac)*(1-yfrac),
            zfrac*xfrac*(1-yfrac),
            zfrac*(1-xfrac)*yfrac,
            zfrac*xfrac*yfrac,
            ))
        row_index = np.concatenate((ind,ind,ind,ind,ind,ind,ind,ind))
        x_index = np.concatenate((xbin,xbin+1,xbin,xbin+1,xbin,xbin+1,xbin,xbin+1))
        y_index = np.concatenate((ybin,ybin,ybin+1,ybin+1,ybin,ybin,ybin+1,ybin+1))
        z_index = np.concatenate((zbin,zbin,zbin,zbin,zbin+1,zbin+1,zbin+1,zbin+1))
        col_index = self._sub2ind(x_index,y_index,z_index)
        shape = (self.x.size,self.wx.size*self.wy.size*self.wz.size)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def Bx(self):
        raise NotImplementedError

    @property
    def By(self):
        raise NotImplementedError

    @property
    def Bz(self):
        Nx, Ny, Nz = self.shape
        k,l,m = np.meshgrid(np.arange(Nx),np.arange(Ny),np.arange(Nz),indexing='ij')
        ind_left  = self._sub2ind(k[:,:,:-2],l[:,:,:-2],m[:,:,:-2]).ravel(order='F');
        ind_self  = self._sub2ind(k[:,:,1:-1],l[:,:,1:-1],m[:,:,1:-1]).ravel(order='F');
        ind_right = self._sub2ind(k[:,:,2:],l[:,:,2:],m[:,:,2:]).ravel(order='F');
        ind_rows  = self._sub2ind(k[:,:,:-2],l[:,:,:-2],m[:,:,:-2]).ravel(order='F');
        nrows = Nx*Ny*(Nz-2)
        row_index = np.concatenate((ind_rows,ind_rows,ind_rows))
        col_index = np.concatenate((ind_left,ind_self,ind_right))
        vals = np.concatenate((np.full(nrows,-0.5),np.full(nrows,1.0),np.full(nrows,-0.5)))
        shape = (nrows,Nx*Ny*Nz)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def B(self):
        raise NotImplementedError

    @property
    def Bxy(self):
        Nx, Ny, Nz = self.shape
        k,l,m = np.meshgrid(np.arange(Nx),np.arange(Ny),np.arange(Nz),indexing='ij')
        ind_left  = self._sub2ind(k[:-1,:,:],l[:-1,:,:],m[:-1,:,:]).ravel(order='F');
        ind_right  = self._sub2ind(k[1:,:,:],l[1:,:,:],m[1:,:,:]).ravel(order='F');
        ind_up  = self._sub2ind(k[:,:-1,:],l[:,:-1,:],m[:,:-1,:]).ravel(order='F');
        ind_down  = self._sub2ind(k[:,1:,:],l[:,1:,:],m[:,1:,:]).ravel(order='F');
        row_index = np.concatenate((ind_left,ind_right,ind_up,ind_down))
        col_index = np.concatenate((ind_right,ind_left,ind_down,ind_up))
        vals = np.ones_like(row_index,dtype=np.double)
        shape = (Nx*Ny*Nz,Nx*Ny*Nz)
        L = sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)
        # weight accordint to the number of neighbors
        nn = np.asarray(L.sum(axis=1)).flatten()
        return sparse.eye(shape[0]) - sparse.diags(1/nn)@L


class InterpLin2:
    def __init__(self,x,y,wx,wy):
        self.x = x # points to interpolate at in the x direction
        self.y = y # points to interpolate at in the y direction
        self.wx = wx # the control points in the x direction, increasing
        self.wy = wy # the control points in the y direction, increasing

    @property
    def shape(self):
        return self.wx.size, self.wy.size

    def _map(self):
        Nx,Ny = self.shape
        fx = np.interp(self.x,self.wx,np.arange(Nx),left=np.nan,right=np.nan)
        fy = np.interp(self.y,self.wy,np.arange(Ny),left=np.nan,right=np.nan)
        ind = np.nonzero(~np.isnan(fx) & ~np.isnan(fy))[0].astype(int)
        fx = fx[ind]
        fy = fy[ind]
        xbin = np.floor(fx).astype(int)
        ybin = np.floor(fy).astype(int)
        xbin[xbin==(Nx-1)] -= 1 # fix edge cases
        ybin[ybin==(Ny-1)] -= 1
        xfrac = fx - xbin
        yfrac = fy - ybin
        return ind, xbin, ybin, xfrac, yfrac

    def _sub2ind(self,ix,iy,Nx=None,Ny=None):
        """convert 2D array index to linear index"""
        if Nx is None:
            Nx = self.wx.size
        if Ny is None:
            Ny = self.wy.size
        #ind = ix + Nx*iy;
        ind = np.ravel_multi_index((ix,iy),(Nx,Ny),order='F')
        return ind

    def interp(self,p):
        ind, xbin, ybin, xfrac, yfrac = self._map()
        vals = np.zeros_like(self.x)
        p = p.flatten(order='F')
        vals[ind] = \
            p[self._sub2ind(xbin,ybin)]*(1-xfrac)*(1-yfrac) + \
            p[self._sub2ind(xbin+1,ybin)]*xfrac*(1-yfrac) + \
            p[self._sub2ind(xbin,ybin+1)]*(1-xfrac)*yfrac + \
            p[self._sub2ind(xbin+1,ybin+1)]*xfrac*yfrac
        return vals

    @property
    def A(self):
        ind, xbin, ybin, xfrac, yfrac = self._map()
        vals = np.concatenate(((1-xfrac)*(1-yfrac), xfrac*(1-yfrac), (1-xfrac)*yfrac, xfrac*yfrac))
        row_index = np.concatenate((ind,ind,ind,ind))
        x_index = np.concatenate((xbin,xbin+1,xbin,xbin+1))
        y_index = np.concatenate((ybin,ybin,ybin+1,ybin+1))
        col_index = self._sub2ind(x_index,y_index)
        shape = (self.x.size,self.wx.size*self.wy.size)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def Bx(self):
        Nx, Ny = self.shape
        k,l = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='ij')
        ind_left  = self._sub2ind(k[:-2,:],l[:-2,:]).ravel(order='F');
        ind_self  = self._sub2ind(k[1:-1,:],l[1:-1,:]).ravel(order='F');
        ind_right = self._sub2ind(k[2:,:],l[2:,:]).ravel(order='F');
        ind_rows  = self._sub2ind(k[:-2,:],l[:-2,:],Nx=Nx-2).ravel(order='F');
        nrows = (Nx-2)*Ny
        row_index = np.concatenate((ind_rows,ind_rows,ind_rows))
        col_index = np.concatenate((ind_left,ind_self,ind_right))
        vals = np.concatenate((np.full(nrows,-0.5),np.full(nrows,1.0),np.full(nrows,-0.5)))
        shape = (nrows,Nx*Ny)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def By(self):
        Nx, Ny = self.shape
        k,l = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='ij')
        ind_left  = self._sub2ind(k[:,:-2],l[:,:-2]).ravel(order='F');
        ind_self  = self._sub2ind(k[:,1:-1],l[:,1:-1]).ravel(order='F');
        ind_right = self._sub2ind(k[:,2:],l[:,2:]).ravel(order='F');
        ind_rows  = self._sub2ind(k[:,:-2],l[:,:-2]).ravel(order='F');
        nrows = Nx*(Ny-2)
        row_index = np.concatenate((ind_rows,ind_rows,ind_rows))
        col_index = np.concatenate((ind_left,ind_self,ind_right))
        vals = np.concatenate((np.full(nrows,-0.5),np.full(nrows,1.0),np.full(nrows,-0.5)))
        shape = (nrows,Nx*Ny)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def B(self):
        Nx, Ny = self.shape
        k,l = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='ij')
        ind_left  = self._sub2ind(k[:-1,:],l[:-1,:]).ravel(order='F');
        ind_right  = self._sub2ind(k[1:,:],l[1:,:]).ravel(order='F');
        ind_up  = self._sub2ind(k[:,:-1],l[:,:-1]).ravel(order='F');
        ind_down  = self._sub2ind(k[:,1:],l[:,1:]).ravel(order='F');
        row_index = np.concatenate((ind_left,ind_right,ind_up,ind_down))
        col_index = np.concatenate((ind_right,ind_left,ind_down,ind_up))
        vals = np.ones_like(row_index,dtype=np.double)
        shape = (Nx*Ny,Nx*Ny)
        L = sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)
        # weight accordint to the number of neighbors
        nn = np.asarray(L.sum(axis=1)).flatten()
        return sparse.eye(shape[0]) - sparse.diags(1/nn)@L


class InterpLin1:
    def __init__(self,x,w):
        self.x = x # the points to interpolate on
        self.w = w # the control points, increasing

    @property
    def shape(self):
        return self.w.size

    def _map(self):
        N = self.shape
        f = np.interp(self.x,self.w,np.arange(N),left=np.nan,right=np.nan)
        x_index = np.nonzero(~np.isnan(f))[0].astype(int)
        f = f[x_index]
        bin_index = np.floor(f).astype(int)
        bin_index[bin_index==(N-1)] -= 1 # fix edge cases
        bin_index[bin_index==self.w.size-1] -= 1 # deal with edge case
        bin_fraction = f - bin_index
        return x_index, bin_index, bin_fraction

    def interp(self,p):
        ix, bin, frac = self._map()
        vals = np.zeros_like(self.x)
        vals[ix] = p[bin]*(1-frac) + p[bin+1]*frac
        return vals

    @property
    def A(self):
        ix, bin, frac = self._map()
        vals = np.concatenate((1-frac,frac))
        row_index = np.concatenate((ix,ix))
        col_index = np.concatenate((bin,bin+1))
        shape = (self.x.size,self.w.size)
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=shape)

    @property
    def B(self):
        N = self.shape
        row_index = np.concatenate((np.arange(N-2),np.arange(N-2),np.arange(N-2)))
        col_index = np.concatenate((np.arange(N-2),np.arange(1,N-1),np.arange(2,N)))
        vals = np.concatenate((np.full(N-2,-.5),np.full(N-2,1),np.full(N-2,-.5)))
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=(N-2,N))

class _BaseModel:
    """Common methods for scaling models"""
    # properties:
    #    x,y,... = coordinates to interpolate at
    #    wx,wy,... = axes of control point grid
    #    u = value at control points
    #
    # for compatibility with MATLAB code, use Fortran array order
    # u.shape == (Nx,Ny,...)
    @property
    def shape(self):
        pass # implemented in subclass

    def interp_matrices(self,*args):
        pass # implemented in subclass

    def interp(self,x):
        pass # implemented in subclass

    def to_nexus(self):
        pass # implemented in subclass

    @staticmethod
    def from_nexus(n):
        pass # implemented in subclass

    def to_xarray(self):
        pass # implemented in subclass

    @staticmethod
    def from_xarray(da):
        pass # implemented in subclass

    def set_u(self,val):
        self.u = val.reshape(self.shape,order='F')

    def plot(self,*args,**kwargs):
        self.to_xarray.plot(*args,**kwargs)

    def copy(self):
        SM = copy.copy(self)
        SM.u = copy.copy(SM.u)
        return SM


class ScalingModel(_BaseModel):
    """A scale factor the varies only along one coordinate"""
    def __init__(self,phi,u=None):
        self.x = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size

    def interp_matrices(self,x):
        Lin1 = InterpLin1(x, self.x)
        return Lin1.A, Lin1.B

    def interp(self,x):
        if self.u is None:
            val = np.ones_like(x,dtype=np.double)
        else:
            val =InterpLin1(x, self.x).interp(self.u)
        return val

    def to_nexus(self):
        phi = NXfield(self.x,name='phi')
        u = NXfield(self.u,name='u')
        return NXdata(signal=u,axes=[phi],name='scaling_model')

    @staticmethod
    def from_nexus(nxdata):
        return ScalingModel(nxdata.phi.nxvalue,nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(
            self.u,
            dims=("phi"),
            coords={"phi":self.x})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords['phi'].data
        return ScalingModel(x,da.data)


class AbsorptionModel(_BaseModel):
    """A smooth scale factor the varies along 3 coordinates"""
    def __init__(self,ix,iy,phi,u=None):
        self.x = ix
        self.y = iy
        self.z = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size, self.z.size

    def interp_matrices(self,x,y,z):
        Lin3 = InterpLin3(x,y,z,self.x,self.y,self.z)
        return Lin3.A, Lin3.Bxy, Lin3.Bz

    def interp(self,x,y,z):
        if self.u is None:
            val = np.ones_like(x,dtype=np.double)
        else:
            val = InterpLin3(x,y,z,self.x,self.y,self.z).interp(self.u)
        return val

    def to_nexus(self):
        ix = NXfield(self.x,name='ix')
        iy = NXfield(self.y,name='iy')
        phi = NXfield(self.z,name='phi')
        u = NXfield(self.u,name='u')
        return NXdata(signal=u,axes=[ix,iy,phi],name='absorption_model')

    @staticmethod
    def from_nexus(nxdata):
        return AbsorptionModel(nxdata.ix.nxvalue,nxdata.iy.nxvalue,nxdata.phi.nxvalue,nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(
            self.u,
            dims=("ix", "iy", "phi"),
            coords={"ix":self.x, "iy":self.x, "phi":self.z})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords['ix'].data
        y = da.coords['iy'].data
        z = da.coords['phi'].data
        return AbsorptionModel(x,y,z,da.data)


class OffsetModel(_BaseModel):
    """A smooth offset the varies along 2 coordinates"""
    def __init__(self,s,phi,u=None):
        self.x = s
        self.y = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size

    def interp_matrices(self,x,y):
        Lin2 = InterpLin2(x,y,self.x,self.y)
        return Lin2.A, Lin2.Bx, Lin2.By

    def interp(self,x,y):
        if self.u is None:
            val = np.zeros_like(x,dtype=np.double)
        else:
            val = InterpLin2(x,y,self.x,self.y).interp(self.u)
        return val

    def to_nexus(self):
        s = NXfield(self.x,name='s')
        phi = NXfield(self.y,name='phi')
        u = NXfield(self.u,name='u')
        return NXdata(signal=u,axes=[s,phi],name='offset_model')

    @staticmethod
    def from_nexus(nxdata):
        return OffsetModel(nxdata.s.nxvalue,nxdata.phi.nxvalue,nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(
            self.u,
            dims=("s", "phi"),
            coords={"s":self.x, "phi":self.y})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords['s'].data
        y = da.coords['phi'].data
        return OffsetModel(x,y,da.data)

class DetectorModel(_BaseModel):
    """A smooth scale factor the varies along the detector face"""
    def __init__(self,ix,iy,u=None):
        self.x = ix
        self.y = iy
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size

    def interp_matrices(self,x,y):
        Lin3 = InterpLin2(x,y,self.x,self.y)
        return Lin3.A, Lin3.B

    def interp(self,x,y):
        if self.u is None:
            val = np.ones_like(x,dtype=np.double)
        else:
            val = InterpLin2(x,y,self.x,self.y).interp(self.u)
        return val

    def to_nexus(self):
        ix = NXfield(self.x,name='ix')
        iy = NXfield(self.y,name='iy')
        u = NXfield(self.u,name='u')
        return NXdata(signal=u,axes=[ix,iy],name='detector_model')

    @staticmethod
    def from_nexus(nxdata):
        return DetectorModel(nxdata.ix.nxvalue,nxdata.iy.nxvalue,nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(
            self.u,
            dims=("ix", "iy"),
            coords={"ix":self.x, "iy":self.x})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords['ix'].data
        y = da.coords['iy'].data
        return AbsorptionModel(x,y,da.data)


class ScaledData:
    def __init__(self,I,sigma,ih,phi=None,ix=None,iy=None,s=None,batch=None,scale=None,offset=None,mask=None):
        if scale is None:
            scale = np.ones_like(I)
        if offset is None:
            offset = np.zeros_like(I)
        if mask is None:
            mask = np.isnan(I) | np.isnan(sigma) | (sigma == 0) | np.isinf(sigma)
        if batch is None:
            self._batches = [slice(0,len(I)+1)]
        else: # batch is a sorted list of integers
            if not np.all(batch[:-1] <= batch[1:]): # must be sorted
                raise ValueError
            breakpoints = [0] + list(np.nonzero(np.diff(batch))[0] + 1) + [len(I)+1]
            starts = breakpoints[:-1]
            stops = breakpoints[1:]
            self._batches = [slice(start,stop) for start, stop in zip(starts,stops)]
            # note: batch number is not saved! they are just stored sequentially in the list

        if phi is None:
            phi = np.full_like(I,np.nan)
        if ix is None:
            ix = np.full_like(I,np.nan)
        if iy is None:
            iy = np.full_like(I,np.nan)
        if s is None:
            s = np.full_like(I,np.nan)

        self._I = I
        self._sigma = sigma
        self._ih = ih.astype(int)
        self._phi = phi
        self._ix = ix
        self._iy = iy
        self._s = s
        self.scale = scale
        self.offset = offset
        self.mask = mask
        self._ihmax = np.max(ih)

    def __getitem__(self,ind):
        dset = ScaledData(
                self._I[ind],
                self._sigma[ind],
                self._ih[ind],
                phi=self._phi[ind],
                ix=self.ix[ind],
                iy=self.iy[ind],
                s=self.s[ind],
                scale=self.scale[ind],
                offset=self.offset[ind],
                mask=self.mask[ind],
        )
        dset._ihmax = self._ihmax
        #print('DEBUG: sliced dataset. Is scale a copy?',dset.scale.base is None)
        return dset

    def batches(self):
        """generator that loops over batches, returning a ScaledData object for each"""
        for sl in self._batches:
            yield self[sl]

    @property
    def nbatches(self):
        return len(self._batches)

    def _masked(self,arr):
        return np.ma.MaskedArray(data=np.ma.getdata(arr),mask=self.mask,copy=False)

    @property
    def I(self):
        # suppress divide-by-zero warning
        old_settings = np.seterr(all='ignore')
        val = self._masked(self._I/self.scale - self.offset)
        np.seterr(**old_settings)
        return val

    @property
    def sigma(self):
        old_settings = np.seterr(all='ignore')
        val = self._masked(self._sigma/self.scale)
        np.seterr(**old_settings)
        return val

    @property
    def ih(self):
        return self._masked(self._ih)

    @property
    def phi(self):
        return self._masked(self._phi)

    @property
    def ix(self):
        return self._masked(self._ix)

    @property
    def iy(self):
        return self._masked(self._iy)

    @property
    def s(self):
        return self._masked(self._s)

    def predict(self,Imerge):
        return self._masked(self.scale*(Imerge[self._ih] + self.offset))

    def copy(self):
        SD = copy.copy(self)
        SD.scale = copy.copy(self.scale)
        SD.offset = copy.copy(self.offset)
        SD.mask = copy.copy(self.mask)
        return SD

    def merge(self):
        n = self._ihmax+1
        w = 1/self.sigma.compressed()**2
        Iw = self.I.compressed()*w
        ind = self.ih.compressed()
        counts = np.bincount(ind,minlength=n)
        wm = np.bincount(ind,weights=w,minlength=n)
        Im = np.bincount(ind,weights=Iw,minlength=n)
        wm = np.ma.masked_where(counts==0,wm,copy=False)
        Im = np.ma.masked_where(counts==0,Im,copy=False)
        Im = Im/wm
        sigmam = 1/np.sqrt(wm)
        return Im, sigmam, counts

    def mask_outliers(self,Im,nsigma):
        resid = (self.scale*(Im[self._ih] + self.offset) - self._I)/self._sigma
        isoutlier = (resid > nsigma) & ~(self.mask)
        self.mask[isoutlier] = True # apply outlier masks
        return isoutlier.sum()


class _ModelRefiner:
    def __init__(self,data,model=None):
        self.data = data
        self.model = model
        self._value = None # cached value of the interpolated model at datapoints

    def calc_scale_offset(self,a=None,b=None,c=None,d=None):
        scale = np.ones_like(self.data._I,dtype=np.double)
        offset = np.zeros_like(scale)
        if a is None:
            a = 1.0
        if b is None:
            b = 1.0
        if c is None:
            c = 0.0
        if d is None:
            d = 1.0
        scale *= a*b*d
        offset += c/b
        return scale, offset

    def _calc_points(self,x,dx=1,nx=None,integer_limits=True,x_min=None,x_max=None):
        if x_min is None:
            x_min = np.min(x)
        if x_max is None:
            x_max = np.max(x)
        if integer_limits:
            x_min = np.floor(x_min)
            x_max = np.ceil(x_max)
        if nx is None:
            nx = np.round((x_max-x_min)/dx).astype(int) + 1
        return np.linspace(x_min,x_max,nx)

    def add_model(self,*args,**kwargs):
        raise NotImplementedError

    def calc_problem(self,Imerge,**kwargs):
        raise NotImplementedError

    def fit(self,Imerge,**kwargs):
        raise NotImplementedError

    @property
    def value(self):
        if self._value is None:
            self._value = self.calc_value()
        return self._value

    def refine(self,Imerge,*args,**kwargs):
        ufit, x2 = self.fit(Imerge,*args,**kwargs)
        self.model.set_u(ufit)
        self._value = None # trigger re-calculation when requested
        return x2

class ScalingModelRefiner(_ModelRefiner):

    def add_model(self,dphi=1,nearest_degree=True,nphi=None):
        phi_points = self._calc_points(self.data.phi,dx=dphi,integer_limits=nearest_degree,nx=nphi)
        self.model = ScalingModel(phi_points)
        return self

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._phi)

    def calc_problem(self,Imerge,a=None,c=None,d=None):
        A0, B = self.model.interp_matrices(
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = b = d = 1, c = 0
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset()
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with b = 1
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a,c=c,d=d)
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred/sigma) @ A0
        y = I/sigma
        AA = A.T @ A
        BB = B.T @ B
        Ay = A.T @ y
        yy = np.sum(y**2)
        return AA, BB, Ay, yy

    def fit(self,Imerge,alpha_mult,a=None,c=None,d=None):
        AA, BB, Ay, yy = self.calc_problem(Imerge,a=a,c=c,d=d)
        alpha = alpha_mult*AA.trace()/BB.trace()
        G = AA + alpha*BB
        if sparse.issparse(G):
            ufit = sparse.linalg.lsqr(G,Ay)[0]
        else:
            ufit = np.linalg.lstsq(G,Ay,rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2*ufit @ Ay + yy
        x2 = x2/self.data.ih.count()
        return ufit, x2

class AbsorptionModelRefiner(_ModelRefiner):

    def add_model(self,dphi=10,nearest_degree=True,nphi=None,nix=20,dix=None,niy=20,diy=None):
        phi_points = self._calc_points(self.data.phi,dx=dphi,integer_limits=nearest_degree,nx=nphi)
        ix_points = self._calc_points(self.data.ix,dx=dix,integer_limits=True,nx=nix)
        iy_points = self._calc_points(self.data.iy,dx=diy,integer_limits=True,nx=niy)
        self.model = AbsorptionModel(ix_points,iy_points,phi_points)

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._ix,self.data._iy,self.data._phi)

    def calc_problem(self,Imerge,b=None,c=None,d=None):
        A0, Bxy, Bz = self.model.interp_matrices(
            self.data.ix.compressed(),
            self.data.iy.compressed(),
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = 1
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(b=b,c=c,d=d)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with a = b = d = 1, c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset()
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred/sigma) @ A0
        y = I/sigma
        AA = A.T @ A
        BBxy = Bxy.T @ Bxy
        BBz = Bz.T @ Bz
        Ay = A.T @ y
        yy = np.sum(y**2)

        return AA, BBxy, BBz, Ay, yy

    def fit(self,Imerge,alpha_xy_mult,alpha_z_mult,b=None,c=None,d=None):
        AA, BBxy, BBz, Ay, yy = self.calc_problem(Imerge,b=b,c=c,d=d)
        alpha_xy = alpha_xy_mult*AA.trace()/BBxy.trace()
        alpha_z = alpha_z_mult*AA.trace()/BBz.trace()
        G = AA + alpha_xy*BBxy + alpha_z*BBz
        if sparse.issparse(G):
            ufit = sparse.linalg.lsqr(G,Ay)[0]
        else:
            ufit = np.linalg.lstsq(G,Ay,rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2*ufit @ Ay + yy
        x2 = x2/self.data.ih.count()
        self.model.set_u(ufit)
        return ufit, x2

class OffsetModelRefiner(_ModelRefiner):

    def add_model(self,dphi=2.5,nearest_degree=True,nphi=None,ns=31,ds=None):
        phi_points = self._calc_points(self.data.phi,dx=dphi,integer_limits=nearest_degree,nx=nphi)
        s_points = self._calc_points(self.data.s,dx=ds,integer_limits=False,nx=ns)
        self.model = OffsetModel(s_points,phi_points)
        return self

    def calc_value(self):
        if self.model is None:
            return 0.0
        return self.model.interp(self.data._s,self.data._phi)

    def calc_problem(self,Imerge,a=None,b=None,d=None):
        A0, Bx, By = self.model.interp_matrices(
            self.data.s.compressed(),
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = 1, c = 0
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(b=b)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with b = 1 and c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a,d=d)
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(1/sigma) @ A0
        y = (I-Ipred)/sigma
        AA = A.T @ A
        BBx = Bx.T @ Bx
        BBy = By.T @ By
        Ay = A.T @ y
        yy = np.sum(y**2)
        H = sparse.eye(A.shape[1])
        return AA, BBx, BBy, H, Ay, yy

    def fit(self,Imerge,alpha_x_mult,alpha_y_mult,alpha_c_mult,min_c,a=None,b=None,d=None):
        AA, BBx, BBy, H, Ay, yy = self.calc_problem(Imerge,a=a,b=b,d=d)
        alpha_x = alpha_x_mult*AA.trace()/BBx.trace()
        alpha_y = alpha_y_mult*AA.trace()/BBy.trace()
        #alpha_c = alpha_c_mult*AA.trace()/H.trace()
        alpha_c = alpha_c_mult*np.max(AA.diagonal())/np.max(H.diagonal())
        G = AA + alpha_x*BBx + alpha_y*BBy + alpha_c*H
        if sparse.issparse(G):
            ufit = sparse.linalg.lsqr(G,Ay)[0]
        else:
            ufit = np.linalg.lstsq(G,Ay,rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2*ufit @ Ay + yy
        x2 = x2/self.data.ih.count()
        if min_c is not None:
            ufit[ufit < min_c] = min_c   # apply threshold
            ufit += min_c - np.min(ufit) # level shift

        return ufit, x2

class DetectorModelRefiner(_ModelRefiner):

    def add_model(self,nix=20,dix=None,niy=20,diy=None):
        ix_points = self._calc_points(self.data.ix,dx=dix,integer_limits=True,nx=nix)
        iy_points = self._calc_points(self.data.iy,dx=diy,integer_limits=True,nx=niy)
        self.model = DetectorModel(ix_points,iy_points)

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._ix,self.data._iy)

    def calc_problem(self,Imerge,a=None,b=None,c=None):
        A0, B = self.model.interp_matrices(
            self.data.ix.compressed(),
            self.data.iy.compressed(),
        )

        # compute predicted intensities d = 1
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a,b=b,c=c)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with a = b = d = 1, c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset()
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred/sigma) @ A0
        y = I/sigma
        AA = A.T @ A
        BB = B.T @ B
        Ay = A.T @ y
        yy = np.sum(y**2)

        return AA, BB, Ay, yy

    def fit(self,Imerge,alpha_mult,a=None,b=None,c=None):
        AA, BB, Ay, yy = self.calc_problem(Imerge,a=a,b=b,c=c)
        alpha = alpha_mult*AA.trace()/BB.trace()
        G = AA + alpha*BB
        if sparse.issparse(G):
            ufit = sparse.linalg.lsqr(G,Ay)[0]
        else:
            ufit = np.linalg.lstsq(G,rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2*ufit @ Ay + yy
        x2 = x2/self.data.ih.count()
        return ufit, x2


class CombinedModelRefiner:
    def __init__(self,data,scaling_model=None,absorption_model=None,offset_model=None,detector_model=None):
        self.data = data
        self.scaling = ScalingModelRefiner(data,scaling_model)
        self.absorption = AbsorptionModelRefiner(data,absorption_model)
        self.offset = OffsetModelRefiner(data,offset_model)
        self.detector = DetectorModelRefiner(data,detector_model)

    @property
    def a(self):
        return self.absorption.value

    @property
    def b(self):
        return self.scaling.value

    @property
    def c(self):
        return self.offset.value

    @property
    def d(self):
        return self.detector.value

    def calc_scale_offset(self):
        scale = np.ones_like(self.data._I,dtype=np.double)
        offset = np.zeros_like(scale)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        scale *= a*b*d
        offset += c/b
        return scale, offset

    def apply(self):
        """ apply the scaling model to the data """
        self.data.scale[:], self.data.offset[:] = self.calc_scale_offset()

    def bfit(self,*args,**kwargs):
        return self.scaling.refine(*args,a=self.a,c=self.c,d=self.d,**kwargs)

    def afit(self,*args,**kwargs):
        return self.absorption.refine(*args,b=self.b,c=self.c,d=self.d,**kwargs)

    def dfit(self,*args,**kwargs):
        return self.detector.refine(*args,a=self.a,b=self.b,c=self.c,**kwargs)

    def cfit(self,*args,**kwargs):
        return self.offset.refine(*args,a=self.a,b=self.b,d=self.d,**kwargs)

class BatchModelRefiner:
    def __init__(self,data,scaling_models=None,absorption_models=None,offset_models=None,detector_model=None):
        self.data = data
        self._batch_refiners = []

        if scaling_models is None:
            scaling_models = [None]*self.data.nbatches
        if absorption_models is None:
            absorption_models = [None]*self.data.nbatches
        if offset_models is None:
            offset_models = [None]*self.data.nbatches

        for ds, sm, am, om in zip(self.data.batches(),scaling_models,absorption_models,offset_models):
            self._batch_refiners.append(CombinedModelRefiner(ds,sm,am,om,detector_model))

        self.detector = DetectorModelRefiner(data,detector_model)

    def add_scaling_models(self,*args,**kwargs):
        [br.scaling.add_model(*args,**kwargs) for br in self._batch_refiners]

    def add_offset_models(self,*args,**kwargs):
        [br.offset.add_model(*args,**kwargs) for br in self._batch_refiners]

    def add_absorption_models(self,*args,**kwargs):
        [br.absorption.add_model(*args,**kwargs) for br in self._batch_refiners]

    def add_detector_model(self,*args,**kwargs):
        tmp = DetectorModelRefiner(self.data)
        tmp.add_model(*args,**kwargs)
        self.detector = tmp
        for br in self._batch_refiners:
            br.detector.model = self.detector.model

    def bfit(self,*args,**kwargs):
        x2 = [br.bfit(*args,**kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    def afit(self,*args,**kwargs):
        x2 = [br.afit(*args,**kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    def cfit(self,*args,**kwargs):
        x2 = [br.cfit(*args,**kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    @property
    def a(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.a*np.ones_like(br.data._I,dtype=np.double))
        return np.concatenate(vals)

    @property
    def b(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.b*np.ones_like(br.data._I,dtype=np.double))
        return np.concatenate(vals)

    @property
    def c(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.c*np.ones_like(br.data._I,dtype=np.double))
        return np.concatenate(vals)

    def dfit(self,*args,**kwargs):
        for br in self._batch_refiners:
            br.detector._value = None # trigger re-compute on next apply
        return self.detector.refine(*args,a=self.a,b=self.b,c=self.c,**kwargs)

    def apply(self):
        [br.apply() for br in self._batch_refiners]
