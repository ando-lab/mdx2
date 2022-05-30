
import copy

import numpy as np
from scipy import sparse

from nexusformat.nexus import NXfield, NXdata, NXgroup

class InterpLin1:
    def __init__(self,x,w):
        self.x = x # the points to interpolate on
        self.w = w # the control points, increasing

    def _map(self):
        f = np.interp(self.x,self.w,np.arange(self.w.size),left=np.nan,right=np.nan)
        x_index = np.nonzero(~np.isnan(f))[0].astype(int)
        f = f[x_index]
        bin_index = np.floor(f).astype(int)
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
        N = self.w.size
        row_index = np.concatenate((np.arange(N-2),np.arange(N-2),np.arange(N-2)))
        col_index = np.concatenate((np.arange(N-2),np.arange(1,N-1),np.arange(2,N)))
        vals = np.concatenate((np.full(N-2,-.5),np.full(N-2,1),np.full(N-2,-.5)))
        return sparse.coo_matrix((vals,(row_index,col_index)),shape=(N-2,N))


class ScalingModel:
    """A scale factor the varies only along one coordinate"""
    def __init__(self,x,u):
        self.x = x
        self.u = u

    def copy(self):
        SM = copy.copy(self)
        SM.u = copy.copy(SM.u)
        return SM

    def reset(self,key='u'):
        if key=='u':
            self.u = np.ones_like(self.u)
        else:
            raise ValueError
        return self

    def interp_matrices(self,x):
        Lin1 = InterpLin1(x, self.x)
        return Lin1.A, Lin1.B

    def interp(self,x):
        return InterpLin1(x, self.x).interp(self.u)

    def to_nexus(self):
        x = NXfield(self.x,name='x')
        u = NXfield(self.u,name='u')
        return NXdata(u,x,name='scaling_model')

    @staticmethod
    def from_nexus(nxdata):
        return ScalingModel(nxdata.x.nxvalue,nxdata.u.nxvalue)


class ScaledData:
    def __init__(self,I,sigma,ih,phi,scale=None,mask=None):
        if scale is None:
            scale = np.ones_like(I)
        if mask is None:
            mask = np.isnan(I) | np.isnan(sigma) | (sigma == 0) | np.isinf(sigma)
        self._I = I
        self._sigma = sigma
        self._ih = ih.astype(int)
        self._phi = phi
        self.scale = scale
        self.mask = mask
        self._ihmax = np.max(ih)

    @property
    def I(self):
        Isc = self._I/self.scale
        return np.ma.MaskedArray(data=Isc,mask=self.mask,copy=False)

    @property
    def sigma(self):
        sigmasc = self._sigma/self.scale
        return np.ma.MaskedArray(data=sigmasc,mask=self.mask,copy=False)

    @property
    def ih(self):
        return np.ma.MaskedArray(data=self._ih,mask=self.mask,copy=False)

    @property
    def phi(self):
        return np.ma.MaskedArray(data=self._phi,mask=self.mask,copy=False)

    def copy(self):
        SD = copy.copy(self)
        SD.scale = copy.copy(self.scale)
        SD.mask = copy.copy(self.mask)
        return SD

    def apply(self,Model):
        self.scale = Model.interp(self._phi)
        return self

    #def predict(self,Imerge):
        # Imeas = a*d*(b*I + c)
        #       = scale*(I + offset)
        #scale = Model.interp(self.phi)
        #return self.scale * Imerge[self._ih]

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

    def fit(self,Model,Imerge,alpha_mult):
        AA, BB, Ay, yy = self.calc_problem(Model,Imerge)
        alpha = alpha_mult*np.trace(AA)/np.trace(BB)
        ufit = np.linalg.lstsq(AA + alpha*BB,Ay,rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2*ufit @ Ay + yy
        x2 = x2/self.ih.count()
        NewModel = Model.copy()
        NewModel.u = ufit
        return NewModel, x2

    def calc_problem(self,Model,Imerge):
        A0, B = Model.interp_matrices(self.phi.compressed()) # A0 and B are sparse
        # compute scaled intensities with b = 1
        tmp = self.copy()
        tmp.scale = Model.copy().reset().interp(self._phi)
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()
        v = Imerge[self.ih.compressed()]/sigma
        A = sparse.diags(v) @ A0
        y = I/sigma
        AA = A.T @ A
        BB = B.T @ B
        Ay = A.T @ y
        yy = np.sum(y**2)
        return AA.toarray(), BB.toarray(), Ay, yy
