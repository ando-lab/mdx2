from copy import deepcopy

import numpy as np
import pandas as pd

from nexusformat.nexus import NXfield, NXdata, NXgroup, NXreflections

from mdx2.dxtbx_machinery import Experiment
from mdx2.geometry import GridData

class Peaks:
    """search peaks in image stack"""
    def __init__(self, phi, iy, ix):
        self.phi = phi
        self.iy = iy
        self.ix = ix

    def to_mask(self,phi_axis,iy_axis,ix_axis,mask_in=None):
        shape = (phi_axis.size,iy_axis.size,ix_axis.size)

        ind0 = np.round(np.interp(self.phi,phi_axis,np.arange(shape[0]))).astype(int)
        ind1 = np.round(np.interp(self.iy,iy_axis,np.arange(shape[1]))).astype(int)
        ind2 = np.round(np.interp(self.ix,ix_axis,np.arange(shape[2]))).astype(int)
        if mask_in is not None:
            mask = mask_in
        else:
            mask = np.zeros(shape,dtype='bool')
        mask[ind0,ind1,ind2] = True
        return mask

    @staticmethod
    def where(mask,phi_axis,iy_axis,ix_axis):
        hotpixels = np.argwhere(mask)
        return Peaks(
            phi_axis[hotpixels[:,0]],
            iy_axis[hotpixels[:,1]],
            ix_axis[hotpixels[:,2]],
            )

    @staticmethod
    def stack(peaks):
        phi = np.concatenate([p.phi for p in peaks])
        ix = np.concatenate([p.ix for p in peaks])
        iy = np.concatenate([p.iy for p in peaks])
        return Peaks(phi,iy,ix)

    @property
    def size(self):
        return self.phi.size

    def to_nexus(self):
        return NXreflections(
            name='peaks',
            observed_phi=self.phi,
            observed_px_x=self.ix,
            observed_px_y=self.iy,
            )

    @staticmethod
    def from_nexus(peaks):
        return Peaks(
            peaks.observed_phi.nxvalue,
            peaks.observed_px_y.nxvalue,
            peaks.observed_px_x.nxvalue,
        )



class HKLTable:
    """Container for data in a table with indices h,k,l"""
    def __init__(self,h,k,l,_ndiv=None,**kwargs):
        self.h = h
        self.k = k
        self.l = l
        self._ndiv = _ndiv
        self.__dict__.update(kwargs)

    @property
    def _data_keys(self):
        return [key for key in self.__dict__ if key not in ['h','k','l','_ndiv']]

    def __len__(self):
        return len(self.h)

    def to_frame(self):
        """ convert to pandas dataframe """
        cols = {k:self.__dict__[k] for k in self.__dict__ if k not in ['_ndiv']}
        return pd.DataFrame(cols)

    def to_asu(self,Symmetry):
        if self._ndiv is None:
            print('warning: hkl table appears to be unbinned. Miller indices will be rounded to nearest integer')
            ndiv = [1,1,1]
        else:
            ndiv = self._ndiv

        H = np.round(self.h*ndiv[0]).astype(int)
        K = np.round(self.k*ndiv[1]).astype(int)
        L = np.round(self.l*ndiv[2]).astype(int)

        H,K,L,op = Symmetry.to_asu(H,K,L)

        newtable = deepcopy(self)
        newtable.h = H/ndiv[0]
        newtable.k = K/ndiv[1]
        newtable.l = L/ndiv[2]
        newtable.op = op

        return newtable


    def bin(self,ndiv=[1,1,1],column_names=None,count_name='count'):
        if column_names is None:
            column_names = self._data_keys
        H = np.round(self.h*ndiv[0]).astype(int)
        K = np.round(self.k*ndiv[1]).astype(int)
        L = np.round(self.l*ndiv[2]).astype(int)
        Hmin, Hmax = np.min(H), np.max(H)
        Kmin, Kmax = np.min(K), np.max(K)
        Lmin, Lmax = np.min(L), np.max(L)
        sz = [1 + Hmax - Hmin, 1 + Kmax - Kmin, 1 + Lmax - Lmin]
        HKL = np.ravel_multi_index((H-Hmin,K-Kmin,L-Lmin),sz)
        HKLu, HKLind, counts = np.unique(HKL,return_inverse=True,return_counts=True)
        Hu,Ku,Lu = np.unravel_index(HKLu,sz)
        Hu,Ku,Lu = Hu+Hmin,Ku+Kmin,Lu+Lmin
        h,k,l = Hu/ndiv[0], Ku/ndiv[1], Lu/ndiv[2]
        outcols = {}
        if count_name is not None:
            outcols[count_name] = counts
        for key in column_names:
            outcols[key] = np.bincount(HKLind, weights=self.__dict__[key])
        return HKLTable(h,k,l,_ndiv=ndiv,**outcols)

    @staticmethod
    def concatenate(tabs):
        _ndiv = tabs[0]._ndiv # assume the first one is canonical
        data_keys = set(tabs[0]._data_keys)
        for j in range(1,len(tabs)):
            data_keys = data_keys.intersection(data_keys)
        def concat(key):
            return np.concatenate([tab.__dict__[key] for tab in tabs])
        h = concat('h')
        k = concat('k')
        l = concat('l')
        cols = {key:concat(key) for key in data_keys}
        return HKLTable(h,k,l,_ndiv=_ndiv,**cols)

    @staticmethod
    def from_frame(df):
        """ create from pandas dataframe with h,k,l as cols or indices """
        df = df.reset_index() # move indices into columns
        h = df.pop('h').to_numpy()
        k = df.pop('k').to_numpy()
        l = df.pop('l').to_numpy()
        data = {key:df[key].to_numpy() for key in df.keys()}
        return HKLTable(h,k,l,**data)

    def to_nexus(self):
        return NXgroup(name='hkl_table',**self.__dict__)

    @staticmethod
    def from_nexus(nxgroup):
        h = nxgroup.h.nxdata
        k = nxgroup.k.nxdata
        l = nxgroup.l.nxdata
        data_keys = [key for key in nxgroup.keys() if key not in ['h','k','l']]
        data = {key:nxgroup[key].nxdata for key in data_keys}
        return HKLTable(h,k,l,**data)

class ImageSeries:
    """Image stack resulting from single sweep of data"""

    def __init__(self,phi,iy,ix,data,exposure_times,maskval=-1):
        self.phi = phi
        self.iy = iy
        self.ix = ix
        self.data = data # can be NXfield or numpy array, doesn't matter
        self.exposure_times = exposure_times
        self._maskval = maskval

    @property
    def shape(self):
        return (self.phi.size, self.iy.size, self.ix.size)

    def __getitem__(self,sl):
        return ImageSeries(
            self.phi[sl[0]],
            self.iy[sl[1]],
            self.ix[sl[2]],
            self._as_np(self.data[sl]), # force a read and set to numpy array
            self.exposure_times[sl[0]],
            )

    @staticmethod
    def _as_np(data):
        if isinstance(data,NXfield):
            return data.nxdata
        else:
            return data

    @property
    def data_masked(self):
        return np.ma.masked_equal(self._as_np(self.data),self._maskval,copy=False)

    def bin_down(self,bins,valid_range=None,count_rate=True):
        bins = np.array(bins)
        nbins = np.ceil(self.shape/bins).astype(int)
        nadd = nbins*bins - self.shape

        def _bin_axis(ax,bin,nbin):
            nadd = bin*nbin - ax.size
            tmp = np.pad(ax.astype(float),((0,nadd)),'constant', constant_values=np.nan)
            tmp = tmp.reshape((nbin,bin))
            return np.nanmean(tmp,axis=1)

        new_phi = _bin_axis(self.phi,bins[0],nbins[0])
        new_ix = _bin_axis(self.ix,bins[2],nbins[2])
        new_iy = _bin_axis(self.iy,bins[1],nbins[1])

        new_data = np.empty(nbins,dtype=float)
        for index, start in enumerate(range(0,nbins[0]*bins[0],bins[0])):
            stop = min(start + bins[0],self.shape[0])
            print(f'binning frames {start} to {stop-1}')
            tmp = self._as_np(self.data[start:stop,:,:])
            pad_width = ((0,0),(0,nadd[1]),(0,nadd[2]))
            tmp = np.pad(tmp.astype(float),pad_width,'constant', constant_values=np.nan)
            if valid_range is not None:
                tmp = np.ma.masked_outside(tmp,valid_range[0],valid_range[1],copy=False)
            tmp = tmp.reshape((bins[0],nbins[1],bins[1],nbins[2],bins[2]))
            new_data[index,:,:] = np.nanmean(tmp,axis=(0,2,4))

        if count_rate:
            new_times = _bin_axis(self.exposure_times,bins[0],nbins[0])
            new_data = new_data/new_times[:,np.newaxis,np.newaxis]

        return GridData((new_phi,new_iy,new_ix),new_data)


    @property
    def chunks(self):
        if isinstance(self.data,NXfield):
            ch = self.data.chunks
            if ch is not None:
                return ch
        # return default chunks
        return (1,self.shape[1],self.shape[2])

    def iter_chunks(self):
        for sl in self.chunk_slice_iterator():
            yield self[sl]

    def chunk_slice_along_axis(self,axis=0):
        c = self.chunks[axis]
        n = self.shape[axis]
        start = range(0,n,c)
        stop = [min(st+c,n) for st in start]
        return [slice(st,sp) for (st,sp) in zip(start,stop)]

    def chunk_slice_iterator(self):
        s1 = self.chunk_slice_along_axis(axis=0)
        s2 = self.chunk_slice_along_axis(axis=1)
        s3 = self.chunk_slice_along_axis(axis=2)
        for a1 in s1:
            for a2 in s2:
                for a3 in s3:
                    yield (a1,a2,a3)

    @staticmethod
    def from_expt(exptfile):
        data_opts={'dtype':np.int32,'compression':'gzip','compression_opts':1,'shuffle':True,'chunks':True}
        E = Experiment.from_file(exptfile)
        phi,iy,ix = E.scan_axes
        shape=(phi.size,iy.size,ix.size)
        data = NXfield(shape=shape,name='data',**data_opts)
        exposure_times = E.exposure_times
        return ImageSeries(phi,iy,ix,data,exposure_times)

    @staticmethod
    def from_nexus(nxdata):
        phi = nxdata.phi.nxvalue
        iy = nxdata.iy.nxvalue
        ix = nxdata.ix.nxvalue
        exposure_times = nxdata.exposure_times.nxvalue
        data = nxdata[nxdata.attrs['signal']] # keep the nexus object intact to allow chunked io
        return ImageSeries(phi,iy,ix,data,exposure_times)

    def to_nexus(self):
        phi = NXfield(self.phi,name='phi')
        ix = NXfield(self.ix,name='ix')
        iy = NXfield(self.iy,name='iy')
        if isinstance(self.data,NXfield):
            signal = self.data
        else:
            signal = NXfield(self.data,name='data')
        return NXdata(signal=signal,axes=[phi,iy,ix],exposure_times=self.exposure_times)

    def find_peaks_above_threshold(self,threshold,verbose=True):
        """find pixels above a threshold"""
        peaklist = []
        for ind,ims in enumerate(self.iter_chunks()):
            im_data = ims.data_masked
            peaks = Peaks.where(im_data>threshold,ims.phi,ims.iy,ims.ix)
            if peaks.size:
                if verbose: print(f'found {peaks.size} peaks in chunk {ind}')
                peaklist.append(peaks)
        peaks = Peaks.stack(peaklist)
        if verbose: print(f'found {peaks.size} peaks in total')
        return peaks

    def index(self,miller_index,mask=None):
        mi = miller_index.regrid(self.phi,self.iy,self.ix)
        phi,iy,ix = np.meshgrid(self.phi,self.iy,self.ix,indexing='ij')
        # HACK to get this the right shape...
        exposure_times = np.tile(self.exposure_times,(self.iy.size,self.ix.size,1))
        exposure_times = np.moveaxis(exposure_times,2,0)
        msk = self.data!=self._maskval
        if mask is not None:
            msk = msk & ~mask # mask is true when pixels are excluded
        return HKLTable(
            mi.h[msk],
            mi.k[msk],
            mi.l[msk],
            phi=phi[msk],
            iy=iy[msk],
            ix=ix[msk],
            counts=self.data[msk],
            seconds=exposure_times[msk],
            )
