from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nexusformat.nexus import NXfield, NXdata, NXgroup, NXreflections

from mdx2.dxtbx_machinery import Experiment
from mdx2.geometry import GridData
from mdx2.utils import slice_sections

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
    def __init__(self,h,k,l,ndiv=[1,1,1],**kwargs):
        self.h = h
        self.k = k
        self.l = l
        self.ndiv = ndiv
        self.__dict__.update(kwargs)

    @property
    def _data_keys(self):
        return [key for key in self.__dict__ if key not in ['h','k','l','ndiv']]

    def __len__(self):
        return len(self.h)

    def to_frame(self):
        """ convert to pandas dataframe """
        cols = {k:self.__dict__[k] for k in self.__dict__ if k not in ['ndiv']}
        return pd.DataFrame(cols)

    def to_asu(self,Symmetry=None):
        """Map to asymmetric unit. If Symmetry is ommitted, P1 space group is assumed (op = 0 for l>=0, op=1 for l<0)"""
        ndiv = self.ndiv
        H = np.round(self.h*ndiv[0]).astype(int)
        K = np.round(self.k*ndiv[1]).astype(int)
        L = np.round(self.l*ndiv[2]).astype(int)

        if Symmetry is None:
            op = (L<0).astype(int)
            L = np.abs(L)
        else:
            H,K,L,op = Symmetry.to_asu(H,K,L)

        newtable = deepcopy(self)
        newtable.h = H/ndiv[0]
        newtable.k = K/ndiv[1]
        newtable.l = L/ndiv[2]
        newtable.op = op

        return newtable

    @property
    def H(self):
        return np.round(self.h*self.ndiv[0]).astype(int)

    @property
    def K(self):
        return np.round(self.k*self.ndiv[1]).astype(int)

    @property
    def L(self):
        return np.round(self.l*self.ndiv[2]).astype(int)

    def to_array_index(self,ori=None):
        H, K, L = self.H, self.K, self.L
        if ori is None:
            O = (np.min(H),np.min(K),np.min(L))
        else:
            O = tuple(np.array(self.ndiv)*np.array(ori))
        return (H - O[0],K - O[1],L - O[2]),O

    def from_array_index(self,index,shape,O=[0,0,0]):
        hklind = np.unravel_index(index,shape)
        hkl = [(ind + o)/n for ind,o,n in zip(hklind,O,self.ndiv)]
        return tuple(hkl)

    def unique(self):
        ind, O = self.to_array_index()
        sz = [np.max(j)+1 for j in ind]
        unique_index, index_map, counts = np.unique(
            np.ravel_multi_index(ind,sz),
            return_inverse=True,
            return_counts=True,
            )
        hkl = self.from_array_index(unique_index,sz,O=O)
        return hkl, index_map, counts

    def bin(self,column_names=None,count_name='count'):
        
        if column_names is None:
            column_names = self._data_keys
            
        # catch the case where the HKLTable is empty, and return an empty table with count_name field
        # (used by integrate)
        if len(self) == 0:
            outcols = {k:[] for k in column_names}
            if count_name is not None:
                outcols[count_name] = np.array([],dtype=np.int64) # np.unique returns counts as int64 dtype
            return HKLTable([],[],[],ndiv=self.ndiv,**outcols)

        #print('finding unique indices')
        (h,k,l), index_map, counts = self.unique()

        outcols = {}
        if count_name is not None:
            #print(f'storing bin counts in column: {count_name}')
            outcols[count_name] = counts
        for key in column_names:
            #print(f'binning data column: {key}')
            outcols[key] = np.bincount(index_map, weights=self.__dict__[key])
        return HKLTable(h,k,l,ndiv=self.ndiv,**outcols)

    @staticmethod
    def concatenate(tabs):
        ndiv = tabs[0].ndiv # assume the first one is canonical
        data_keys = set(tabs[0]._data_keys)
        for j in range(1,len(tabs)):
            data_keys = data_keys.intersection(data_keys)
        def concat(key):
            return np.concatenate([tab.__dict__[key] for tab in tabs])
        h = concat('h')
        k = concat('k')
        l = concat('l')
        cols = {key:concat(key) for key in data_keys}
        return HKLTable(h,k,l,ndiv=ndiv,**cols)

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
        #self.phi = phi
        #self.iy = iy
        #self.ix = ix
        self.phi = np.atleast_1d(phi)
        self.iy = np.atleast_1d(iy)
        self.ix = np.atleast_1d(ix)
        self.exposure_times = np.atleast_1d(exposure_times)
        self.data = data # can be NXfield or numpy array, doesn't matter
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

    def bin_down(self,bins,valid_range=None,count_rate=True,nproc=1,mask=None):

        bins = np.array(bins)
        nbins = np.ceil(self.shape/bins).astype(int)
        sl_0 = slice_sections(self.shape[0],nbins[0])
        sl_1 = slice_sections(self.shape[1],nbins[1])
        sl_2 = slice_sections(self.shape[2],nbins[2])
        
        new_phi = np.array([self.phi[sl].mean() for sl in sl_0])
        new_iy = np.array([self.iy[sl].mean() for sl in sl_1])
        new_ix = np.array([self.ix[sl].mean() for sl in sl_2])

        def binslab(sl):
            outslab = np.empty([len(sl_1),len(sl_2)],dtype=float)
            tmp = self._as_np(self.data[sl,:,:])
            tmp = np.ma.masked_equal(tmp,self._maskval,copy=False)
            if valid_range is not None:
                tmp = np.ma.masked_outside(tmp,valid_range[0],valid_range[1],copy=False)
            if mask is not None:
                msk = self._as_np(mask[sl,:,:])
                tmp = np.ma.masked_where(msk,tmp,copy=False)
            for ind_y,sl_y in enumerate(sl_1): # not vectorized - could be slow?
                for ind_x,sl_x in enumerate(sl_2):
                    val = tmp[:,sl_y,sl_x].mean()
                    if isinstance(val,np.ma.masked_array):
                        val = np.nan
                    outslab[ind_y,ind_x] = val
            return outslab

        if nproc==1:
            slabs = []
            for ind,sl in enumerate(sl_0):
                print(f'binning frames {sl.start} to {sl.stop-1}')
                slabs.append(binslab(sl))
            new_data = np.stack(slabs)
        else:
            with Parallel(n_jobs=nproc,verbose=10) as parallel:
                new_data = np.stack(parallel(delayed(binslab)(sl) for sl in sl_0))

        if count_rate:
            new_times = np.array([self.exposure_times[sl].mean() for sl in sl_0])
            new_data = new_data/new_times[:,np.newaxis,np.newaxis]

        return GridData((new_phi,new_iy,new_ix),new_data,axes_names=['phi','iy','ix'])


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

#    def find_peaks_above_threshold(self,threshold,verbose=True):
#        """find pixels above a threshold"""
#        peaklist = []
#        for ind,ims in enumerate(self.iter_chunks()):
#            im_data = ims.data_masked
#            peaks = Peaks.where(im_data>threshold,ims.phi,ims.iy,ims.ix)
#            if peaks.size:
#                if verbose: print(f'found {peaks.size} peaks in chunk {ind}')
#                peaklist.append(peaks)
#        peaks = Peaks.stack(peaklist)
#        if verbose: print(f'found {peaks.size} peaks in total')
#        return peaks

    def find_peaks_above_threshold(self,threshold,verbose=True,nproc=1):
        """find pixels above a threshold"""
        def peaksearch(sl):
            ims = self[sl]
            im_data = ims.data_masked
            peaks = Peaks.where(im_data>threshold,ims.phi,ims.iy,ims.ix)
            if peaks.size:
                return peaks

        if nproc==1:
            peaklist = []
            for ind,sl in enumerate(self.chunk_slice_iterator()):
                peaks = peaksearch(sl)
                if peaks is not None:
                    if verbose: print(f'found {peaks.size} peaks in chunk {ind}')
                    peaklist.append(peaks)
            peaks = Peaks.stack(peaklist)
        else:
            with Parallel(n_jobs=nproc,verbose=10) as parallel:
                peaklist = parallel(delayed(peaksearch)(sl) for sl in self.chunk_slice_iterator())
            peaks = Peaks.stack([p for p in peaklist if p is not None])
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
