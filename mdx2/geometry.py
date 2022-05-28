import numpy as np
import re
import numexpr as ne

from nexusformat.nexus import * # NXsample, NXcollection, NXdata, NXfield

from mdx2.dxtbx_machinery import Experiment
from mdx2.utils import interp_g2g_trilinear, interp3, interp2

class GaussianPeak:
    """Handy functions for 3d Gaussians"""

    def __init__(self,r0,sigma):
        self.r0 = r0
        self.sigma = sigma

    def __str__(self):
        return f'r0 = {self.r0}\nsigma =\n{self.sigma}'

    @staticmethod
    def fit_to_points(x,y,z,sigma_cutoff=1):
        """fit quadratic form to a set of points"""
        r = np.stack((x,y,z)).T
        r0 = np.mean(r,axis=0)
        u,s,vh = np.linalg.svd(r-r0,full_matrices=False)
        v = vh.T
        sigma = v @ np.diag(s/np.sqrt(u.shape[0]))

        GP = GaussianPeak(r0,sigma)

        if sigma_cutoff is None:
            return GP
        else:
            d2 = GP.quadform_at_points(x,y,z)
            is_outlier = d2 > sigma_cutoff**2
            x = x[~is_outlier]
            y = y[~is_outlier]
            z = z[~is_outlier]
            return GaussianPeak.fit_to_points(x,y,z,sigma_cutoff=None), is_outlier

    def quadform_at_points(self,x,y,z):
        r = np.stack((x,y,z)).reshape(3,-1)
        d = np.linalg.inv(self.sigma) @ (r - self.r0[:,np.newaxis])
        return np.sum(d*d,axis=0).reshape(x.shape)

    def mask(self,x,y,z,sigma_cutoff=1):
        # assume x,y,z are all the same shape
        d2 = self.quadform_at_points(x,y,z)
        return d2 < sigma_cutoff**2

    @staticmethod
    def from_nexus(gp):
        return GaussianPeak(np.array(gp.r0),np.array(gp.sigma))

    def to_nexus(self):
        return NXgroup(name='gaussian_peak',sigma=self.sigma,r0=self.r0)


class GridData:

    def __init__(self,axes,data,axes_names=None):
        self.axes = axes # tuple of length 3 with axes of length l,m,n
        self.data = data # stack of arrays of equal size: l by m by n by K
        if axes_names is None:
            axes_names = [f'axis{j}' for j in range(len(axes))]
        self.axes_names = axes_names

    def regrid(self,new_axes):
        new_data = interp_g2g_trilinear(*self.axes,self.data,*new_axes)
        return GridData(new_axes,new_data)

    @property
    def ndims(self):
        return len(self.axes)

    @property
    def ncols(self):
        if len(self.data.shape) == self.ndims:
            return 1
        else:
            return self.data.shape[self.ndims]

    @property
    def data_unstacked(self):
        if self.ncols==1: # single 3D array
            return [self.data]
        else:
            return [self.data[...,ind] for ind in range(self.ncols)]

    def split(self):
        return [GridData(self.axes,data) for data in self.data_unstacked]

    def interpolate(self,*args,**kwargs):
        outdata = []
        for col in self.data_unstacked:
            if self.ndims==3:
                vals = interp3(*self.axes,col,*args,**kwargs)
            elif self.ndims==2:
                vals = interp2(*self.axes,col,*args,**kwargs)
            else:
                raise(RuntimeError('interpolation not implemented yet for dimensions other and 2 or 3'))
            outdata.append(vals)
        if len(outdata) == 1:
            return outdata[0]
        else:
            return np.stack(outdata,axis=len(args[0].shape))

    def to_nexus(self):
        axes = [NXfield(ax,name=n) for ax,n in zip(self.axes,self.axes_names)]
        return NXdata(self.data,axes)

    @staticmethod
    def from_nexus(nxdata):
        data_name = nxdata.attrs['signal']
        data = nxdata[data_name].nxvalue
        axes_names = nxdata.attrs['axes']
        axes = [nxdata[name].nxvalue for name in axes_names]
        return GridData(axes,data,axes_names=axes_names)

    def bin(self,new_axes):
        pass
        #edges = [np.concatenate([-np.inf],ax[1:]*0.5 + ax[:-1]*0.5,[np.inf]) for ax in new_axes]
        #indices = [np.searchsorted(edge,axis) - 1 for zip(edges,self.axes)]

        # meshgrid
        # ravel_multi_index
        # bincount (loop weights over K)
        # reshape and return

        # or...
        # split along 0
        # sum along 0 for each in split
        # split along 1,
        # sum along 1 for each in split
        # ...

        # <---- LEFT OFF HERE ---->



class MillerIndex(GridData):
    """array of Miller indices"""
    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self,phi,iy,ix,h,k,l):
        self.phi = phi
        self.iy = iy
        self.ix = ix
        self.h = h
        self.k = k
        self.l = l

    @property
    def data(self):
        return np.stack((self.h,self.k,self.l),axis=3)

    @property
    def axes(self):
        return (self.phi,self.iy,self.ix)

    def regrid(self,new_phi,new_iy,new_ix):
        new_axes = (new_phi,new_iy,new_ix)
        G3D = GridData.regrid(self,new_axes)
        hkl = tuple(G3D.data_unstacked)
        return MillerIndex(*new_axes,*hkl)

    def interpolate(self,phi,iy,ix,**kwargs):
        hkl = GridData.interpolate(self,phi,iy,ix,**kwargs)
        return hkl[...,0], hkl[...,1], hkl[...,2]

    @staticmethod
    def from_expt(exptfile,sample_spacing=[1,10,10]):
        # sample spacing (phi, y, x) in units of (degrees, pixels, pixels)
        E = Experiment.from_file(exptfile)
        phi,iy,ix = E.calc_scan_axes(centered=False,spacing=sample_spacing)
        h,k,l = E.calc_hkl_on_grid(phi,iy,ix)
        return MillerIndex(phi,iy,ix,h,k,l)

    def to_nexus(self):
        ix = NXfield(self.ix,name='ix')
        iy = NXfield(self.iy,name='iy')
        phi = NXfield(self.phi,name='phi',units='degree')
        axes = [phi,iy,ix]
        nxgroup = NXgroup(
            name='miller_index',
            h=NXdata(self.h,axes),
            k=NXdata(self.k,axes),
            l=NXdata(self.l,axes),
        )
        return nxgroup

    @staticmethod
    def from_nexus(nxgroup):
        get_value = lambda nxdata: nxdata[nxdata.attrs['signal']].nxvalue
        phi = nxgroup.h.phi.nxvalue
        iy = nxgroup.h.iy.nxvalue
        ix = nxgroup.h.ix.nxvalue

        return MillerIndex(
            phi,
            iy,
            ix,
            get_value(nxgroup.h),
            get_value(nxgroup.k),
            get_value(nxgroup.l),
        )


class Corrections(GridData):
    """geometric corrections"""
    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self,iy,ix,
        solid_angle=None,
        efficiency=None,
        attenuation=None,
        polarization=None,
        inverse_lorentz=None,
        d3s=None,
        ):

        self.iy = iy
        self.ix = ix
        self.solid_angle=solid_angle
        self.efficiency=efficiency
        self.attenuation=attenuation
        self.polarization=polarization
        self.inverse_lorentz=inverse_lorentz
        self.d3s=d3s

    _data_keys = ['solid_angle','efficiency','attenuation','polarization','inverse_lorentz','d3s']

    @property
    def data(self):
        all_data = [self.__dict__[key] for key in self._data_keys]
        return np.stack(all_data,axis=2)

    @property
    def axes(self):
        return (self.iy,self.ix)

    def interpolate(self,iy,ix,order=1):
        outdata = GridData.interpolate(self,iy,ix,order=order)
        kvpairs = {key:outdata[...,ind] for ind,key in enumerate(self._data_keys)}
        return kvpairs

    def regrid(self,new_iy,new_ix):
        new_axes = (new_iy,new_ix)
        G = GridData.regrid(self,new_axes)
        kvpairs = {key:val for key,val in zip(self._data_keys,G.data_unstacked)}
        return Corrections(*G.axes,**kvpairs)

    @staticmethod
    def from_expt(exptfile,sample_spacing=[10,10]):
        # sample spacing (y, x) in units of pixels
        E = Experiment.from_file(exptfile)
        _,iy,ix = E.calc_scan_axes(centered=False,spacing=(1,)+tuple(sample_spacing))
        corr = E.calc_corrections_on_grid(iy,ix)
        return Corrections(iy,ix,**corr)

    def to_nexus(self):
        ix = NXfield(self.ix,name='ix')
        iy = NXfield(self.iy,name='iy')
        axes = [iy,ix]
        nxgroup = NXgroup(
            name='corrections',
            solid_angle=NXdata(self.solid_angle,axes),
            efficiency=NXdata(self.efficiency,axes),
            attenuation=NXdata(self.attenuation,axes),
            polarization=NXdata(self.polarization,axes),
            inverse_lorentz=NXdata(self.inverse_lorentz,axes),
            d3s=NXdata(self.d3s,axes),
        )
        return nxgroup

    @staticmethod
    def from_nexus(nxgroup):
        get_value = lambda nxdata: nxdata[nxdata.attrs['signal']].nxvalue
        iy = nxgroup.solid_angle.iy.nxvalue
        ix = nxgroup.solid_angle.ix.nxvalue
        return Corrections(
            iy,
            ix,
            solid_angle=get_value(nxgroup.solid_angle),
            efficiency=get_value(nxgroup.efficiency),
            attenuation=get_value(nxgroup.attenuation),
            polarization=get_value(nxgroup.polarization),
            inverse_lorentz=get_value(nxgroup.inverse_lorentz),
            d3s=get_value(nxgroup.d3s),
        )

    def __str__(self):
        return str(self.__dict__)


class Symmetry:
    """space group symmetry information"""
    # This is a container class for derived space group info, mainly to handle file format conversion

    def __init__(self,**kwargs):
        # lazy constructor....
        self.__dict__.update(kwargs)

    @property
    def _asu_test_np(self):
        # make substitutions for numpy compatibility
        str = self.reciprocal_space_asu
        str = re.sub(r'([hkl0\<\>\=]+)',r'(\1)',str)
        str = re.sub('and',r'&',str)
        str = re.sub('or',r'|',str)
        return str

    def is_asu(self,h,k,l):
        if not isinstance(h,np.ndarray):
            h = np.array(h)
            k = np.array(k)
            l = np.array(l)
        return ne.evaluate(
            self._asu_test_np,
            local_dict={'h':h.astype(int),'k':k.astype(int),'l':l.astype(int)},
            )

    def to_asu(self,h,k,l):

        if not isinstance(h,np.ndarray):
            h = np.array(h)
            k = np.array(k)
            l = np.array(l)

        opindex = np.empty_like(h,dtype=int)
        ops = self.laue_group_operators

        hkl_in = np.stack((h,k,l))
        hkl_out = np.empty_like(hkl_in)

        for n in reversed(range(ops.shape[0])):
            op = ops[n,:]
            hkl_n = np.tensordot(op,hkl_in,axes=1)
            is_asu = self.is_asu(hkl_n[0,...],hkl_n[1,...],hkl_n[2,...])
            opindex[is_asu] = n
            hkl_out[:,is_asu] = hkl_n[:,is_asu]

        return hkl_out[0,...], hkl_out[1,...], hkl_out[2,...], opindex




    def to_nexus(self):
        return NXgroup(
            name='symmetry',
            space_group_number=self.space_group_number,
            space_group_symbol=self.space_group_symbol,
            crystal_system=self.crystal_system,
            space_group_operators=self.space_group_operators,
            laue_group_number=self.laue_group_number,
            laue_group_symbol=self.laue_group_symbol,
            laue_group_operators=self.laue_group_operators,
            reciprocal_space_asu=self.reciprocal_space_asu,
        )

    @staticmethod
    def from_nexus(obj):
        return Symmetry(
            space_group_number=obj.space_group_number.nxvalue,
            space_group_symbol=obj.space_group_symbol.nxvalue,
            crystal_system=obj.crystal_system.nxvalue,
            space_group_operators=obj.space_group_operators.nxvalue,
            laue_group_number=obj.laue_group_number.nxvalue,
            laue_group_symbol=obj.laue_group_symbol.nxvalue,
            laue_group_operators=obj.laue_group_operators.nxvalue,
            reciprocal_space_asu=obj.reciprocal_space_asu.nxvalue,
        )

    @staticmethod
    def from_expt(exptfile):
        E = Experiment.from_file(exptfile)
        return Symmetry(
            space_group_number=E.space_group_number,
            space_group_symbol=E.space_group_symbol,
            crystal_system=E.crystal_system,
            space_group_operators=E.space_group_operators,
            laue_group_number=E.laue_group_number,
            laue_group_symbol=E.laue_group_symbol,
            laue_group_operators=E.laue_group_operators,
            reciprocal_space_asu=E.reciprocal_space_asu,
        )

    def __str__(self):
        return str(self.__dict__)

class Crystal:
    """unit cell information"""
    # This is a container class for unit cell info, to handle file format conversion
    #
    # see also:
    #   https://manual.nexusformat.org/classes/base_classes/NXsample.html
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_nexus(sample):
        return Crystal(
            space_group=sample.space_group.nxvalue,
            unit_cell=sample.unit_cell.nxvalue,
            orientation_matrix=sample.orientation_matrix.nxvalue,
            ub_matrix=sample.ub_matrix.nxvalue,
        )

    def to_nexus(self):
        return NXsample(
            name='crystal',
            space_group=self.space_group,
            unit_cell=self.unit_cell,
            orientation_matrix=self.orientation_matrix,
            ub_matrix=self.ub_matrix,
        )

    @staticmethod
    def from_expt(exptfile):
        E = Experiment.from_file(exptfile)
        return Crystal(
            space_group=E.space_group,
            unit_cell=E.unit_cell,
            orientation_matrix=E.orientation_matrix,
            ub_matrix=E.ub_matrix,
        )

    def __str__(self):
        return str(self.__dict__)
