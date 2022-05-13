import numpy as np

from nexusformat.nexus import NXsample, NXcollection, NXdata, NXfield, NXentry
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

def fit_quadform_to_points(x,y,z,sigma_cutoff=3):
    G3,is_outlier = Gaussian3.fit_to_points(x,y,z,sigma_cutoff=sigma_cutoff)
    return G3.r0, G3.sigma, is_outlier

#def fit_quadform_to_points(x,y,z,sigma_cutoff=3):
    # """fit quadratic form to a set of points with one step of outlier rejection"""
    # r = np.stack((x,y,z)).T
    # r0 = np.mean(r,axis=0)
    # u,s,vh = np.linalg.svd(r-r0,full_matrices=False)
    # is_outlier = np.any(np.abs(u*np.sqrt(u.shape[0])) > sigma_cutoff,axis=1)
    # # remove outlier points and fit again
    # r = r[~is_outlier,:];
    # r0 = np.mean(r,axis=0)
    # u,s,vh = np.linalg.svd(r-r0,full_matrices=False)
    # v = vh.T
    # sigma = v @ np.diag(s/np.sqrt(u.shape[0]))
    # return r0, sigma, is_outlier
#
# def eval_quadform_at_points(r0,sigma,x,y,z):
#     r = np.stack((x,y,z))
#     d = np.linalg.inv(sigma) @ (r - r0[:,np.newaxis])
#     return np.sum(d*d,axis=0)

class Gaussian3:
    """Handy functions for 3d Gaussians"""

    def __init__(self,r0,sigma):
        self.r0 = r0
        self.sigma = sigma

    @staticmethod
    def fit_to_points(x,y,z,sigma_cutoff=None):
        """fit quadratic form to a set of points"""
        r = np.stack((x,y,z)).T
        r0 = np.mean(r,axis=0)
        u,s,vh = np.linalg.svd(r-r0,full_matrices=False)
        v = vh.T
        sigma = v @ np.diag(s/np.sqrt(u.shape[0]))

        G3 = Gaussian3(r0,sigma)

        if sigma_cutoff is not None:
            d2 = G3.quadform_at_points(x,y,z)
            is_outlier = d2 > sigma_cutoff**2
            x = x[~is_outlier]
            y = y[~is_outlier]
            z = z[~is_outlier]
            return Gaussian3.fit_to_points(x,y,z,sigma_cutoff=None), is_outlier
        else:
            return G3

    def quadform_at_points(self,x,y,z):
        r = np.stack((x,y,z))
        d = np.linalg.inv(self.sigma) @ (r - self.r0[:,np.newaxis])
        return np.sum(d*d,axis=0)


class MillerIndex:
    """array of Miller indices"""
    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def interpolate(self,phi,iy,ix,order=1):
        # faster version of scipy.interpolate.interpn using map_coordinates
        phi_index = np.interp(phi,self.phi,np.arange(self.phi.size))
        ix_index = np.interp(ix,self.ix,np.arange(self.ix.size))
        iy_index = np.interp(iy,self.iy,np.arange(self.iy.size))
        coordinates = np.stack((phi_index,iy_index,ix_index))
        h = map_coordinates(self.h,coordinates,order=order)
        k = map_coordinates(self.k,coordinates,order=order)
        l = map_coordinates(self.l,coordinates,order=order)
        return h,k,l

#    @property
#    def interpolator(self):
#        axes = (self.phi,self.iy,self.ix)
#        hinterp = RegularGridInterpolator(axes,self.h)
#        kinterp = RegularGridInterpolator(axes,self.k)
#        linterp = RegularGridInterpolator(axes,self.l)
#        return lambda phi,iy,ix: (hinterp((phi,iy,ix)),kinterp((phi,iy,ix)),linterp((phi,iy,ix)))

#    def interpolate(self,phi,iy,ix):
#        axes = (self.phi,self.iy,self.ix)
#        points = (phi,iy,ix)
#        h = interpn(axes,self.h,points)
#        k = interpn(axes,self.k,points)
#        l = interpn(axes,self.l,points)
#        return h, k, l

    @staticmethod
    def from_dxtbx_experiment(expt,sampling=(1,10,10)):
        # sampling (phi, y, x) in units of (degrees, pixels, pixels)

        def calc_rotation_matrix_at_phi(goniometer,phi_vals_deg):
            from scitbx import matrix
            assert goniometer.num_scan_points == 0 # scan varying settings not implemented here
            # get the S matrix
            S = goniometer.get_setting_rotation()
            S = np.array(S).reshape([3,3])
            # get the F matrix
            F = goniometer.get_fixed_rotation()
            F = np.array(F).reshape([3,3])
            # get the R matrix at each angle
            axis = matrix.col(goniometer.get_rotation_axis_datum())
            R = np.empty([len(phi_vals_deg),3,3])
            for ind, phi in enumerate(phi_vals_deg):
                Rvals = tuple(axis.axis_and_angle_as_r3_rotation_matrix(phi,deg=True))
                R[ind,:,:] = np.array(Rvals).reshape(3,3)
            # multiply
            return S @ R @ F

        def interp_rotation_matrices(U1,U2,rotation_fraction):
            # U1, U2 represented as a (flat) tuple, returns a (flat) tuple
            from scitbx import matrix
            # compute the operator that rotates from from U1 to U2
            M = matrix.sqr(U2) * matrix.sqr(U1).transpose()
            Mq = M.r3_rotation_matrix_as_unit_quaternion()
            angle, axis = Mq.unit_quaternion_as_axis_and_angle(deg=False)
            # interpolate the rotation
            M_frac = axis.axis_and_angle_as_r3_rotation_matrix(angle*rotation_fraction, deg=False)
            return tuple(M_frac*matrix.sqr(U1))

        def calc_U_matrix_at_phi(crystal,scan,phi_vals_deg):
            assert crystal.num_scan_points > 0 # use for a scan-varying model only
            base, frac = phi_to_base_fraction_index(scan,phi_vals_deg)
            U = np.empty([np.size(base),3,3])
            for ind, (b, f) in enumerate(zip(base, frac)):
                U1 = crystal.get_U_at_scan_point(b)
                U2 = crystal.get_U_at_scan_point(b + 1)
                Uvals = interp_rotation_matrices(U1,U2,f)
                U[ind,:,:] = np.array(Uvals).reshape(3,3)
            return U

        def phi_to_base_fraction_index(scan,phi_vals_deg):
            # get integer scan index and fractional part for each angle
            # the fractional part is between zero and one for phi angles within the
            # scan range. the integer part can range from 0,(n-1) where n is the
            # maximum array index
            imin, imax = scan.get_array_range()
            b = []
            f = []
            for ind, phi in enumerate(phi_vals_deg):
                scan_index = scan.get_array_index_from_angle(phi)
                scan_index_base = int(scan_index)
                if scan_index_base >= imax:
                    scan_index_base = imax - 1
                elif scan_index_base < imin:
                    scan_index_base = imin
                b.append(scan_index_base)
                f.append(scan_index - scan_index_base)
            return b, f

        def calc_B_matrix_at_phi(crystal,scan,phi_vals_deg):
            assert crystal.num_scan_points > 0 # use for a scan-varying model only
            base, frac = phi_to_base_fraction_index(scan,phi_vals_deg)
            B = np.empty([np.size(base),3,3])
            for ind, (b, f) in enumerate(zip(base, frac)):
                B1 = np.array(crystal.get_B_at_scan_point(b)).reshape([3,3])
                B2 = np.array(crystal.get_B_at_scan_point(b + 1)).reshape([3,3])
                B[ind,:,:] = B1*(1-f) + B2*f # element-wise linear interpolation
            return B

        def index_grid_to_lab(panel,ix,iy):
            from dxtbx import flumpy
            xy = np.dstack(np.meshgrid(ix, iy)) # ny-by-nx-by-2
            sz = xy.shape[:-1]
            xy = flumpy.vec_from_numpy(np.double(xy))
            xymm = panel.pixel_to_millimeter(xy)
            xyz = panel.get_lab_coord(xymm)
            xyz = xyz.as_numpy_array()
            xyz = xyz.reshape(sz + (3,))
            xyz = np.moveaxis(xyz,2,0)
            return xyz

        crystal = expt.crystal
        goniometer = expt.goniometer
        panel = expt.detector[0] # single-panel detectors assumed
        scan = expt.scan
        beam = expt.beam

        # compute lab coordinates of detector pixels
        nx,ny = panel.get_image_size()
        phimin, phimax = scan.get_oscillation_range()

        ceil = lambda n,d: -1*int(-n//d)
        ix = np.linspace(-0.5,nx-0.5,1 + ceil(nx,sampling[2]))
        iy = np.linspace(-0.5,ny-0.5,1 + ceil(ny,sampling[1]))
        phi = np.linspace(phimin,phimax,1 + ceil(phimax-phimin,sampling[0]))

        xyz = index_grid_to_lab(panel,ix,iy)

        R = calc_rotation_matrix_at_phi(goniometer,phi)
        B = calc_B_matrix_at_phi(crystal,scan,phi)
        U = calc_U_matrix_at_phi(crystal,scan,phi)

        # array of orthogonalization matrices
        UB = R @ U @ B

        # array of fractionalization matrix
        invUB = np.linalg.inv(UB)

        # compute lab coordinates of detector pixels
        xyz = index_grid_to_lab(panel,ix,iy) #[iy,ix] order

        # compute s
        d = np.sqrt(np.sum(xyz*xyz,axis=0))
        wlen = beam.get_wavelength()
        s1 = xyz/d/wlen
        s0 = np.array(beam.get_s0())
        s = s1 - s0[:,np.newaxis,np.newaxis]

        hkl = np.tensordot(invUB,s,axes=1)

        return MillerIndex(
            ix=ix,
            iy=iy,
            phi=phi,
            h=hkl[:,0,:,:],
            k=hkl[:,1,:,:],
            l=hkl[:,2,:,:],
            )

    def to_nexus(self):
        ix = NXfield(self.ix,name='ix')
        iy = NXfield(self.iy,name='iy')
        phi = NXfield(self.phi,name='phi',units='degree')

        h = NXfield(self.h,name='value')
        k = NXfield(self.k,name='value')
        l = NXfield(self.l,name='value')

        return NXentry(
            h=NXdata(h,[phi,iy,ix]),
            k=NXdata(k,[phi,iy,ix]),
            l=NXdata(l,[phi,iy,ix]),
        )

    @staticmethod
    def from_nexus(entry):
        return MillerIndex(
            ix=np.array(entry.h.ix),
            iy=np.array(entry.h.iy),
            phi=np.array(entry.h.phi),
            h=np.array(entry.h.value),
            k=np.array(entry.k.value),
            l=np.array(entry.l.value),
        )


    def __str__(self):
        return str(self.__dict__)



class Corrections:
    """geometric corrections"""
    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_dxtbx_imageset(iset,sampling=(10,10)):

        def calc_mu_air(wavelength):
            """approximate attenuation coefficient of air"""
            # (mdx-lib) geom.MaterialProperties.calcMu('Air',1.0) --> 3.0012e-04
            # (this function) mu_air(1.0) --> 3.328e-04
            from cctbx.eltbx import attenuation_coefficient
            att_N = attenuation_coefficient.get_table('N')
            att_O = attenuation_coefficient.get_table('O')
            att_Ar = attenuation_coefficient.get_table('Ar')
            mu_N = att_N.mu_at_angstrom(wavelength)/10.
            mu_O = att_O.mu_at_angstrom(wavelength)/10.
            mu_Ar = att_Ar.mu_at_angstrom(wavelength)/10.
            molfrac_O2 = 0.210
            molfrac_N2 = 0.781
            molfrac_Ar = 0.009
            return molfrac_O2*mu_O + molfrac_N2*mu_N + molfrac_Ar*mu_Ar # approximately right

        def index_to_lab(panel,xy):
            from dxtbx import flumpy
            sz = xy.shape[:-1]
            xy = flumpy.vec_from_numpy(np.double(xy))
            xymm = panel.pixel_to_millimeter(xy)
            xyz = panel.get_lab_coord(xymm)
            xyz = xyz.as_numpy_array()
            xyz = xyz.reshape(sz + (3,))
            return xyz

        def calc_solid_angle(panel,cosw,d):
            # The solid angle in units of rad^2
            qx,qy = panel.get_pixel_size()
            return qx*qy*cosw/(d*d)

        def calc_efficiency(panel,cosw):
            # efficiency factor
            mu = panel.get_mu()
            t = panel.get_thickness()
            return 1 - np.exp(-mu*t/cosw)

        def calc_attenuation(beam,d):
            # attenuation due to air
            wlen = beam.get_wavelength()
            mu_air = calc_mu_air(wlen)
            return np.exp(-mu_air*d)

        def calc_polarization(beam,xyz,d):
            # polarization factor
            e0 = beam.get_unit_s0()
            pn = beam.get_polarization_normal()
            p = beam.get_polarization_fraction()
            pv1 = np.cross(e0,pn)
            pv2 = np.cross(pv1,e0)
            cos_phi1 = np.dot(xyz,pv1)/d
            cos_phi2 = np.dot(xyz,pv2)/d
            return p*(1-cos_phi1*cos_phi1) + (1-p)*(1-cos_phi2*cos_phi2)

        def calc_inverse_lorentz(beam,goniometer,xyz,d):
            m2 = goniometer.get_rotation_axis()
            e0 = beam.get_unit_s0()
            return np.abs(np.dot(np.cross(e0,xyz),m2)/d)

        def calc_d3s(beam,scan,solidAngle,Linv):
            # volume of reciprocal space swept by a pixel during a rotation frame
            phi0,dphi = scan.get_oscillation()
            wlen = beam.get_wavelength()
            dphi_radians = dphi*np.pi/180;
            return Linv*solidAngle*dphi/wlen**3

        panel = iset.get_detector()[0] # no multi-panel detectors
        goniometer = iset.get_goniometer()
        scan = iset.get_scan()
        beam = iset.get_beam()

        # compute lab coordinates of detector pixels
        nx,ny = panel.get_image_size()

        if sampling is not None:
            ceil = lambda n,d: -1*(-n//d)
            x = np.linspace(0,nx,ceil(nx,sampling[0]))
            y = np.linspace(0,ny,ceil(ny,sampling[1]))
        else:
            x = np.arange(nx)
            y = np.arange(ny)

        xy = np.dstack(np.meshgrid(x, y))
        xyz = index_to_lab(panel,xy)

        # some helpful geometry that I'll reuse
        d = np.sqrt(np.sum(xyz*xyz,axis=2))
        dnorm = panel.get_normal()
        cosw = np.dot(xyz,dnorm)/d

        solid_angle = calc_solid_angle(panel,cosw,d)
        efficiency = calc_efficiency(panel,cosw)
        attenuation = calc_attenuation(beam,d)
        polarization = calc_polarization(beam,xyz,d)
        inverse_lorentz = calc_inverse_lorentz(beam,goniometer,xyz,d)
        d3s = calc_d3s(beam,scan,solid_angle,inverse_lorentz)

        return Corrections(
            ix=x,
            iy=y,
            solid_angle=solid_angle,
            efficiency=efficiency,
            attenuation=attenuation,
            polarization=polarization,
            inverse_lorentz=inverse_lorentz,
            d3s=d3s,
        )

    def to_nexus(self):
        ix = NXfield(self.ix,name='ix')
        iy = NXfield(self.iy,name='iy')

        solid_angle = NXfield(self.solid_angle,name='value',units='steradian')
        efficiency = NXfield(self.efficiency,name='value')
        attenuation = NXfield(self.attenuation,name='value')
        polarization = NXfield(self.polarization,name='value')
        inverse_lorentz = NXfield(self.inverse_lorentz,name='value')
        d3s = NXfield(self.d3s,name='value',units='1/ang^3')

        return NXentry(
            solid_angle=NXdata(solid_angle,[iy,ix]),
            efficiency=NXdata(efficiency,[iy,ix]),
            attenuation=NXdata(attenuation,[iy,ix]),
            polarization=NXdata(polarization,[iy,ix]),
            inverse_lorentz=NXdata(inverse_lorentz,[iy,ix]),
            d3s=NXdata(d3s,[iy,ix]),
        )

    @staticmethod
    def from_nexus(entry):
        return Corrections(
            ix=entry.solid_angle.ix,
            iy=entry.solid_angle.iy,
            solid_angle=entry.solid_angle.value,
            efficiency=entry.efficiency.value,
            attenuation=entry.attenuation.value,
            polarization=entry.polarization.value,
            inverse_lorentz=entry.inverse_lorentz.value,
            d3s=entry.d3s.value,
        )

    def __str__(self):
        return str(self.__dict__)


class Symmetry:
    """space group symmetry information"""
    # This is a container class for derived space group info, mainly to handle file format conversion

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_nexus(self):
        return NXcollection(
            name='symmetry',
            #space_group_number=self.space_group_number,
            #space_group_symbol=self.space_group_symbol,
            #crystal_system=self.crystal_system,
            space_group_operators=self.space_group_operators,
            #laue_group_number=self.laue_group_number,
            #laue_group_symbol=self.laue_group_symbol,
            laue_group_operators=self.laue_group_operators,
            reciprocal_space_asu=self.reciprocal_space_asu,
        )

    @staticmethod
    def from_nexus(symmetry):
        return Symmetry(
            #space_group_number=symmetry.space_group_number.nxvalue,
            #space_group_symbol=symmetry.space_group_symbol.nxvalue,
            #crystal_system=symmetry.crystal_system.nxvalue,
            space_group_operators=symmetry.space_group_operators.nxvalue,
            #laue_group_number=symmetry.laue_group_number.nxvalue,
            #laue_group_symbol=symmetry.laue_group_symbol.nxvalue,
            laue_group_operators=symmetry.laue_group_operators.nxvalue,
            reciprocal_space_asu=symmetry.reciprocal_space_asu.nxvalue,
        )

    @staticmethod
    def from_dxtbx_crystal(crystal):
        symm = crystal.get_crystal_symmetry()
        sg = symm.space_group()
        return Symmetry.from_sgtbx_space_group(sg)

    @staticmethod
    def from_sgtbx_space_group(sg):
        """Read in / compute parameters from sgtbx.space_group object"""
        # symm = dxtbx_crystal.get_crystal_symmetry()

        # space group info
        sgi = sg.info()
        #sgt = sgi.type()
        # laue group info

        lg = sg.build_derived_laue_group()
        #lgi = lg.info()
        #lgt = lgi.type()

        # compute the operators in matrix format
        sg_ops = np.stack([np.array(op.as_double_array()).reshape(3,4) for op in sg.all_ops()],axis=0)
        lg_ops = np.stack([np.array(op.r().as_double()).reshape(3,3) for op in lg.all_ops()],axis=0)
        asu = sgi.reciprocal_space_asu().reference_as_string()

        return Symmetry(
            #space_group_number=sgt.number(),
            #space_group_symbol=sgt.lookup_symbol(),
            #crystal_system=sg.crystal_system(),
            space_group_operators=sg_ops,
            #laue_group_number=lgt.number(),
            #laue_group_symbol=sg.laue_group_type(),
            laue_group_operators=lg_ops,
            reciprocal_space_asu=asu,
            )

    def __str__(self):
        return str(self.__dict__)

class Crystal:
    """unit cell information"""
    # This is just a container class. Parameters are not checked
    # The purpose is to bridge between mdx2, nexus NXsample, and dxtbx.model.Crystal
    #
    # see:
    #   https://manual.nexusformat.org/classes/base_classes/NXsample.html
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_nexus(sample):
        return UnitCell(
            space_group=sample.space_group.nxvalue,
            parameters=sample.unit_cell.nxvalue,
            orientation_matrix=sample.orientation_matrix.nxvalue,
            ub_matrix=sample.ub_matrix.nxvalue,
        )

    def to_nexus(self):
        return NXsample(
            name='crystal',
            space_group=self.space_group,
            unit_cell=self.parameters,
            orientation_matrix=self.orientation_matrix,
            ub_matrix=self.ub_matrix,
        )

    @staticmethod
    def from_dxtbx_crystal(crystal):
        """Read in / compute parameters from dxtbx.model.Crystal object"""

        parameters = crystal.get_unit_cell().parameters()
        U = np.array(crystal.get_U()).reshape(3,3)
        A = np.array(crystal.get_A()).reshape(3,3)
        #O = np.array(uc.orthogonalization_matrix()).reshape(3,3)
        space_group = crystal.get_space_group().info().type().lookup_symbol()

        return Crystal(
            space_group=space_group,
            parameters=parameters,
            orientation_matrix=U,
            ub_matrix=A)

    def __str__(self):
        return str(self.__dict__)
