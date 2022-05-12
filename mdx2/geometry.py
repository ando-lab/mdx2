import numpy as np

from nexusformat.nexus import NXsample, NXcollection, NXdata, NXfield, NXentry

class MillerIndex:
    """array of Miller indices"""
    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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

        ceil = lambda n,d: -1*(-n//d)
        x = np.linspace(0,nx,ceil(nx,sampling[0]))
        y = np.linspace(0,ny,ceil(ny,sampling[1]))
        #x = np.arange(nx)
        #y = np.arange(ny)
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
