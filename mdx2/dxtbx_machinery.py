
import numpy as np
import re
from multiprocessing import Process, JoinableQueue
from joblib import Parallel, delayed

from dxtbx.model.experiment_list import ExperimentList

class Experiment:
    """Wrapper for dxtbx Experiment object"""
    def __init__(self,expt):
        self._crystal = expt.crystal
        self._goniometer = expt.goniometer
        self._panel = expt.detector[0] # single-panel detectors assumed
        self._scan = expt.scan
        self._beam = expt.beam

    @staticmethod
    def from_file(exptfile):
        elist = ExperimentList.from_file(exptfile)
        expt = elist[0] # single experiment in file assumed
        return Experiment(expt)

    @property
    def image_size(self):
        nx,ny = self._panel.get_image_size()
        return nx,ny

    @property
    def oscillation_range(self):
        phimin, phimax = self._scan.get_oscillation_range()
        return phimin, phimax

    @property
    def oscillation(self):
        phi0, dphi = self._scan.get_oscillation()
        return phi0, dphi

    @property
    def num_images(self):
        nimgs = self._scan.get_num_images()
        return nimgs

    @property
    def exposure_times(self):
        return np.array(tuple(self._scan.get_exposure_times()))

    @property
    def space_group_number(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        sgt = sgi.type()
        return sgt.number()

    @property
    def space_group_symbol(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        sgt = sgi.type()
        return sgt.lookup_symbol()

    @property
    def crystal_system(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        return sg.crystal_system()

    @property
    def reciprocal_space_asu(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        asu = sgi.reciprocal_space_asu()
        asu_str = asu.reference_as_string()
        # make substitutions for numpy compatibility
        asu_str = re.sub(r'([hkl0\<\>\=]+)',r'(\1)',asu_str)
        asu_str = re.sub('and',r'&',asu_str)
        asu_str = re.sub('or',r'|',asu_str)
        # apply a change-of-basis operation if needed
        # see: https://github.com/cctbx/cctbx_project/blob/master/cctbx/sgtbx/reciprocal_space_asu.h
        if not asu.is_reference():
            op = asu.cb_op().as_hkl()
            hklmap = {k:f'({v})' for k,v in zip(['h','k','l'],op.split(','))}
            asu_str = re.sub('(h|k|l)',lambda x: hklmap[x.group()], asu_str)
        return asu_str

    @property
    def space_group_operators(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        ops = sg.all_ops()
        sg_ops = np.stack([np.array(op.as_double_array()).reshape(3,4) for op in ops],axis=0)
        return sg_ops

    @property
    def laue_group_number(self):
        symm = self._crystal.get_crystal_symmetry()
        lg = symm.space_group().build_derived_laue_group()
        return lg.info().type().number()

    @property
    def laue_group_symbol(self):
        symm = self._crystal.get_crystal_symmetry()
        return symm.space_group().laue_group_type()

    @property
    def laue_group_operators(self):
        symm = self._crystal.get_crystal_symmetry()
        lg = symm.space_group().build_derived_laue_group()
        ops = lg.all_ops()
        return np.stack([np.array(op.r().as_double()).reshape(3,3) for op in ops],axis=0)

    @property
    def ub_matrix(self):
        return np.array(self._crystal.get_A()).reshape(3,3)

    @property
    def space_group(self):
        sg = self._crystal.get_space_group()
        return sg.info().type().lookup_symbol()

    @property
    def orientation_matrix(self):
        return np.array(self._crystal.get_U()).reshape(3,3)

    @property
    def unit_cell(self):
        uc = self._crystal.get_unit_cell()
        return uc.parameters()

    @property
    def scan_axes(self):
        return self.calc_scan_axes()

    @property
    def reflection_conditions(self):
        sg = self._crystal.get_space_group()
        centering_type = sg.conventional_centring_type_symbol()
        return lookup_reflection_conditions(centering_type)

    def calc_scan_axes(self,centered=True,samples=None,spacing=None):
        nx,ny = self.image_size
        nimgs = self.num_images
        phi0, dphi = self.oscillation

        if samples is not None:
            # use samples as provided
            pass
        elif spacing is not None:
            # compute samples from spacing
            ceil = lambda n,d: -1*int(-n//d)
            samp_x = ceil(nx,spacing[2])
            samp_y = ceil(ny,spacing[1])
            samp_phi = ceil(dphi*nimgs,spacing[0])
            if centered:
                samples = (samp_phi,samp_y,samp_x)
            else:
                samples = (1 + samp_phi,1 + samp_y,1 + samp_x)
        else:
            if centered:
                samples = (nimgs,ny,nx)
            else:
                samples = (nimgs+1,ny+1,nx+1)

        if centered:
            phi = phi0 + dphi*np.linspace(0.5,nimgs-0.5,samples[0])
            iy = np.linspace(0,ny-1,samples[1])
            ix = np.linspace(0,nx-1,samples[2])
        else: # edges
            phi = phi0 + dphi*np.linspace(0,nimgs,samples[0])
            iy = np.linspace(-0.5,ny-0.5,samples[1])
            ix = np.linspace(-0.5,nx-0.5,samples[2])

        return phi,iy,ix

    def calc_xyz_on_grid(self,iy,ix):
        x,y,z = index_grid_to_lab(self._panel,ix,iy)
        return x,y,z # shape? -- TODO: split into x, y, z

    def calc_s_on_grid(self,*args):
        x,y,z = self.calc_xyz_on_grid(*args)
        xyz = np.stack((x,y,z))
        d = np.sqrt(np.sum(xyz*xyz,axis=0))
        wlen = self._beam.get_wavelength()
        s1 = xyz/d/wlen
        s0 = np.array(self._beam.get_s0())
        s = s1 - s0[:,np.newaxis,np.newaxis]
        sx,sy,sz = s[0,:,:], s[1,:,:], s[2,:,:]
        return sx, sy, sz # shape? -- TODO: split into sx, sy, sz

    def calc_hkl_on_grid(self,phi,*args):
        sx,sy,sz = self.calc_s_on_grid(*args)
        s = np.stack((sx,sy,sz))

        R = calc_rotation_matrix_at_phi(self._goniometer,phi)
        B = calc_B_matrix_at_phi(self._crystal,self._scan,phi)
        U = calc_U_matrix_at_phi(self._crystal,self._scan,phi)

        # array of orthogonalization matrices
        UB = R @ U @ B

        # array of fractionalization matrix
        invUB = np.linalg.inv(UB)

        hkl = np.tensordot(invUB,s,axes=1)
        h,k,l = hkl[:,0,:,:], hkl[:,1,:,:], hkl[:,2,:,:]

        return h, k, l

    def calc_corrections_on_grid(self,*args):
        x,y,z = self.calc_xyz_on_grid(*args)
        xyz = np.dstack((x,y,z))

        # some helpful geometry that I'll reuse
        d = np.sqrt(np.sum(xyz*xyz,axis=2))
        dnorm = self._panel.get_normal()
        cosw = np.dot(xyz,dnorm)/d

        solid_angle = calc_solid_angle(self._panel,cosw,d)
        efficiency = calc_efficiency(self._panel,cosw)
        attenuation = calc_attenuation(self._beam,d)
        polarization = calc_polarization(self._beam,xyz,d)
        inverse_lorentz = calc_inverse_lorentz(self._beam,self._goniometer,xyz,d)
        d3s = calc_d3s(self._beam,self._scan,solid_angle,inverse_lorentz)

        return {
            'solid_angle':solid_angle,
            'efficiency':efficiency,
            'attenuation':attenuation,
            'polarization':polarization,
            'inverse_lorentz':inverse_lorentz,
            'd3s':d3s,
            }

class ImageSet:
    """Wrapper for dxtbx imageset object"""
    def __init__(self,iset):
        self._iset = iset

    @staticmethod
    def from_file(exptfile):
        elist = ExperimentList.from_file(exptfile)
        iset = elist.imagesets()[0] # read only the first image set if multiple
        return ImageSet(iset)

    @property
    def num_frames(self):
        scan = self._iset.get_scan()
        return scan.get_num_images()

    @property
    def frame_shape(self):
        det = self._iset.get_detector()[0] # <-- single panel assumed
        nx, ny = det.get_image_size()
        return (ny,nx)


    def read_frame(self,ind,verbose=True,maskval=-1):
        # for now, apply mask by default and return just the image as an ndarray
        if verbose: print(f'{self.__class__.__name__}: reading frame {ind}')
        im = self._iset.get_raw_data(ind)[0]
        msk = self._iset.get_mask(ind)[0]
        msk = ~msk
        if maskval is not None:
            im.set_selected(msk,maskval) # flex array magic
            image = im.as_numpy_array()
        else:
            image = np.ma.masked_array(im.as_numpy_array(),mask=msk.as_numpy_array())
        return image

    def read_stack(self,start,stop):
        stack = []
        # read the rest of the frames
        for ind in range(start,stop):
            stack.append(self.read_frame(ind))

        return np.stack(stack)

    def read_all(self,target_array,buffer=1):
        nframes = self.num_frames

        if buffer == 1:
            for start in range(0,nframes):
                target_array[start,:,:] = self.read_frame(start)
        else:
            for start in range(0,nframes,buffer):
                stop = min(start + buffer,nframes)
                target_array[start:stop,:,:] = self.read_stack(start,stop)

    def read_all_parallel(self,target_array,buffer=1,nproc=1,verbose=True):

        def nxswriter(q):
            while True:
                val = q.get()
                if val is None: break
                start,stop,arr = val # unpack the tuple
                if verbose: print(f'{self.__class__.__name__}: writing block to nexus file: {start}-{stop}')
                target_array[start:stop,:,:] = arr
                q.task_done()
            # Finish up
            q.task_done()

        q = JoinableQueue(maxsize=2)
        writeprocess = Process(target=nxswriter, args=(q,))
        writeprocess.start()

        nframes = self.num_frames
        with Parallel(n_jobs=nproc) as parallel:
            for start in range(0,nframes,buffer):
                stop = min(start + buffer,nframes)
                res = parallel(delayed(self.read_frame)(ind) for ind in range(start,stop))
                q.put((start,stop,np.stack(res)))
        q.put(None) # Poison pill
        q.join()
        writeprocess.join()

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

    #imin, imax = scan.get_array_range()
    imin = 0
    imax = scan.get_num_images()
    phi0, dphi = scan.get_oscillation()
    b = []
    f = []
    for ind, phi in enumerate(phi_vals_deg):
        #scan_index = scan.get_array_index_from_angle(phi)
        scan_index = (phi - phi0)/dphi
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
    x, y, z = xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]
    return x, y, z

# corrections
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
    dphi_radians = dphi*np.pi/180
    return Linv*solidAngle*dphi_radians/wlen**3

def lookup_reflection_conditions(centering_type):
    # reflection conditions due to centering
    # see: https://dictionary.iucr.org/Reflection_conditions

    if centering_type == 'P':
        return 'True'
    elif centering_type == 'F':
        # h + k, h + l and k + l = 2n or: h, k, l all odd or all even
        return '(((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)) | ((h % 2 == k % 2) & (h % 2 == l % 2))'
    elif centering_type == 'I':
        # h + k + l = 2n
        return '(h + k + l) % 2 == 0'
    elif centering_type == 'A':
        # k + l = 2n
        return '(k + l) % 2 == 0'
    elif centering_type == 'B':
        # l + h = 2n
        return '(l + h) % 2 == 0'
    elif centering_type == 'C':
        # h + k = 2n
        return '(h + k) % 2 == 0'
    elif centering_type == 'D':
        # h + k + l = 3n
        return '(h + k + l) % 3 == 0'
    elif centering_type =='H':
        # h − k = 3n
        return '(h - k) % 3 == 0'
    elif centering_type == 'R':
        # − h + k + l = 3n (standard obverse setting)
        return '(- h + k + l) % 3 == 0'
    else:
        # centering type not recognized!
        return None
