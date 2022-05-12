import numpy as np

from dxtbx.model.experiment_list import ExperimentList
from nexusformat.nexus import NXfield, NXdata

def iter_chunks(chunks,shape):
    # for now, assume chunks and shape are length 3
    assert len(chunks)==3
    assert len(shape)==3
    s1 = [slice(s,min(s+chunks[0],shape[0])) for s in range(0,shape[0],chunks[0])]
    s2 = [slice(s,min(s+chunks[1],shape[1])) for s in range(0,shape[1],chunks[1])]
    s3 = [slice(s,min(s+chunks[2],shape[2])) for s in range(0,shape[2],chunks[2])]
    for a1 in s1:
        for a2 in s2:
            for a3 in s3:
                yield (a1,a2,a3)

class Peaks:
    """search peaks in image stack"""
    # wrapper class for interfacing with nexus image stack
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def find_peaks_in_nexus(imagedata,threshold=None,verbose=True):
        chunks = imagedata.images.chunks
        shape = imagedata.images.shape
        phi_axis = np.array(imagedata.phi)
        ix_axis = np.array(imagedata.ix)
        iy_axis = np.array(imagedata.iy)
        pixels = [] # dump them into a list

        for ind,sl in enumerate(iter_chunks(chunks,shape)):
            imchunk = imagedata.images[sl]
            hotpixels = np.argwhere(imchunk > threshold)
            numhot = hotpixels.size
            if numhot:
                if verbose:
                    print(f'found {numhot} pixels (chunk {ind})')
                hotpixels[:,0] = phi_axis[sl[0]][hotpixels[:,0]]
                hotpixels[:,1] = iy_axis[sl[1]][hotpixels[:,1]]
                hotpixels[:,2] = ix_axis[sl[2]][hotpixels[:,2]]
                pixels.append(hotpixels)

        pixels = np.vstack(pixels) # convert to numpy array

        return Peaks(phi=pixels[:,0],iy=pixels[:,1],ix=pixels[:,2])


class ImageSet:
    """Wrapper for dxtbx imageset object"""
    def __init__(self,iset,maskval=None,dtype=np.int32,verbose=True):
        self._iset = iset # do I need to copy it?
        self.maskval = maskval # option to overwrite masked pixels with maskval
        self.verbose = verbose
        self.dtype = dtype

        det = iset.get_detector()[0] # <-- single panel assumed
        self.nx, self.ny = det.get_image_size()

        scan = self._iset.get_scan()
        self.nframes = scan.get_num_images()

    @property
    def shape(self):
        det = self._iset.get_detector()[0] # <-- single panel assumed
        nx, ny = det.get_image_size()
        scan = self._iset.get_scan()
        nframes = scan.get_num_images()
        return (nframes,ny,nx)

    def get_axes(self):
        scan = self._iset.get_scan()
        phi0, dphi = scan.get_oscillation()
        nframes,ny,nx = self.shape
        ix_axis = np.arange(0,nx)
        iy_axis = np.arange(0,ny)
        phi_axis = phi0 + dphi*(np.arange(0,nframes) +  0.5)
        return (phi_axis,iy_axis,ix_axis)

    def read_frame(self,ind):
        if self.verbose: print(f'mdx2.data.ImageSet: reading frame {ind}')
        im = self._iset.get_raw_data(ind)[0]
        msk = self._iset.get_mask(ind)[0]
        msk = ~msk
        if self.maskval is not None:
            im.set_selected(msk,self.maskval) # flex array magic
        image = im.as_numpy_array()
        mask = msk.as_numpy_array()
        return np.ma.masked_array(image,mask=mask)

    def read_stack(self,start,stop):
        shape = self.shape
        imstack = np.ma.empty(shape=(stop-start,shape[1],shape[2]),dtype=self.dtype)

        for ind in range(start,stop):
            imstack[ind-start,:,:] = self.read_frame(ind)

        return imstack

    def read_all(self,target_array,buffer=1):
        nframes = self.shape[0]

        if buffer == 1:
            for start in range(0,nframes):
                target_array[start,:,:] = self.read_frame(start)
        else:
            for start in range(0,nframes,buffer):
                stop = min(start + buffer,nframes)
                target_array[start:stop,:,:] = self.read_stack(start,stop)

    def to_nexus(self,compression='gzip',shuffle=True,chunks=(20,100,100)):
        images = NXfield(shape=self.shape,dtype=self.dtype,name='images')

        if compression is not None:
            if chunks is not None and chunks[0] > self.shape[0]:
                chunks = (self.shape[0],chunks[1],chunks[2])
            images.compression = compression
            images.compression_opts = 1 # use light compression by default (faster)
            images.shuffle = shuffle
            images.chunks = chunks

        phi_axis,iy_axis,ix_axis = self.get_axes()
        phi = NXfield(phi_axis,name='phi')
        ix = NXfield(ix_axis,name='ix')
        iy = NXfield(iy_axis,name='iy')
        data = NXdata(images,[phi,iy,ix])

        return data
