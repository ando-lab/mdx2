import numpy as np

from dxtbx.model.experiment_list import ExperimentList
from nexusformat.nexus import NXfield

class ImageStack:
    """read images and write image stack to nexus data file"""
    #
    # class to wrap NXdata object and to import raw image data using dxtbx machinery.
    #
    # source: (NXdata, which may be in memory on on disk)
    # data: (ImageSet object, with read method)

    def __init__(self,data):
        # data should be an NXdata object
        self.data = data

    @staticmethod
    def init_from_source(source,compression='gzip',shuffle=True,chunks=(20,100,100)):
        shape=source.shape
        dtype=source.dtype
        if compression is None:
            data = NXfield(shape=shape,dtype=dtype)
        else:
            if chunks is not None and chunks[0] > shape[0]:
                chunks = (shape[0],chunks[1],chunks[2])
            data = NXfield(shape=shape,dtype=dtype,compression=compression,shuffle=shuffle,chunks=chunks)
        return ImageStack(data)

    def read_from_source(self,source,buffer=None):
        if buffer is None and self.data.compression is not None:
            buffer = self.data.compression[0]

        nframes = self.data.shape[0]

        if buffer is None or buffer == 1:
            for start in range(0,nframes):
                self.data[start,:,:] = source.read_frame(start)
        else:
            for start in range(0,nframes,buffer):
                stop = min(start + buffer,nframes)
                self.data[start:stop,:,:] = source.read_stack(start,stop)


class ImageSet:
    """Wrapper for dxtbx imageset object"""
    def __init__(self,iset):
        self._iset = iset # do I need to copy it?
        self.maskval = -1
        self.dtype = np.int32

        det = iset.get_detector()[0] # <-- single panel assumed
        self.nx, self.ny = det.get_image_size()

        scan = self._iset.get_scan()
        self.nframes = scan.get_num_images()

        self.verbose=True

    @property
    def shape(self):
        return (self.nframes,self.ny,self.nx)

    def read_frame(self,ind):
        if self.verbose: print(f'mdx2.data.ImageSet.read_frame: reading frame {ind}.')
        im = self._iset.get_raw_data(ind)[0]
        msk = self._iset.get_mask(ind)[0]
        im.set_selected(~msk,self.maskval) # flex array magic
        return im.as_numpy_array()

    def read_stack(self,start,stop):
        shape = self.shape
        imstack = np.empty(shape=(stop-start,shape[1],shape[2]),dtype=self.dtype)

        if self.verbose: print(f'mdx2.data.ImageSet.read_stack: reading frames {start} to {stop-1}.')

        for ind in range(start,stop):
            imstack[ind-start,:,:] = self.read_frame(ind)
