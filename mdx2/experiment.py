import numpy as np
import h5py
from dxtbx.model.experiment_list import ExperimentList

def read_image_stack(iset,dtype=np.int32,maskval=-1):
    nims = iset.size()
    nx, ny = iset.get_detector()[0].get_image_size() # <-- single panel assumed
    imstack = np.empty((nims,ny,nx),dtype=dtype)
    for ind in range(iset.size()):
        im = iset.get_raw_data(ind)[0]
        msk = iset.get_mask(ind)[0]
        im.set_selected(~msk,maskval) # flex array magic
        imstack[ind,:,:] = im.as_numpy_array().astype(dtype)
    return imstack

def add_nxdata_to_h5(f,axes,labels,signal,**kwargs):
    """Initialize a nxdata entry for storing N-D array"""
    # lay out the file structure (try to follow nexus spec.)
    nxentry = f.create_group("/entry")
    nxentry.attrs["NX_class"] = "NXentry"
    nxdata = nxentry.create_group("data")
    nxdata.attrs["NX_class"] = "NXdata"

    # write the axes
    nxdata.attrs["axes"] = labels
    for ind, (axis,label) in enumerate(zip(axes,labels)):
        nxdata.attrs[label + "_indices"] = ind
        nxdata.create_dataset(label, data=axis)

    # write the signal
    signal_shape = [np.size(ax) for ax in axes]
    nxdata.attrs["signal"] = signal
    dataset = nxdata.create_dataset(signal, shape=signal_shape, **kwargs)
    return dataset

def buffered_read_write(iset,dataset,nbuffer,**kwargs):
    scan = iset.get_scan()
    nframes = scan.get_num_images()
    for start in range(0,nframes,nbuffer):
        stop = min(start + nbuffer,nframes)
        print('loading images from',start,'to',stop-1)
        imstack = read_image_stack(iset[start:stop],**kwargs)
        print('writing to file')
        dataset[start:stop,:,:] = imstack
    return

def import_images(exptin,h5out,chunksize=(20,100,100),dtype=np.int32):

    elist = ExperimentList.from_file(exptin)
    iset = elist.imagesets()[0]

    # chunksize: frame -by- iy -by- ix
    if chunksize is not None:
        nbuffer = chunksize[0]
        h5options = {
            "chunks":chunksize,
            "compression":"gzip",
            "compression_opts":1,
            "shuffle":True,
            }
    else:
        nbuffer = 1
        h5options = {}

    detector = iset.get_detector()
    nx,ny = detector[0].get_image_size() # assume single-panel

    scan = iset.get_scan()
    nframes = scan.get_num_images()
    phi0, dphi = scan.get_oscillation()

    ix = np.arange(0,nx)
    iy = np.arange(0,ny)
    phi = phi0 + dphi*(np.arange(0,nframes) +  0.5)

    with h5py.File(h5out, "w") as f:
        dataset = add_nxdata_to_h5(f,(phi,iy,ix),("phi","iy","ix"),"counts",dtype=dtype,**h5options)
        buffered_read_write(iset,dataset,nbuffer,dtype=dtype)

    return
