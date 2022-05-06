"""
A minimal NeXus file format for storing array data
"""

import numpy as np

import h5py

import mdx2 # version

mdx2_version = mdx2.__version__

def read_data(h5name,dsetname,**kwargs):
    pass

def write_data(h5name,dsetname,data,**kwargs):
    pass


def print_tree(h5name,root='/'):

    def print_info(v, d=0):
        for n in v:
            if isinstance(v[n], h5py.Dataset):
                print('   '*d + n + ': ' + str(v[n].shape))
                isleafnode = True
            else:
                print('   '*d + n)
                isleafnode = False
            for a in v[n].attrs:
                print('   '*(d + 1) + '@'+a,'=',v[n].attrs[a])
            if not isleafnode:
                print_info(v[n], d + 1)

    with h5py.File(h5name) as f:
        print(h5name + ':')
        print_info(f[root])


def new_file(h5name):
    with h5py.File(h5name,'w') as f:
        nxentry = f.create_group('/entry')
        nxentry.attrs["NX_class"]='NXentry'
        nxprogram = nxentry.create_dataset("program_name", data="mdx2")
        nxprogram.attrs["version"]=mdx2_version


def add_dataset(h5name,dsetname,axes,labels):
    fullpath = "/entry/" + dsetname
    with h5py.File(h5name,'a') as f:
        #assert fullpath not in f, "dataset already exists in file"
        nxdset = f.create_group(fullpath)
        nxdset.attrs["NX_class"]="NXdata"
        nxdset.attrs["axes"] = labels
        for ind, (label, axis) in enumerate(zip(labels,axes)):
            #nxdset.attrs[label + "_indices"] = ind
            nxdset.create_dataset(label,data=axis)


def add_data(h5name,dsetname,**kwargs):
    with h5py.File(h5name,'a') as f:
        nxdata = f["/entry/" + dsetname]
        #assert 'data' not in nxdata, "data already exists in dataset"
        # get the shape
        shape = tuple(nxdata[ax].size for ax in nxdata.attrs["axes"])
        nxdata.create_dataset("data", shape=shape, **kwargs)
