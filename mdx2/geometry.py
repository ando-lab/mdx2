import numpy as np

from nexusformat.nexus import NXsample, NXcollection

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
