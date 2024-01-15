# Data processing tutorial: insulin

This tutorial demonstrates the essential steps of reciprocal space mapping with [mdx2](https://github.com/ando-lab/mdx2) on a small dataset from cubic insulin.

The tutorial is based on the [data processing workshop](https://github.com/ando-lab/erice-2022-data-reduction) at the [2022 Erice School on Diffuse Scattering](https://crystalerice.org/2022/programme_ds.php) and it accompanies a chapter on data processing in the forthcoming Methods in Enzymology volume "Crystallography of Protein Dynamics”.

To begin, download the jupyter notebooks individually or clone the _mdx2_ repository. Then, follow the instructions below to download the dataset and create a stand-alone Python environment with all of the software used in the tutorial.

The code is designed to run on a personal computer with ~20 Gb of free disk storage and at least 4 Gb of RAM. A unix-like operating system is assumed (Linux, OSX, or Windows Subsystem for Linux).

### Downloading the tutorial dataset

The dataset from insulin is available on Zenodo (<https://dx.doi.org/10.5281/zenodo.6536805>). First, download `insulin_2_1.tar` and extract the tar archive. The file will expand to a directory called `images` with subfolders `insulin_2_1`, `insulin_2_bkg`, and `metrology`. Place `images` in the same directory as the Jupyter notebooks.

### Setting up the Python environment

Install _miniconda3_ with _python_ version _3.10_. Installers and instructions are available on the web: <https://docs.conda.io/en/latest/miniconda.html>. To prevent miniconda from interfering with existing conda installations, it should be prevented from modifying the users' bash scripts or running "conda init" (`-b` flag). We also recommend choosing a non-default install location; `~/miniconda-mdx2` is assumed in the following examples, but can be modified if needed.

Example (MacOS x64):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-MacOSX-x86_64.sh -O ~/miniconda-installer.sh
bash ~/miniconda-installer.sh -b -p $HOME/miniconda-mdx2
```

Next, activate the mdx2 conda environment and install mdx2's dependencies

```bash
source ~/miniconda-mdx2/bin/activate
conda install -c conda-forge dxtbx nexusformat pandas numexpr
```

If the conda environment is de-activated (by typing `conda deactivate` or starting a new terminal session), it can be re-activated by typing `source ~/miniconda-mdx2/bin/activate`. For convenience, the following line can be added to the shell startup script:

Example (`~/.bash_profile`)
```bash
alias activate_mdx2="source ~/miniconda-mdx2/bin/activate"
```

Then, the mdx2 environment can be activated by typing `activate_mdx2`.

### Installing _mdx2_

First, activate the `mdx2` conda environment (see above). Then install mdx2 from the GitHub repository using pip

```bash
pip install git+https://github.com/ando-lab/mdx2.git@v0.3.1
```

The tag `@v0.3.1` at the end of the repository address specifies version 0.3.1 for consistency with this tutorial. If this tag is omitted, the most up-to-date version will be installed (see Version notes at <https://github.com/ando-lab/mdx2>).

The _mdx2_ tools should now be available at the command line. Check the version as follows:

```bash
mdx2.version
```

The version number `0.3.1` should be displayed.

### Installing _dials_, _nexpy_, and _jupyter lab_

First, activate the `mdx2` conda environment (see above). Then install the programs using conda

```bash
conda install -c conda-forge dials nexpy jupyterlab
```

With the _mdx2_ conda environment active, _nexpy_ can be launched from the command line by typing `nexpy`. Similarly, Jupyter Lab can be launched by typing `jupyter lab`. The dials command-line tools are also available. For instance `dials.version` prints the version information and install location.
