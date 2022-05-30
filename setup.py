from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("mdx2/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mdx2',
    version=__version__,
    description='MDX2: total scattering for macromolecular crystallography',
    long_description=readme,
    author='Steve P. Meisburger',
    author_email='spm82@cornell.edu',
    url='https://github.com/ando-lab/mdx2',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    python_requires=">=3",
    install_requires=[
        "numpy",
        "scipy",
        "dxtbx", # this does not work for some reason...
        "nexusformat", # or nexpy ?
    ],
    entry_points={
        'console_scripts': [
            'mdx2.version=mdx2.command_line.version:run',
            'mdx2.import_data=mdx2.command_line.import_data:run',
            'mdx2.import_geometry=mdx2.command_line.import_geometry:run',
            'mdx2.find_peaks=mdx2.command_line.find_peaks:run',
            'mdx2.mask_peaks=mdx2.command_line.mask_peaks:run',
            'mdx2.tree=mdx2.command_line.tree:run',
            'mdx2.bin_image_series=mdx2.command_line.bin_image_series:run',
            'mdx2.integrate=mdx2.command_line.integrate:run',
            'mdx2.correct=mdx2.command_line.correct:run',
            'mdx2.merge=mdx2.command_line.merge:run',
            'mdx2.map=mdx2.command_line.map:run',
            'mdx2.scale=mdx2.command_line.scale:run',
            ],
    },
    include_package_data=True,
)
