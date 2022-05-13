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
            'mdx2.hello=mdx2.command_line.hello:run',
            'mdx2.import_images=mdx2.command_line.import_images:run',
            'mdx2.import_crystal=mdx2.command_line.import_crystal:run',
            'mdx2.import_corrections=mdx2.command_line.import_corrections:run',
            'mdx2.import_miller_index=mdx2.command_line.import_miller_index:run',
            'mdx2.find_peaks=mdx2.command_line.find_peaks:run',
            'mdx2.analyze_peaks=mdx2.command_line.analyze_peaks:run',
            'mdx2.peak_mask=mdx2.command_line.peak_mask:run',
            ],
    },
    include_package_data=True,
)
