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
    ],
    entry_points={
        'console_scripts': [
            'mdx2.hello=mdx2.command_line.hello:run',
            'mdx2.import_images=mdx2.command_line.import_images:run',
            ],
    },
    include_package_data=True,
)
