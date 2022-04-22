def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("mdx2")[0].version
    return version

__version__ = getVersionNumber()
