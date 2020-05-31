from setuptools import setup

version = open("ccb/__version__.py").read().strip('"\n')

setup_args = {
    "name": "ccb",
    "version": version,
    "url": "https://github.com/earth-chris/ccb",
    "license": "MIT",
    "author": "Christopher Anderson",
    "author_email": "cbanders@stanford.edu",
    "description": "Species distribution modeling support tools",
    "keywords": ["maxent", "biogeography", "SDM", "species distribution modeling", "ecologyy", "conservation"],
    "packages": ["ccb"],
    "include_package_data": True,
    "platforms": "any",
    "scripts": ["bin/gbif-to-vector.py", "bin/vector-to-maxent.py"],
    "data_files": [("maxent", ["ccb/maxent/maxent.jar", "ccb/maxent/README.txt", "ccb/maxent/LICENSE.txt"])],
}

setup(**setup_args)
