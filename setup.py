from setuptools import setup

version = open("elapid/__version__.py").read().strip('"\n')
long_description = open("README.md", "r", encoding="utf-8").read()

setup_args = {
    "name": "elapid",
    "version": version,
    "url": "https://github.com/earth-chris/elapid",
    "license": "MIT",
    "author": "Christopher Anderson",
    "author_email": "cbanders@stanford.edu",
    "description": "Species distribution modeling support tools",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "keywords": [
        "maxent",
        "biogeography",
        "SDM",
        "species distribution modeling",
        "ecologyy",
        "conservation",
    ],
    "packages": ["elapid"],
    "include_package_data": True,
    "platforms": "any",
    "scripts": ["bin/gbif-to-vector.py", "bin/vector-to-maxent.py"],
    "data_files": [
        (
            "maxent",
            [
                "elapid/maxent/maxent.jar",
                "elapid/maxent/README.txt",
                "elapid/maxent/LICENSE.txt",
            ],
        )
    ],
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
}

setup(**setup_args)
