# Elapid

This library includes convenience functions and scripts to support species distribution modeling (SDM) efforts. This includes a python wrapper for [MaxEnt][home-maxent] and tools for translating [GBIF-][home-gbif] and MaxEnt-format datasets to traditional geospatial formats.

The name `elapid` was chosen as homage to Nix, H.A. (1986) *A Biogeographic Analysis of Australian Elapid Snakes*, the paper widely credited with defining the essential bioclimatic variables used in species distribution modeling. It's also a play on this software being a `python` wrapper for species distribution modeling tools.

## Dependencies

The source repository contains a conda environment with all the required packages for running this software.

If you want to install it on your own machine, there are three external dependencies you'll need to setup.

- `gdal`
- `openjdk`
- `GEOS`, a dependency for `shapely`.

`gdal` is notoriously difficult to install and work with, and is beyond the scope of this readme. `openjdk` installs `java`, which is used to run the Maxent software. This can be installed on linux via:

```bash
sudo apt-get install openjdk-11-jre
```

It can also be installed install via [conda][home-conda].

```bash
conda install openjdk
```

## Install

`pip install elapid` should do the trick.

### via conda

```bash
git clone https://github.com/earth-chris/elapid.git
cd elapid/
conda env update
```

Then you should have a conda environment you can actiave with `conda activate elapid`. You can then e.g. run the executable `vector-to-maxent -h`, or `import elapid` in python from this environment.

If you're interested in using the default `ipython` profile, you can set an environment variable to do this for you. From the base repository directory, run the following:

```bash
conda env config vars set IPYTHONDIR=$PWD/ipython
```

You'll have to run `conda deactivate` then `conda activate elapid` for the changes to take effect. After that you'll be able to run `ipython` with our default settings.

## Contributing

External contribution guidelines are not formally supported at this time--reach out to the [package developers](#contact) to facilitate contributions.

The `elapid` conda environment contains code formatting libraries, including `black`, `flake8`, `isort` and `pytest`. These tools are automatically applied to all code commits via the `pre-commit` library. To install `pre-commit`, run the following command:

```bash
pre-commit install
```

`pre-commit` behavior is defined in `.pre-commit-config.yaml`, specifying how to format the code, and cleaning up any formatting issues prior to accepting any commit. This should ensure that all pull requests are well-formatted.

## Contact

* Christopher Anderson is the primary developer [[email][email-cba]] [[github][github-cba]].


[email-cba]: mailto:cbanders@stanford.edu
[email-jrs]: mailto:jrsmith7@stanford.edu
[github-cba]: https://github.com/earth-chris
[github-jrs]: https://github.com/jeffreysmith-jrs
[home-conda]: https://docs.conda.io/
[home-gbif]: https://gbif.org
[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
