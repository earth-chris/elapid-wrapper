# elapid

<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

Convenience functions and scripts for species distribution modeling (SDM)  in python. This includes a [MaxEnt][home-maxent] wrapper and tools for working with [GBIF-][home-gbif] and MaxEnt-format datasets.

The name was chosen as homage to the paper by H.A. Nix, *A Biogeographic Analysis of Australian Elapid Snakes* (1986), which is widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

## Dependencies

The source repository contains a conda environment with the packages required to run this software. If you want to install it on your own machine, there are three external dependencies you'll need to setup.

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

## Installation

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

## License

The Maxent software is bundled and served with this package, and is available under an MIT license. You can read more about the Maxent software [here][home-maxent].

This source code is likewise available under MIT license.

## Contributing

External contribution guidelines are not formally supported at this time--reach out to the [package developers](#contact) to facilitate contributions.

The `elapid` conda environment contains code formatting libraries, including `black`, `flake8`, `isort` and `pytest`. These tools are automatically applied to all code commits via the `pre-commit` library. To install `pre-commit`, run the following command:

```bash
pre-commit install
```

`pre-commit` behavior is defined in `.pre-commit-config.yaml`, specifying how to format the code, and cleaning up any formatting issues prior to accepting any commit. This should ensure that all pull requests are well-formatted.

## Contact

* Christopher Anderson is the primary developer [[web][home-cba]] [[email][email-cba]] [[github][github-cba]].


[email-cba]: mailto:cbanders@stanford.edu
[github-cba]: https://github.com/earth-chris
[github-jrs]: https://github.com/jeffreysmith-jrs
[home-cba]: https://earth-chris.github.io
[home-conda]: https://docs.conda.io/
[home-gbif]: https://gbif.org
[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
