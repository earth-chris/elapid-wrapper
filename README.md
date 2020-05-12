# CCB

This library includes convenience functions and scripts to support species distribution modeling (SDM) efforts. This includes a python wrapper for [MaxEnt][home-maxent] and tools for translating [GBIF-][home-gbif] and MaxEnt-format datasets to traditional geospatial formats. 

## Pre-requisites

The easiest way is to install via [conda][home-conda]. Make sure you have `conda` installed (either via minconda (recommended) or anaconda).

## Install

### via conda

```bash
git clone https://github.com/stanford-ccb/ccb.git
cd ccb/
conda env create --file=environment.yml
```

Once you've created the environment, activate it and install `ccb`.

```bash
conda activate ccb
pip install -r requirements.txt
python setup.py install
```

Then you should have a conda environment you can actiave with `conda activate ccb`. You can then e.g. run the executable `vector-to-maxent -h`, or `import ccb` in python from this environment.

If you're interested in using the ccb default `ipython` profile, you can set an environment variable to do this for you. From the base `ccb` directory, run the following:

```bash
conda activate ccb
conda env config vars set IPYTHONDIR=$PWD/ipython
```

You'll have to run `conda deactivate` then `conda activate ccb` for the changes to take effect. After that you'll be able to run `ipython` with our default settings.

### non-conda install

Clone the repository and create a system `ccb` install.

```bash
git clone https://github.com/stanford-ccb/ccb.git
pip install -r requirements.txt
python setup.py install
```

## Contributing

External contribution guidelines are not formally supported at this time--reach out to the [package developers](#contact) to facilitate contributions.

A separate `conda` development environment was created to facilitate contributing well-formatted code to `ccb`. Create this environment with the following command.

```bash
conda env create --file=environment-dev.yml
```

This environment contains additional code formatting libraries, including `black`, `flake8`, `isort` and `pytest`. These tools are automatically applied to all code commits via the `pre-commit` library. To install `pre-commit`, run the following command:

```bash
pre-commit install
```

`pre-commit` behavior is defined in `.pre-commit-config.yaml`, specifying how to format the code, and cleaning up any formatting issues prior to accepting any commit. This should ensure that all pull requests are well-formatted.

## Contact

* Christopher Anderson is the primary developer [[email][email-cba]] [[github][github-cba]]
* Jeff Smith also has keys to the car [[email][email-jrs]] [[github][github-jrs]]


[email-cba]: mailto:cbanders@stanford.edu
[email-jrs]: mailto:jrsmith7@stanford.edu
[github-cba]: https://github.com/earth-chris
[github-jrs]: https://github.com/jeffreysmith-jrs
[home-conda]: https://docs.conda.io/
[home-gbif]: https://gbif.org
[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
