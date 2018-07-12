#! /usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# hack some encoding stuff because fml
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


############################
# setting the command line argument parser

# load the world boundaries shapefile for data subsetting
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[['geometry', 'name', 'continent']]

# get the unique continent and country names for user-specified subsetting
countries = world['name'].unique().tolist()
continents = world['continent'].unique().tolist()

# set the default taxa as a list to parse for the arguments
taxa = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# create the argument parser to feed input options
parser = argparse.ArgumentParser(description='Cleans and reprojects GBIF data')

# set input/output file arguments
parser.add_argument('-i', '--input', help='path to GBIF CSV file', type=str, required=True)
parser.add_argument('-o', '--output', help='path to the output vector', type=str, required=True)

# now various flags and arguments
parser.add_argument('-e', '--epsg', help='the output projection (in gdal-readable format)',
                    default='3410')
parser.add_argument('-t', '--taxonomic_resolution', help='the taxonomic group that must have data to be included',
                    default='genus', choices=taxa, metavar='')
parser.add_argument('-p', '--precision', help='the minimum allowable spatial precision (in meters)', default=None)
parser.add_argument('-y', '--min_year', help='the earliest year to grab data from', default=2000)

# allow subsetting by country or continent
parser.add_argument('--continent', help='subset the data by continent',
                    default=None, choices=continents, nargs='+', metavar='')
parser.add_argument('--country', help='subset the data by country', default=None,
                    choices=countries, nargs='+', metavar='')

# finall, parse the arguments
args = parser.parse_args()


############################
# set the defaults for which data to keep, which to toss

# first, set a bunch of defaults for which columns and records to keep
col_names = ['kingdom', 
            'phylum', 
            'class', 
            'order', 
            'family', 
            'genus',
            'species',
            'decimallatitude',
            'decimallongitude',
            'coordinateuncertaintyinmeters',
            'year',
            'basisofrecord',
            'datasetkey']

col_types = {'kingdom': str, 
            'phylum': str, 
            'class': str, 
            'order': str, 
            'family': str, 
            'genus': str,
            'species': str, 
            'decimallatitude': str,
            'decimallongitude': str,
            'coordinateuncertaintyinmeters': str,
            'year':  str,
            'basisofrecord': str,
            'datasetkey': str}

observation_type = ['HUMAN_OBSERVATION',
                   'PRESERVED_SPECIMEN', 
                   'SPECIMEN', 
                   'OBSERVATION', 
                   'LITERATURE',
                   'LIVING_SPECIMEN', 
                   'MATERIAL_SAMPLE', 
                   'MACHINE_OBSERVATION']
                   
valid_by = [args.taxonomic_resolution, 'decimallatitude','decimallongitude']
duplicate_by = ['species', 'genus', 'decimallatitude','decimallongitude']

# set the columns worth keeping
final_columns = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'year']


############################
# start reading the data and processing

print("[ STATUS ]: Reading input file: {}".format(args.input))

# read the data into memory and subset to just the relevant columns
full = pd.read_csv(args.input, header=0, sep='\t', usecols=col_names, dtype=col_types, error_bad_lines=False)

# then convert the numberic data types from strings
full['decimallatitude'] = pd.to_numeric(full['decimallatitude'], errors='coerce')
full['decimallongitude'] = pd.to_numeric(full['decimallongitude'], errors='coerce')
full['year'] = pd.to_numeric(full['year'], errors='coerce')
full['coordinateuncertaintyinmeters'] = pd.to_numeric(full['coordinateuncertaintyinmeters'], errors='coerce')

# start cleaning data
print("[ STATUS ]: Removing bad data points")

# keep only valid records
full = full.dropna(subset=valid_by) 

# drop records that are too inpercise
if args.precision is not None:
    full = full.loc[full['coordinateuncertaintyinmeters'] <= args.precision]

# drop points with 0.0 lat/lon
full = full.loc[full['decimallatitude']  != 0.0]
full = full.loc[full['decimallongitude'] != 0.0]

# drop it like its old
full = full.loc[full['year'] >= args.min_year]

# set observation types
full = full[full['basisofrecord'].isin(observation_type)]

# finally, remove duplicate points and clear the index
full = full[np.invert(full.duplicated(subset=duplicate_by))]
full.reset_index(drop = True, inplace = True)

# now, we'll convert it to a geopandas dataframe
geometry = [Point(xy) for xy in zip(full.decimallongitude, full.decimallatitude)] 
crs = {'init': 'epsg:4326'}
gdf = gpd.GeoDataFrame(full[final_columns], crs=crs, geometry=geometry)

# then intersect these data with country boundaries to ensure we're landlubbing
print("[ STATUS ]: Converting to GeoPandas format")
landlubbers = gpd.sjoin(gdf, world, op='within')

# subset by continent or country if set
if args.continent is not None:
    print("[ STATUS ]: Subsetting by continent(s): {}".format(', '.join(args.continent)))
    inds = landlubbers['continent'].isin(args.continent)
    landlubbers = landlubbers[inds]
    
if args.country is not None:
    print("[ STATUS ]: Subsetting by country(s): {}".format(', '.join(args.country)))
    inds = landlubbers['name'].isin(args.country)
    landlubbers = landlubbers[inds]

# then reproject the data if not using the standard wgs-84 lat/lon
if args.epsg is not 4326:
    print("[ STATUS ]: Reprojecting to EPSG: {}".format(args.epsg))
    #new_crs = {'init': args.t_srs.lower()}
    landlubbers.to_crs(epsg=args.epsg, inplace=True)
    #landlubbers.crs = new_crs

# then finally subset and rename the output columns
landlubbers.drop('index_right', 1, inplace=True)
landlubbers.rename(columns={'name': 'country'}, inplace=True)

# export the output file
landlubbers.to_file(args.output)
print("[ STATUS ]: Finished cleaning GBIF data!")
print("[ STATUS ]: See output file: {}".format(args.output))