#! /usr/bin/env python
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


############################
# set the command line argument parser
parser = argparse.ArgumentParser(description='Converts vector data to maxent format csv data')

# set input/output file arguments
parser.add_argument('-i', '--input', help='path to vector file', type=str, required=True)
parser.add_argument('-o', '--output', help='path to the output csv', type=str, required=True)

# set the attribute field to identify the species
parser.add_argument('-f', '--field', help='the attribute field to set as the species', type=str,
                    default='species')
                    
# allow user to set an attribute field as the x/y data
parser.add_argument('--xfield', help='use a certain attribute field as the Y data', default=None)
parser.add_argument('--yfield', help='use a certain attribute field as the Y data', default=None)

# now various flags and arguments
parser.add_argument('-e', '--epsg', help='the output projection', default=None)

# finall, parse the arguments
args = parser.parse_args()


############################
# set a couple of useful functions
def getXY(pt):
    return (pt.x, pt.y)
    

def read_input(path):
    try:
        data = gpd.read_file(path)
        geo = True
    except:
        try:
            data = pd.read_csv(path)
            geo = False
        except:
            print("[ ERROR! ]: Unable to read file: {}".format(path))
            print("[ ERROR! ]: Please ensure it is a vector file or a CSV")
            sys.exit(1)
            
    return data, geo
            

def check_fields(df, args, geo):
    # get the vector attributes
    attributes = df.columns.tolist()
    
    # check the species field is listed as a column in the dataframe
    if not args.field in attributes:
        print("[ ERROR! ]: Field set for species ID: {}".format(args.field))
        print("[ ERROR! ]: is not a vector attribute - select from the following:")
        print("[ ERROR! ]: [ {} ]".format(', '.join(attributes)))
        sys.exit(1)
        
    # if geometry is set as an attribute, check they are listed in the dataframe
    if args.xfield is not None:
        if not args.xfield in attributes:
            print("[ ERROR! ]: Field set for x data: {}".format(args.xfield))
            print("[ ERROR! ]: is not an attribute - select from the following:")
            print("[ ERROR! ]: [ {} ]".format(', '.join(attributes)))
            sys.exit(1)
    
    if args.yfield is not None:    
        if not args.yfield in attributes:
            print("[ ERROR! ]: Field set for y data: {}".format(args.yfield))
            print("[ ERROR! ]: is not a vector attribute - select from the following:")
            print("[ ERROR! ]: [ {} ]".format(', '.join(attributes)))
            sys.exit(1)
            
    # or, if the input data are not a vector, ensure that the x, y, and projection info are set
    if not geo:
        if not args.xfield in attributes:
            print("[ ERROR! ]: Field set for x data (using --xfield): {}".format(args.xfield))
            print("[ ERROR! ]: is either not set or is not an attribute - select from the following:")
            print("[ ERROR! ]: [ {} ]".format(', '.join(attributes)))
            sys.exit(1)
            
        if not args.yfield in attributes:
            print("[ ERROR! ]: Field set for y data (using --yfield): {}".format(args.yfield))
            print("[ ERROR! ]: is either not set or is not an attribute - select from the following:")
            print("[ ERROR! ]: [ {} ]".format(', '.join(attributes)))
            sys.exit(1)
            

############################
# read the input file
vector, geo = read_input(args.input)

# check that the fields passed as arguments are correct
check_fields(vector, args, geo)

# report starting
print("[ STATUS ]: Running vector-to-maxent")
print("[ STATUS ]: Converting input file : {}".format(args.input))
print("[ STATUS ]: To maxent-format file : {}".format(args.output))

# if the input data are vectors, pull the x/y from the geometry and add to new columns
if geo:
    # first, reproject if set
    if args.epsg is not None:
        vector.to_crs(epsg=args.epsg, inplace=True)
        
    # get the centroids to put the data in a useful format
    centroids = vector['geometry'].centroid
    
    # pull the x/y for each point 
    x, y = [list(pt) for pt in zip(*map(getXY, centroids))]
    
    # then slap 'em on as new columns
    vector['x'] = x
    vector['y'] = y
    
    # and set them as the x and yfields if not already passed by the user
    if args.xfield is None:
        args.xfield = 'x'
    if args.yfield is None:
        args.yfield = 'y'
        
# create a new dataframe with just the species/x/y data
labels_input = [args.field, args.xfield, args.yfield]
labels_output = {args.field: 'species', args.xfield: 'X', args.yfield: 'Y'}
maxent_df = vector[labels_input].rename(columns=labels_output)

# then write the output file
maxent_df.to_csv(args.output, index=False)

# celebrate widely!
print("[ STATUS ]: Finished vector-to-maxent!")
print("[ STATUS ]: See output file: {}".format(args.output))

