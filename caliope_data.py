from matplotlib import pyplot as plt # import libraries
import pandas as pd # import libraries
import netCDF4 # import libraries


data_dir = "./es-ftp.bsc.es:8021/"
fp=f'{data_dir}CALIOPE/no2/NO2/sconcno2_2023010100.nc' # your file name with the eventual path
nc = netCDF4.Dataset(fp) # reading the nc file and creating Dataset


print(nc.variables.keys()) # print the keys of the dataset
print(nc["sconcno2"]) # print the variable sconcno2
y = nc["sconcno2"][2]
x = nc["sconcno2"][3]





""" in this dataset each component will be 
in the form nt,nz,ny,nx i.e. all the variables will be flipped. """

""" imshow is a 2D plot function
according to what I have said before this will plot the second
iteration of the vertical slize with y = 0, one of the vertical
boundaries of your model. """
