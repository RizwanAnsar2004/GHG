from netCDF4 import Dataset
import numpy as np

# Paths to your NetCDF files
annual_file_path = '/Users/admin/Downloads/Gridded Methane Data/GEPA_Annual.nc'
daily_file_path = '/Users/admin/Downloads/Gridded Methane Data/GEPA_Daily.nc'
monthly_file_path = '/Users/admin/Downloads/Gridded Methane Data/GEPA_Monthly.nc'

# Function to read and print the content of a NetCDF file
def read_netcdf(file_path):
    # Open the NetCDF file
    dataset = Dataset(file_path, mode='r')
    
    # Print file information
    print(f"\nNetCDF File: {file_path}")
    print("File Name:", dataset.filepath())
    print("Dimensions:", dataset.dimensions)
    print("Variables:", dataset.variables)
    print("Global Attributes:", dataset.ncattrs())
    
    # Print details about each variable
    print("\nVariables Details:")
    for var_name in dataset.variables:
        var = dataset.variables[var_name]
        print(f"Variable Name: {var_name}")
        print(f"  Dimensions: {var.dimensions}")
        print(f"  Shape: {var.shape}")
        print(f"  Data Type: {var.dtype}")
        print(f"  Attributes: {var.ncattrs()}")

    # Close the dataset
    dataset.close()

# Read each of the NetCDF files
read_netcdf(annual_file_path)
read_netcdf(daily_file_path)
read_netcdf(monthly_file_path)
