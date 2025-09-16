# ERA5 pre-processor for ERF simulations

This directory contains the python scripts to write the initial condition file, lateral forcing files and surface fluxes files  
for hurricane simulations in ERF from the ERA5 weather data.

1. Follow the steps here https://cds.climate.copernicus.eu/how-to-api to create a free account   
   to download ERA5 data

2. Give the year, month, day, time and the geographical area to download the data in a text file.  
For example, to download the data on August 26, 2020 at 00:00 (24 hour format)
```
year: 2020
month: 08
day: 26
time: 00:00
area: 50,-130,10,-50
```
Note: The geographical area is specified as latitude maximum, longitude minimum, latitude minimum, longitude maximum.

3. Run the script.    
`srun -n 32 python3 WriteICFromERA5Data.py <input_file> --do_forecast=true --forecast_time_hours=72 --interval_hours=3`    
where `input_file` is the file in Step 2 above. This uses 32 MPI ranks to download and process the weather data for    
hurricane Henri for a total of 72 hours with an interval of 3 hours.   

4. The output VTK files for visualization is written into  `Output/VTK/3D` for 3D data and `Output/VTK/Surface` for surface data. These  
can be visualized in VisIt or ParaView. 

5. The following directories are to be copied into the ERF run directory.  
`Output/3D`- The binary files (`*bin`) for lateral forcing. The first file in `Output/3D` can be used as the initial condition file.   
`Output/Surface` -  The binary files (`*.bin`) for surface fluxes.  
The domain extents to be used for the ERF run is written into `Output/domain_extents.txt`.  

Various example inputs for different hurricanes are provided in this folder. 


## VTK Visualization

To output a map of the US based on the Lambert conformal conic projection
centered over a specified area (corresponding to the input above):
```
write_US_map USMap_LambertProj.vtk --area 50 -130 10 -50
```

To output the NOAA forecast cone over the same region:
```
write_map_region HurricaneLaura_NOAA.csv NOAAForecastCone.vtk --area 50 -130 10 -50
```
