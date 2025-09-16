#!/bin/bash
# © Lennart Ramme, MPI-M, 2025

# This scripts creates global data out of the LUH mapped data downloaded from https://luh.umd.edu/data.shtml
# Downloaded files have to be located in the following directory of FRIDA-Clim for this script to work:
indir="../data/external/landuse/LUH_data/"
# with a separate directory for each scenario (extensions of scenarios go into their own directory as can be
# seen from below).

# The processing of LUH mapped data is done with the Climate Data Operators (CDO) version 1.8.2 
# (http://mpimet.mpg.de/cdo). On Linux you can install cdo with the following command:
# sudo apt update
# sudo apt install cdo

# This script should be run from this directory and will create data for each scenario in the same
# directory as above. As the processing can take quite some time, we recommend to reduce the following
# list to include only the scenarios of interest.
for sce in hist ssp119 ssp126 ssp245 ssp370 ssp434 ssp460 ssp534 ssp585 ssp126ext ssp534ext ssp585ext; do

	echo Preparing LUH data for $sce
	echo    ... landuse areas
	infile=indir${sce}'_LUH_data/multiple-states_input4MIPs_landState_*'
	outfile=indir${sce}'_LUH_data/'${sce}'_global_landuse_areas_from_LUH.nc'

	cdo fldsum -mul area.nc -selname,primf,primn,secdf,secdn,pastr,range,urban,c3ann,c4ann,c3per,c4per,c3nfx $infile $outfile


	echo    ... transitions
	infile=indir${sce}'_LUH_data/multiple-transitions_input4MIPs_landState_*'
	outfile=indir${sce}'_LUH_data/'${sce}'_global_transitions_from_LUH.nc'

	cdo fldsum -mul area.nc -delname,secnf_bioh,secyf_bioh,secmf_bioh,primn_bioh,primf_bioh,lat_bounds,lon_bounds,time_bnds $infile $outfile


	echo    ... wood harvest
	infile=indir${sce}'_LUH_data/multiple-transitions_input4MIPs_landState_*'
	outfile=indir${sce}'_LUH_data/'${sce}'_global_wood_harvest_biomass.nc'

	cdo fldsum -selname,secnf_bioh,secyf_bioh,secmf_bioh,primn_bioh,primf_bioh $infile $outfile

	echo done

done
