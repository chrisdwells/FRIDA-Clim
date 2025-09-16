# © Lennart Ramme, MPI-M, 2025

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import sys
import os

allOutNames = ['primf', 'primn', 'secdf', 'secdn', 'urban', 'crops', 'grass']
allInNames = ['secdf', 'secdn', 'urban', 'crops', 'grass']

allWoodHarvestAreaNames = ['primf_harv', 'primn_harv', 'secnf_harv', 'secyf_harv', 'secmf_harv']
allWoodHarvestBiomassNames = ['primf_bioh', 'primn_bioh', 'secnf_bioh', 'secyf_bioh', 'secmf_bioh']

allStockNames = ['primf', 'secdf', 'primn', 'secdn', 'urban', 'crops', 'grass']

FRIDAStocks = ['forest', 'grass', 'cropland', 'degraded']
iForest=0
iGrass=1
iCrop=2
iDegraded=3

FRIDAStocksPlus1 = ['forest', 'non-forest', 'grass', 'cropland', 'degraded']
iForestStock=0
inonForestStock=1
iGrassStock=2
iCropStock=3
iDegradedStock=4

outdir = '../data/external/landuse/'
indir = '../data/external/landuse/LUH_data/'
if not os.path.exists(outdir): os.makedirs(outdir)

for sce in ['hist', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534', 'ssp585',
            'ssp126ext', 'ssp534ext', 'ssp585ext']:
    print(sce)
    
    if not os.path.exists(outdir+sce): os.makedirs(outdir+'/'+sce)
    
    outpath = outdir+sce+'/'
    
    if sce == 'hist': time_shift = 850
    elif sce[-3:] == 'ext': time_shift=2100
    else: time_shift = 2015
    
    
    #### TRANSITIONS ###################
    ds=Dataset(indir+sce+'_LUH_data/'+sce+'_global_transitions_from_LUH.nc')
    time = ds.variables['time'][:]+time_shift
    
    allFlows = np.zeros((len(time), len(allOutNames), len(allInNames)))
    
    for transitionName in ds.variables.keys():
        if transitionName in ['lon', 'lat', 'time', 'secnf_harv', 'secyf_harv', 'secmf_harv', 'primf_harv', 'primn_harv']: continue
        if transitionName[:5] in ['c3per', 'c4per', 'c3ann', 'c4ann', 'c3nfx']: i = 5
        elif transitionName[:5] in ['pastr', 'range']: i = 6
        else: i = allOutNames.index(transitionName[:5])


        if transitionName[-5:] in ['c3per', 'c4per', 'c3ann', 'c4ann', 'c3nfx']: j = 3
        elif transitionName[-5:] in ['pastr', 'range']: j = 4
        else: j = allInNames.index(transitionName[-5:])

        allFlows[:,i,j] += ds.variables[transitionName][:,0,0]
        
    allFlowsFRIDA = np.zeros((len(time), len(FRIDAStocks), len(FRIDAStocks)))
    for i, OutName in enumerate(allOutNames):
        for j, InName in enumerate(allInNames):

            if OutName in ['primf', 'secdf']: iOut = iForest
            elif OutName in ['primn', 'secdn']: iOut = iGrass
            elif OutName in ['urban']: iOut = iDegraded
            elif OutName in ['crops']: iOut = iCrop
            elif OutName in ['grass']: iOut = iGrass
            else: sys.exit('Something wrong!')

            if InName in ['secdf']: iIn = iForest
            elif InName in ['secdn']: iIn = iGrass
            elif InName in ['urban']: iIn = iDegraded
            elif InName in ['crops']: iIn = iCrop
            elif InName in ['grass']: iIn = iGrass
            else: sys.exit('Something wrong!')

            allFlowsFRIDA[:,iOut,iIn] += allFlows[:,i,j]*1e-4
               
    
    ### From gross transitions to net transitions
    for i in range(len(FRIDAStocks)):
        allFlowsFRIDA[:,i,i] = 0.0
        
        for j in range(i+1,len(FRIDAStocks)):
            netFlow = allFlowsFRIDA[:,i,j] - allFlowsFRIDA[:,j,i]
            allFlowsFRIDA[:,i,j] = np.maximum(0, netFlow)
            allFlowsFRIDA[:,j,i] = np.maximum(0, -netFlow)
                
        
    ## writing data
    columns = [f"{s1}_to_{s2}" for s1 in FRIDAStocks for s2 in FRIDAStocks]
    reshaped_data = allFlowsFRIDA.reshape(allFlowsFRIDA.shape[0], -1)
    df = pd.DataFrame(reshaped_data, index=time, columns=columns)
    df.to_csv(outpath+sce+"_LUH_transitions_for_FRIDA.csv")
    
    print('  ..writing transitions done!')
    
    #### WOOD HARVEST
    ## AREA
    woodHarvestArea = np.zeros(len(time))
    for name in ds.variables.keys():
        if name in allWoodHarvestAreaNames: 
            woodHarvestArea += ds.variables[name][:,0,0]*1e-4

    ds.close()

    ## BIOMASS
    ds=Dataset(indir+sce+'_LUH_data/'+sce+'_global_wood_harvest_biomass.nc')

    woodHarvestBiomass = np.zeros(len(time))
    for name in ds.variables.keys():
        if name in allWoodHarvestBiomassNames: 
            woodHarvestBiomass += ds.variables[name][:,0,0]*1e-12

    ds.close()


    ### writing harvest data:
    df = pd.DataFrame(woodHarvestArea, index=time, columns=['wood_harvest_area'])
    df.to_csv(outpath+sce+"_LUH_wood_harvest_area_for_FRIDA.csv")

    df = pd.DataFrame(woodHarvestBiomass, index=time, columns=['wood_harvest_biomass'])
    df.to_csv(outpath+sce+"_LUH_wood_harvest_biomass_for_FRIDA.csv")

    print('  ..writing harvest done!')
    
    
    #### STOCK AREAS ###############################
    ds=Dataset(indir+sce+'_LUH_data/'+sce+'_global_landuse_areas_from_LUH.nc')
    if sce == 'hist': timeStock = np.append(time, np.asarray([2015]))
    elif sce[-3:] == 'ext': timeStock = np.append(time, np.asarray([2300]))
    else: timeStock = np.append(time, np.asarray([2100]))

    allStocks = np.zeros((len(timeStock), len(allStockNames)))
    for stockName in ds.variables.keys():
        if stockName in ['lon', 'lat', 'time']: continue
        if stockName in ['c3per', 'c4per', 'c3ann', 'c4ann', 'c3nfx']: i = 5
        elif stockName in ['pastr', 'range']: i = 6
        else: i = allStockNames.index(stockName)

        allStocks[:,i] += ds.variables[stockName][:,0,0]

    ds.close()

    allStocksFRIDA = np.zeros((len(timeStock), len(FRIDAStocksPlus1)))
    for i, StockName in enumerate(allStockNames):

        if StockName in ['primf', 'secdf']: iStock = iForestStock
        elif StockName in ['primn', 'secdn']: iStock = inonForestStock
        elif StockName in ['urban']: iStock = iDegradedStock
        elif StockName in ['crops']: iStock = iCropStock
        elif StockName in ['grass']: iStock = iGrassStock
        else: sys.exit('Something wrong!')

        allStocksFRIDA[:,iStock] += allStocks[:,i]*1e-4

    ### Writing stock:
    df = pd.DataFrame(allStocksFRIDA, index=timeStock, columns=FRIDAStocksPlus1)
    df.to_csv(outpath+sce+"_LUH_stocks_for_FRIDA.csv")
    
    print('  ..writing stocks done!')
