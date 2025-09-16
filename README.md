# FRIDA
WorldTrans FRIDA-Clim climate model

This repository contains the standalone FRIDA-Clim climate model from the wider FRIDA IAM model (Feedback-based knowledge Repository for Integrated Assessments). 

The frida_clim_processing/scripts/ folder contains scripts for generating the inputs to the calibration model (land use, emissions, wood biomass, NPP in 1750). These shouldn't need running as their outputs are small and stored in frida_clim_processing/data/processed_for_frida. They have to be run in numerical order.

The calibration/scripts folder holds the scripts for running the calibration. Currently two different calibrations are implemented, varying in their emissions inputs: frida_iam, which reflects the inputs generated for the FRIDA IAM, and rcmip, which follows the RCMIP inputs. The models and data associated with these are stored in their respective folders in calibration/, and the calibration of choice is defined in the .env file, along with the ensemble sizes. The model(s) need running between scripts, with instructions in each script. Again these need to be run in numerical order.

The model is being developed using <a href="https://www.iseesystems.com/store/products/stella-architect.aspx">Stella Architect</a>.

You can run and use the model locally without purchasing Stella Architect using the <a href="https://www.iseesystems.com/softwares/player/iseeplayer.aspx">isee Player</a>

