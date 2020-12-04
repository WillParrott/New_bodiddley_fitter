import collections
import sys
import h5py
import gvar as gv
import numpy as np
import corrfitter as cf
#import corrbayes
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from functions import *
from plotting import *
plt.rc("font",**{"size":18})
import os.path
import pickle
import datetime
##B B^0 differes about 100MeV kaon by 400MeV Include this in priors#

################ VC PARAMETERS  #############################
VCp = collections.OrderedDict()
VCp['conf'] = 'VCp'
VCp['filename'] = "VCp_998cfg"
VCp['file_location'] = "./Bipashadata/data/"
VCp['masses'] = ['0.8605']
VCp['twists'] = ['0','2.013','3.05','3.969'] 
VCp['m_l'] = '0.00235'
VCp['m_s'] = '0.0678'
VCp['Ts'] = [9,12,15,18]
VCp['tp'] = 48 
VCp['L'] = 32
VCp['tmaxesBG'] = [24]
VCp['tmaxesBNG'] = [24]
VCp['tmaxesKG'] = [24,24,24,16]  
VCp['tminBG'] = 3
VCp['tminBNG'] = 3
VCp['tminKG'] = 3 #parenst to nexp7 daughters to nexp3
VCp['Stmin'] = 2 
VCp['Vtmin'] = 2
VCp['an'] = '0.15(20)'    #prior for all an and heavy osciallting an
VCp['aon'] = '0.05(5)'
VCp['SVnn0'] = '1.25(50)'
VCp['SVn'] = '0.0(5)'
VCp['SV0'] = '0.0(1.0)'
VCp['VVnn0'] = '1.0(5)'
VCp['VVn'] = '0.0(5)'
VCp['VV0'] = '0.0(1.0)'
VCp['loosener'] = 0.2                          #Loosener on a_eff
VCp['oloosener'] = 1.0       #Loosener on oscillating ground  state a
VCp['Mloosener'] = 0.05                        #Loosener on ground state
VCp['oMloosener'] = 0.1       #Loosener on oscillating ground state
VCp['Vloosener'] = 0.2 
VCp['a'] = 0.1715/(1.1367*0.1973 )
VCp['BG-Tag'] = 'D_G5-G5_m{0}'
VCp['BNG-Tag'] = 'D_G5T-G5T_m{0}'
VCp['KG-Tag'] = 'K_G5-G5_tw{0}'
VCp['threePtTagS'] = 'scalar_T{0}_tw{4}'
VCp['threePtTagV'] = 'vector_T{0}_tw{4}'
VCp['svd'] = 1.0 
VCp['binsize'] = 16      
################ Cp PARAMETERS  #############################
Cp = collections.OrderedDict()
Cp['conf'] = 'Cp'
Cp['filename'] = "Cp-binned-985_cfg"
Cp['file_location'] = "./Bipashadata/data/"
Cp['masses'] = ['0.643']
Cp['twists'] = ['0','2.405','3.641','4.735']
Cp['m_l'] = '0.00184'
Cp['m_s'] = '0.0527'
Cp['Ts'] = [12,15,18,21]
Cp['tp'] = 64
Cp['L'] = 48
Cp['tmaxesBG'] = [32]
Cp['tmaxesBNG'] = [32]                                                                                      
Cp['tmaxesKG'] = [32,32,31,23]            
Cp['tminBG'] = 4
Cp['tminBNG'] = 5
Cp['tminKG'] = 5 
Cp['Stmin'] = 1
Cp['Vtmin'] = 1
Cp['an'] = '0.15(10)'
Cp['aon'] = '0.05(5)'
Cp['SVnn0'] = '1.25(50)'                        #prior for SV_nn[0][0]
Cp['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
Cp['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
Cp['VVnn0'] = '1.0(5)'
Cp['VVn'] = '0.0(5)'
Cp['VV0'] = '0.0(1.0)'
Cp['loosener'] = 0.1                          #Loosener on a_eff
Cp['oloosener'] = 1.0       #Loosener on oscillating ground  state a
Cp['Mloosener'] = 0.02                        #Loosener on ground state
Cp['oMloosener'] = 0.05       #Loosener on oscillating ground state
Cp['Vloosener'] = 0.3
Cp['a'] = 0.1715/(1.4149*0.1973 )
Cp['BG-Tag'] = 'D_G5-G5_m{0}'
Cp['BNG-Tag'] = 'D_G5T-G5T_m{0}'
Cp['KG-Tag'] = 'K_G5-G5_tw{0}'
Cp['threePtTagS'] = 'scalar_tw{4}_T{0}'
Cp['threePtTagV'] = 'vector_T{0}_tw{4}'             
Cp['svd'] = 1.0  
Cp['binsize'] = 1
################ VC PARAMETERS  #############################
VC = collections.OrderedDict()
VC['conf'] = 'VC'
VC['filename'] = "test-KDscalarvectortensor_1020cfgs_neg"
VC['file_location'] = "../../../DK/vcoarse5/extract/"
VC['masses'] = ['0.888']
VC['twists'] = ['0','0.3665','1.097','1.828']
VC['m_l'] = '0.013'
VC['m_s'] = '0.0705'
VC['Ts'] = [9,12,15,18]
VC['tp'] = 48
VC['L'] = 16
VC['tmaxesBG'] = [24]
VC['tmaxesBNG'] = [24]
VC['tmaxesKG'] = [24,24,24,20]            #48 is upper limit, ie goes up to 47
VC['tmaxesKNG'] = [24,24,24,18] 
VC['tminBG'] = 2
VC['tminBNG'] = 2
VC['tminKG'] = 2 #parenst to nexp7 daughters to nexp3
VC['tminKNG'] = 2                           # 3 for 5 twists, 5 for first 4 
VC['Stmin'] = 2
VC['Vtmin'] = 2
VC['Ttmin'] = 2
VC['an'] = '0.2(2)'
VC['aon'] = '0.05(5)'
VC['SVnn0'] = '1.25(50)'                        #prior for SV_nn[0][0]
VC['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
VC['SV0'] = '0.0(1.5)'                          #Prior for SV_no[0][0] etc
VC['VVnn0'] = '1.0(5)'                         
VC['VVn'] = '0.0(5)'
VC['VV0'] = '0.0(1.5)'
VC['TVnn0'] = '0.15(10)'
VC['TVn'] = '0.0(1)'
VC['TV0'] = '0.0(3)'
VC['loosener'] = 0.05                          #Loosener on a_eff 
VC['oloosener'] = 1.0       #Loosener on oscillating ground  state a
VC['Mloosener'] = 0.2                        #Loosener on ground state 
VC['oMloosener'] = 0.2       #Loosener on oscillating ground state 
VC['Vloosener'] = 0.2  #Loosener on Vnn[0][0]
VC['a'] = 0.1715/(1.1119*0.1973 )
VC['BG-Tag'] = 'D_G5-G5_m{0}'
VC['BNG-Tag'] = 'D_G5T-G5T_m{0}'
VC['KG-Tag'] = 'K_G5-G5_tw{0}'
VC['KNG-Tag'] = 'K_G5-G5X_tw{0}'
VC['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
VC['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
VC['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
VC['svd'] = 1.0
VC['binsize'] = 1
################ C PARAMETERS  #############################
C = collections.OrderedDict()
C['conf'] = 'C'
C['filename'] = "threemass-KDscalarvectortensor_1053cfgs_neg"
C['file_location'] = "../../../DK/coarse-5/extract/"
C['masses'] = ['0.664','0.8','0.9']
C['twists'] = ['0','0.441','1.323','2.205','2.646']
C['m_l'] = '0.0102'
C['m_s'] = '0.0545'
C['Ts'] = [12,15,18,21]
C['tp'] = 64
C['L'] = 24
C['tmaxesBG'] = [32,32,32]
C['tmaxesBNG'] = [32,32,32]
C['tmaxesKG'] = [32,32,32,27,21]            #48 is upper limit, ie goes up to 47
C['tmaxesKNG'] = [32,32,32,23,21] 
C['tminBG'] = 3
C['tminBNG'] = 3
C['tminKG'] = 3 #parenst to nexp7 daughters to nexp3
C['tminKNG'] = 3                           # 3 for 5 twists, 5 for first 4 
C['Stmin'] = 1
C['Vtmin'] = 1
C['Ttmin'] = 1
C['an'] = '0.2(2)'
C['aon'] = '0.03(3)'
C['SVnn0'] = '1.25(50)'                        #prior for SV_nn[0][0]
C['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
C['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
C['VVnn0'] = '1.0(5)'                         
C['VVn'] = '0.0(5)'
C['VV0'] = '0.0(1.5)'
C['TVnn0'] = '0.15(10)'
C['TVn'] = '0.0(1)'
C['TV0'] = '0.0(3)'
C['loosener'] = 0.1                          #Loosener on a_eff 
C['oloosener'] = 1.0       #Loosener on oscillating ground  state a
C['Mloosener'] = 0.05                        #Loosener on ground state 
C['oMloosener'] = 0.2       #Loosener on oscillating ground state
C['Vloosener'] = 0.1  #Loosener on Vnn[0][0] 
C['a'] = 0.1715/(1.3826*0.1973 )
C['BG-Tag'] = 'D_G5-G5_m{0}'
C['BNG-Tag'] = 'D_G5T-G5T_m{0}'
C['KG-Tag'] = 'K_G5-G5_tw{0}'
C['KNG-Tag'] = 'K_G5-G5X_tw{0}'
C['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
C['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
C['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
C['svd'] = 1.0
C['binsize'] = 1
################ F PARAMETERS  #############################
F = collections.OrderedDict()
F['conf'] = 'F'
F['filename'] = "test-KBscalarvectortensor_499cfgs_neg"
F['file_location'] = "../../fine5/extract/"
F['masses'] = ['0.449','0.566','0.683','0.8']
F['twists'] = ['0','0.4281','1.282','2.141','2.570']
#F['mtw'] = [[1,1,1,1,0,0],[1,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,1,1,1]]
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['tmaxesBG'] = [48,48,48,48]
F['tmaxesBNG'] = [48,48,48,48]
F['tmaxesKG'] = [48,48,48,30,25]            #48 is upper limit, ie goes up to 47
F['tmaxesKNG'] = [48,48,48,30,25] 
F['tminBG'] = 3
F['tminBNG'] = 3
F['tminKG'] = 4 #parenst to nexp7 daughters to nexp3
F['tminKNG'] = 3                           # 3 for 5 twists, 5 for first 4 
F['Stmin'] = 2
F['Vtmin'] = 2
F['Ttmin'] = 2
F['an'] = '0.10(10)'
F['aon'] = '0.05(05)'
F['SVnn0'] = '1.50(75)'                        #prior for SV_nn[0][0]
F['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
F['SV0'] = '0.0(1.5)'                          #Prior for SV_no[0][0] etc
F['VVnn0'] = '1.0(5)'                         
F['VVn'] = '0.0(5)'
F['VV0'] = '0.0(1.5)'
F['TVnn0'] = '0.2(1)'
F['TVn'] = '0.0(1)'
F['TV0'] = '0.0(3)'
F['loosener'] = 0.05                          #Loosener on a_eff 
F['oloosener'] = 1.0       #Loosener on oscillating ground  state a
F['Mloosener'] = 0.05                        #Loosener on ground state 
F['oMloosener'] = 0.1       #Loosener on oscillating ground state 
F['Vloosener'] = 0.2 #Loosener on Vnn[0][0]
F['a'] = 0.1715/(1.9006*0.1973 )
F['BG-Tag'] = 'B_G5-G5_m{0}'
F['BNG-Tag'] = 'B_G5T-G5T_m{0}'
F['KG-Tag'] = 'K_G5-G5_tw{0}'
F['KNG-Tag'] = 'K_G5-G5X_tw{0}'
F['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
F['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
F['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
F['svd'] = 1.0
F['binsize'] = 1
################ SF PARAMETERS #############################
SF = collections.OrderedDict()
SF['conf'] = 'SF'
SF['filename'] = "nohimem-KBscalarvectortensor_415cfgs_negscalarvector"
SF['file_location'] = "../../superfine/extract/"
SF['masses'] = ['0.274','0.45','0.6','0.8']
SF['twists'] = ['0','1.261','2.108','2.946','3.624']
#SF['mtw'] = [[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1]]
SF['m_l'] = '0.0048'
SF['m_s'] = '0.0234'
SF['Ts'] = [20,25,30]
SF['tp'] = 144
SF['L'] = 48
SF['tmaxesBG'] = [72,72,72,72]
SF['tmaxesBNG'] = [72,72,72,72]
SF['tmaxesKG'] = [72,72,55,35,32]
SF['tmaxesKNG'] = [72,72,55,35,32]
SF['tminBG'] = 5 #
SF['tminBNG'] = 2 # 
SF['tminKG'] = 5
SF['tminKNG'] = 3
SF['Stmin'] = 2
SF['Vtmin'] = 2
SF['Ttmin'] = 2
SF['an'] = '0.05(5)'
SF['aon'] = '0.02(2)'
SF['SVnn0'] = '1.5(1.0)'                        #Prior for SV_nn[0][0]
SF['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
SF['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
SF['VVnn0'] = '1.0(5)'
SF['VVn'] = '0.0(5)'
SF['VV0'] = '0.0(2.0)'
SF['TVnn0'] = '0.2(1)'
SF['TVn'] = '0.0(1)'
SF['TV0'] = '0.0(3)'
SF['loosener'] = 0.2                         #Loosener on a_eff
SF['oloosener'] = 0.5                         #Loosener on osciallting ground state a
SF['Mloosener'] = 0.05                        #Loosener on ground state 
SF['oMloosener'] = 0.1                       #Loosener on oscillating ground state
SF['Vloosener'] = 0.3  #Loosener on Vnn[0][0]
SF['a'] = 0.1715/(2.896*0.1973)
SF['BG-Tag'] = 'B_G5-G5_m{0}'
SF['BNG-Tag'] = 'B_G5T-G5T_m{0}'
SF['KG-Tag'] = 'K_G5-G5_tw{0}'
SF['KNG-Tag'] = 'K_G5-G5X_tw{0}'
SF['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
SF['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
SF['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
SF['svd'] = 1.0
SF['binsize'] = 1
################ UF PARAMETERS #############################
UF = collections.OrderedDict()
UF['conf'] = 'UF'
UF['filename'] = "unbinned-run-KBscalarvectortensor_375cfgs_neg"
UF['file_location'] = "../../ultrafine/extract/"
UF['masses'] = ['0.194','0.45','0.6','0.8']
UF['twists'] = ['0','0.706','1.529','2.235','4.705']
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [24,33,40]
UF['tp'] = 192
UF['L'] = 64
UF['tmaxesBG'] = [96,96,96,93]
UF['tmaxesBNG'] = [96,96,96,93]
UF['tmaxesKG'] = [96,96,92,68,39]
UF['tmaxesKNG'] = [96,96,92,66,39]
UF['tminBG'] = 6
UF['tminBNG'] = 5
UF['tminKG'] = 7  
UF['tminKNG'] = 5
UF['Stmin'] = 2#4#2
UF['Vtmin'] = 2#6#2
UF['Ttmin'] = 2#5#2
UF['an'] = '0.08(0.10)'
UF['aon'] = '0.01(2)'
UF['SVnn0'] = '1.25(1.0)'                        #Prior for SV_nn[0][0]
UF['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
UF['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
UF['VVnn0'] = '0.75(75)'
UF['VVn'] = '0.0(5)'
UF['VV0'] = '0.0(2.0)'
UF['TVnn0'] = '0.2(1)'
UF['TVn'] = '0.0(1)'
UF['TV0'] = '0.0(2)'
UF['loosener'] = 0.5                         #Loosener on a_eff
UF['oloosener'] = 0.5                         #Loosener on oscillating ground state a
UF['Mloosener'] = 0.02                        #Loosener on ground state 
UF['oMloosener'] = 0.2                       #Loosener on oscillating ground state
UF['Vloosener'] = 0.5  #Loosener on Vnn[0][0]
UF['a'] = 0.1715/(3.892*0.1973)
UF['BG-Tag'] = 'B_G5-G5_m{0}'
UF['BNG-Tag'] = 'B_G5T-G5T_m{0}'
UF['KG-Tag'] = 'K_G5-G5_tw{0}'
UF['KNG-Tag'] = 'K_G5-G5X_tw{0}'
UF['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
UF['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
UF['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
UF['svd'] = 1.0
UF['binsize'] = 1
################ Fp PARAMETERS  #############################
Fp = collections.OrderedDict()
Fp['conf'] = 'Fp'
#Fp['filename'] = "fphys-KBscalarvectortensor_119cfgs_neg"
Fp['filename'] = "all_streams_620cfgs"
Fp['file_location'] = "../../fine_physical/extract/"
Fp['masses'] = ['0.433','0.683','0.8']
Fp['twists'] = ['0','0.8563','2.998','5.140']
#Fp['mtw'] = [[1,1,1,1,0,0],[1,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,1,1,1]]
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['Ts'] = [14,17,20]
Fp['tp'] = 96
Fp['L'] = 64
Fp['tmaxesBG'] = [48,48,48]
Fp['tmaxesBNG'] = [48,48,48]
Fp['tmaxesKG'] = [48,48,44,25]            #48 is upper limit, ie goes up to 47
Fp['tmaxesKNG'] = [48,48,44,24] 
Fp['tminBG'] = 4
Fp['tminBNG'] = 4 
Fp['tminKG'] = 6
Fp['tminKNG'] = 4                           
Fp['Stmin'] = 2
Fp['Vtmin'] = 2
Fp['Ttmin'] = 2
Fp['an'] = '0.10(10)'
Fp['aon'] = '0.05(05)'
Fp['SVnn0'] = '1.50(75)'                        #prior for SV_nn[0][0]
Fp['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
Fp['SV0'] = '0.0(1.5)'                          #Prior for SV_no[0][0] etc
Fp['VVnn0'] = '1.0(5)'                         
Fp['VVn'] = '0.0(5)'
Fp['VV0'] = '0.0(1.5)'
Fp['TVnn0'] = '0.2(1)'
Fp['TVn'] = '0.0(1)'
Fp['TV0'] = '0.0(3)'
Fp['loosener'] = 0.1 #05                          #Loosener on a_eff 
Fp['oloosener'] = 1.0       #Loosener on oscillating ground  state a
Fp['Mloosener'] = 0.1 #05                        #Loosener on ground state 
Fp['oMloosener'] = 0.2 #1       #Loosener on oscillating ground state 
Fp['Vloosener'] = 0.3  #Loosener on Vnn[0][0]
Fp['a'] = 0.1715/(1.9518*0.1973 )
Fp['BG-Tag'] = 'B_G5-G5_m{0}'
Fp['BNG-Tag'] = 'B_G5T-G5T_m{0}'
Fp['KG-Tag'] = 'K_G5-G5_tw{0}'
Fp['KNG-Tag'] = 'K_G5-G5X_tw{0}'
Fp['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
Fp['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
Fp['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
Fp['svd'] = 1.0
Fp['binsize'] = 1 
################ USER INPUTS ################################
#############################################################

Fit = Fp                                            # Choose to fit F, SF , UF
FitMasses = [0,1,2,3]                                 # Choose which masses to fit
FitTwists = [0,1,2,3,4]                               # Choose which twists to fit
FitTs = [0,1,2,3]
FitCorrs = np.array([['BG','BNG'],['KG','KNG'],[['S'],['V'],['T']]],dtype=object)  #Choose which corrs to fit ['G','NG','D','S','V'], set up in chain [[link1],[link2]], [[parrallell1],[parallell2]] ...]
SaveFit = False
noise = True #'all noise
SepMass = False
SvdFactor = 1.0*Fit['svd']                       # Multiplies saved SVD
PriorLoosener = 1.0                   # Multiplies all prior error by loosener
Nmax = 4                              # Number of exp to fit for 2pts in chained, marginalised fit
Nmin = 4                              #Number to start on
FitToGBF = False                  # If false fits to Nmax
Chained = False   # If False puts all correlators above in one fit no matter how they are organised
Marginalised = False # set to eg 6. Two points will be run up to 6 then marginalised to Nmin<N<Nmax

####
ResultPlots = False         # Tell what to plot against, "Q", "N","Log(GBF)", False
smallsave = True #saves only the ground state non-oscillating and 3pts
##############################################################
setup = ['KG-S-BG','KG-V-BNG','KNG-T-BNG']
notwist0 = ['T'] #list any fits which do not use tw-0
non_oscillating = [] #any daughters which do no osciallate (only tw 0 is affected)
##############################################################
##############################################################

def main():
    # get which correlators we want
    daughters,currents,parents = read_setup(setup)
    # plots corrs if first time with this data file 
    plots(Fit,daughters,parents,currents)
    allcorrs,links,parrlinks = elements_in_FitCorrs(FitCorrs)
    # remove masses and twists we don't want to fit
    make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents)
    # make models
    if Chained:
        modelsA,modelsB = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass)
        data = make_data('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),Fit['binsize'])
    elif SepMass:
        massmodels = collections.OrderedDict()
        data = make_data('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),Fit['binsize'])
        masslist = copy.deepcopy(Fit['masses'])
        for mass in masslist:
            massmodels[mass] = {}
            Fit['masses'] = [mass]
            massmodels[mass]['models'],massmodels[mass]['svdcut'] = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass) 
        Fit['masses'] = copy.deepcopy(masslist)
    else: 
        models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass)
        data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models,Fit['binsize'])#process for speed
        
############################ Do chained fit #########################################################
    if Chained and Marginalised == False:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_chained_fit(data,prior,N,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_chained_fit(data,prior,N,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None)
############################ Do chained marginalised fit ##############################################
    elif Chained and Marginalised != False:
        if FitToGBF:    #fits with priorn(Marginalised) and nterm = n,n
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,Marginalised,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_chained_marginalised_fit(data,prior,N,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,GBF1,Marginalised)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,Marginalised,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_chained_marginalised_fit(data,prior,N,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None,Marginalised)

################################ do SepMass fit ######################################################

    elif SepMass:
        for N in range(Nmin,Nmax+1):
            masslist = copy.deepcopy(Fit['masses'])
            result = gv.BufferDict()
            priors = collections.OrderedDict()
            for mass in masslist:
                Fit['masses'] = [mass]
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                priors[mass] = prior
                result[mass] = do_sep_mass_fit(data,prior,N,massmodels[mass]['models'],massmodels[mass]['svdcut'],Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,None)
            Fit['masses'] = copy.deepcopy(masslist)
            combine_sep_mass_fits(result,Fit,priors,allcorrs,N,SvdFactor,PriorLoosener,currents,SaveFit,smallsave)
######################### Do unchained fit ############################################################
    
    else:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_unchained_fit(data,prior,N,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_unchained_fit(data,prior,N,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,None)        


#####################################################################################################
            
    return()
main()
