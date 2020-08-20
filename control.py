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
VC['Stmin'] = 1
VC['Vtmin'] = 1
VC['Ttmin'] = 1
VC['an'] = '0.2(2)'
VC['aon'] = '0.05(05)'
VC['SVnn0'] = '1.25(50)'                        #prior for SV_nn[0][0]
VC['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
VC['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
VC['VVnn0'] = '1.0(5)'                         
VC['VVn'] = '0.0(5)'
VC['VV0'] = '0.0(1.0)'
VC['TVnn0'] = '0.2(2)'
VC['TVn'] = '0.0(2)'
VC['TV0'] = '0.0(5)'
VC['loosener'] = 0.05                          #Loosener on a_eff 
VC['oloosener'] = 1.0       #Loosener on oscillating ground  state a
VC['Mloosener'] = 0.05                        #Loosener on ground state 
VC['oMloosener'] = 0.1       #Loosener on oscillating ground state 
VC['a'] = 0.1715/(1.1119*0.1973 )
VC['BG-Tag'] = 'D_G5-G5_m{0}'
VC['BNG-Tag'] = 'D_G5T-G5T_m{0}'
VC['KG-Tag'] = 'K_G5-G5_tw{0}'
VC['KNG-Tag'] = 'K_G5-G5X_tw{0}'
VC['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
VC['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
VC['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'
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
C['tminBNG'] =3
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
C['VV0'] = '0.0(1.0)'
C['TVnn0'] = '0.2(2)'
C['TVn'] = '0.0(2)'
C['TV0'] = '0.0(5)'
C['loosener'] = 0.1                          #Loosener on a_eff 
C['oloosener'] = 1.0       #Loosener on oscillating ground  state a
C['Mloosener'] = 0.05                        #Loosener on ground state 
C['oMloosener'] = 0.2       #Loosener on oscillating ground state 
C['a'] = 0.1715/(1.3826*0.1973 )
C['BG-Tag'] = 'D_G5-G5_m{0}'
C['BNG-Tag'] = 'D_G5T-G5T_m{0}'
C['KG-Tag'] = 'K_G5-G5_tw{0}'
C['KNG-Tag'] = 'K_G5-G5X_tw{0}'
C['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
C['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
C['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'

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
F['TVnn0'] = '0.5(5)'
F['TVn'] = '0.0(5)'
F['TV0'] = '0.0(5)'
F['loosener'] = 0.05                          #Loosener on a_eff 
F['oloosener'] = 1.0       #Loosener on oscillating ground  state a
F['Mloosener'] = 0.05                        #Loosener on ground state 
F['oMloosener'] = 0.1       #Loosener on oscillating ground state 
F['a'] = 0.1715/(1.9006*0.1973 )
F['BG-Tag'] = 'B_G5-G5_m{0}'
F['BNG-Tag'] = 'B_G5T-G5T_m{0}'
F['KG-Tag'] = 'K_G5-G5_tw{0}'
F['KNG-Tag'] = 'K_G5-G5X_tw{0}'
F['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
F['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
F['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'

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
#SF['tmaxG'] = 67                             #72 is upper limit, ie includes all data 
#SF['tmaxNG'] = 71
#SF['tmaxD'] = 71                             #72 is upper limit, ie goes up to 71
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
SF['TV0'] = '0.0(5)'
SF['loosener'] = 0.2                         #Loosener on a_eff
SF['oloosener'] = 0.5                         #Loosener on osciallting ground state a
SF['Mloosener'] = 0.05                        #Loosener on ground state 
SF['oMloosener'] = 0.1                       #Loosener on oscillating ground state
SF['a'] = 0.1715/(2.896*0.1973)
SF['BG-Tag'] = 'B_G5-G5_m{0}'
SF['BNG-Tag'] = 'B_G5T-G5T_m{0}'
SF['KG-Tag'] = 'K_G5-G5_tw{0}'
SF['KNG-Tag'] = 'K_G5-G5X_tw{0}'
SF['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
SF['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
SF['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'

################ UF PARAMETERS #############################
UF = collections.OrderedDict()
UF['conf'] = 'UF'
UF['filename'] = "run-KBscalarvectortensor_261cfgs_neg"
UF['file_location'] = "../../ultrafine/extract/"
UF['masses'] = ['0.194','0.45','0.6','0.8']
UF['twists'] = ['0','0.706','1.529','2.235','4.705']
#SF['mtw'] = [[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1]]
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [33,40]
UF['tp'] = 192
UF['L'] = 64
UF['tmaxesBG'] = [96,96,88,80]
UF['tmaxesBNG'] = [96,85,86,75]
UF['tmaxesKG'] = [96,96,85,65,33]
UF['tmaxesKNG'] = [96,96,85,65,33]
UF['tminBG'] = 7
UF['tminBNG'] = 5
UF['tminKG'] = 7
UF['tminKNG'] = 6
#UF['tmaxG'] = 67                             #72 is upper limit, ie includes all data 
#UF['tmaxNG'] = 71
#UF['tmaxD'] = 71                             #72 is upper limit, ie goes up to 71
UF['Stmin'] = 2
UF['Vtmin'] = 2
UF['Ttmin'] = 2
UF['an'] = '0.08(0.10)'
UF['aon'] = '0.01(2)'
UF['SVnn0'] = '1.5(1.0)'                        #Prior for SV_nn[0][0]
UF['SVn'] = '0.0(5)'                        #Prior for SV_??[n][n]
UF['SV0'] = '0.0(1.0)'                          #Prior for SV_no[0][0] etc
UF['VVnn0'] = '1.0(5)'
UF['VVn'] = '0.0(5)'
UF['VV0'] = '0.0(2.0)'
UF['TVnn0'] = '0.2(1)'
UF['TVn'] = '0.0(1)'
UF['TV0'] = '0.0(5)'
UF['loosener'] = 0.5                         #Loosener on a_eff
UF['oloosener'] = 0.5                         #Loosener on oscillating ground state a
UF['Mloosener'] = 0.02                        #Loosener on ground state 
UF['oMloosener'] = 0.2                       #Loosener on oscillating ground state
UF['a'] = 0.1715/(3.892*0.1973)
UF['BG-Tag'] = 'B_G5-G5_m{0}'
UF['BNG-Tag'] = 'B_G5T-G5T_m{0}'
UF['KG-Tag'] = 'K_G5-G5_tw{0}'
UF['KNG-Tag'] = 'K_G5-G5X_tw{0}'
UF['threePtTagS'] = 'scalar_T{0}_m{1}_m{2}_m{3}_tw{4}'
UF['threePtTagV'] = 'vector_T{0}_m{1}_m{2}_m{3}_tw{4}'
UF['threePtTagT'] = 'tensor_T{0}_m{1}_m{2}_m{3}_tw{4}'

################ USER INPUTS ################################
#############################################################

Fit = SF                                               # Choose to fit F, SF , UF
FitMasses = [0,1,2,3]                                 # Choose which masses to fit
FitTwists = [0,1,2,3,4]                               # Choose which twists to fit
FitTs = [0,1,2]#,3]
FitCorrs = [['BG','BNG'],['KG','KNG'],[['S'],['V'],['T']]]  #Choose which corrs to fit ['G','NG','D','S','V'], set up in chain [[link1],[link2]], [[parrallell1],[parallell2]] ...]
Chained = False   # If False puts all correlators above in one fit no matter how they are organised
Marginalised = False # set to eg 6. Two points will be run up to 6 then marginalised to Nmin<N<Nmax
SaveFit = True
smallsave = True #saves only the ground state non-oscillating and 3pts
svdnoise = False
priornoise = False
ResultPlots = False         # Tell what to plot against, "Q", "N","Log(GBF)", False
SvdFactor = 0.001                       # Multiplies saved SVD
PriorLoosener = 1.0                   # Multiplies all prior error by loosener
Nmax = 5                               # Number of exp to fit for 2pts in chained, marginalised fit
Nmin = 5                              #Number to start on
FitToGBF = False                  # If false fits to Nmax
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
        modelsA,modelsB = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks)
        data = make_data('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']))
    else: 
        models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks)
        data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models)#process for speed
        
############################ Do chained fit #########################################################
    if Chained and Marginalised == False:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_chained_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_chained_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None)
############################ Do chained marginalised fit ##############################################
    elif Chained and Marginalised != False:
        if FitToGBF:    #fits with priorn(Marginalised) and nterm = n,n
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,Marginalised,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_chained_marginalised_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,GBF1,Marginalised)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,Marginalised,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_chained_marginalised_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None,Marginalised)
######################### Do unchained fit ############################################################
    else:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                GBF2 = do_unchained_fit(data,prior,N,models,svdcut,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating)
                do_unchained_fit(data,prior,N,models,svdcut,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,None)        


#####################################################################################################
            
    return()
main()
