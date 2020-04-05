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

################ F PARAMETERS  #############################
F = collections.OrderedDict()
F['conf'] = 'F'
F['filename'] = "KBscalarvectortensor_398cfgs_negFalse"
F['file_location'] = "../../fine_test/extract/"
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
F['tmaxesKG'] = [48,48,48,48,48]            #48 is upper limit, ie goes up to 47
F['tmaxesKNG'] = [48,48,48,48,48] 
F['tminBG'] = 3
F['tminBNG'] = 3
F['tminKG'] = 3
F['tminKNG'] = 3                            # 3 for 5 twists, 5 for first 4 
F['Stmin'] = 2
F['Vtmin'] = 2
F['Ttmin'] = 2
F['an'] = '0.1(1)'
F['SVnn0'] = '0.5(5)'                        #prior for SV_nn[0][0]
F['SVn'] = '0.01(25)'                        #Prior for SV_??[n][n]
F['SV0'] = '0.3(5)'                          #Prior for SV_no[0][0] etc
F['VVnn0'] = '0.5(5)'                         
F['VVn'] = '0.01(25)'
F['VV0'] = '0.3(5)'
F['TVnn0'] = '0.5(5)'
F['TVn'] = '0.01(25)'
F['TV0'] = '0.3(5)'
F['loosener'] = 0.10                          #Loosener on a_eff
F['Mloosener'] = 0.05                        #Loosener on ground state 
F['oMloosener'] = 0.2                       #Loosener on oscillating ground state
F['a'] = 0.1715/(1.9006*0.1973)
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
SF['filename'] = "nohimem-KBscalarvectortensor_158cfgs_negscalarvector"
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
SF['tmaxesKG'] = [65,65,50,40,30]
SF['tmaxesKNG'] = [65,65,50,40,30]
SF['tminBG'] = 6
SF['tminBNG'] = 6
SF['tminKG'] = 4
SF['tminKNG'] = 4
#SF['tmaxG'] = 67                             #72 is upper limit, ie includes all data 
#SF['tmaxNG'] = 71
#SF['tmaxD'] = 71                             #72 is upper limit, ie goes up to 71
SF['Stmin'] = 2
SF['Vtmin'] = 2
SF['Ttmin'] = 2
SF['an'] = '0.1(0.020)'
SF['SVnn0'] = '0.5(5)'                        #Prior for SV_nn[0][0]
SF['SVn'] = '0.01(25)'                        #Prior for SV_??[n][n]
SF['SV0'] = '0.01(50)'                          #Prior for SV_no[0][0] etc
SF['VVnn0'] = '0.5(5)'
SF['VVn'] = '0.01(25)'
SF['VV0'] = '0.01(50)'
SF['TVnn0'] = '0.5(5)'
SF['TVn'] = '0.01(25)'
SF['TV0'] = '0.01(50)'
SF['loosener'] = 0.5                         #Loosener on a_eff
SF['Mloosener'] = 0.05                        #Loosener on ground state 
SF['oMloosener'] = 0.10                       #Loosener on oscillating ground state
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
UF['filename'] = "run-KBscalarvectortensor_3cfgs_neg"
UF['file_location'] = "../../ultrafine/extract/"
UF['masses'] = ['0.194','0.45','0.6','0.8']
UF['twists'] = ['0','0.706','1.529','2.235','4.705']
#SF['mtw'] = [[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1]]
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [33,40]
UF['tp'] = 192
UF['L'] = 64
UF['tmaxesBG'] = [96,96,96,96]
UF['tmaxesBNG'] = [96,96,96,96]
UF['tmaxesKG'] = [96,96,96,96,96]
UF['tmaxesKNG'] = [96,96,96,96,96]
UF['tminBG'] = 8
UF['tminBNG'] = 7
UF['tminKG'] = 9
UF['tminKNG'] = 9
#UF['tmaxG'] = 67                             #72 is upper limit, ie includes all data 
#UF['tmaxNG'] = 71
#UF['tmaxD'] = 71                             #72 is upper limit, ie goes up to 71
UF['Stmin'] = 2
UF['Vtmin'] = 2
UF['Ttmin'] = 2
UF['an'] = '0.1(0.10)'
UF['SVnn0'] = '0.5(5)'                        #Prior for SV_nn[0][0]
UF['SVn'] = '0.01(25)'                        #Prior for SV_??[n][n]
UF['SV0'] = '0.01(50)'                          #Prior for SV_no[0][0] etc
UF['VVnn0'] = '0.5(5)'
UF['VVn'] = '0.01(25)'
UF['VV0'] = '0.01(50)'
UF['TVnn0'] = '0.5(5)'
UF['TVn'] = '0.01(25)'
UF['TV0'] = '0.01(50)'
UF['loosener'] = 0.7                         #Loosener on a_eff
UF['Mloosener'] = 0.005                        #Loosener on ground state 
UF['oMloosener'] = 0.03                       #Loosener on oscillating ground state
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
FitTs = [0,1,2]
FitCorrs = [['BG','BNG'],['KG','KNG'],[['S'],['V'],['T']]]  #Choose which corrs to fit ['G','NG','D','S','V'], set up in chain [[link1],[link2]], [[parrallell1],[parallell2]] ...]
Chained = True   # If False puts all correlators above in one fit no matter how they are organised
Marginalised = False #True
SaveFit = True
smallsave = True #saves only the ground state non-oscillating and 3pts
svdnoise = False
priornoise = False
ResultPlots = False         # Tell what to plot against, "Q", "N","Log(GBF)", False
SvdFactor = 1.0                       # Multiplies saved SVD
PriorLoosener = 1.0                   # Multiplies all prior error by loosener
Nmax = 4                               # Number of exp to fit for 2pts in chained, marginalised fit
Nmin = 3                              #Number to start on
FitToGBF = False                     # If false fits to Nmax
##############################################################
setup = ['KG-S-BG','KG-V-BNG','KNG-T-BNG']
notwist0 = ['KNG','T'] #list any fits which do not use tw-0
non_oscillating = [] #any daughters which do no osciallate (only tw 0 is affected)
middle = 3/8                      #middle in Meff Aeff estimate 3/8 normal
gap = 1/14                        #gap in the Meff Aeff estimate 1/14 works well cannot exceed 1/8
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
    # average data 
    data = make_data('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']))
    # make models
    if Chained:
        modelsA,modelsB = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks)
    else: 
        models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks)
############################ Do chained fit #########################################################
    if Chained:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,middle,gap,notwist0,non_oscillating)
                GBF2 = do_chained_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,middle,gap,notwist0,non_oscillating)
                do_chained_fit(data,prior,N,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None)
######################### Do unchained fit ############################################################
    else:
        if FitToGBF:
            N = Nmin
            GBF1 = -1e10
            GBF2 = GBF1 + 10
            while GBF2-GBF1 > 1:
                GBF1 = GBF2
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,middle,gap,notwist0,non_oscillating)
                GBF2 = do_unchained_fit(data,prior,N,models,svdcut,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,GBF1)
                N += 1
            
        else:
            for N in range(Nmin,Nmax+1):
                prior = make_prior(Fit,N,allcorrs,currents,daughters,parents,PriorLoosener,data,middle,gap,notwist0,non_oscillating)
                do_unchained_fit(data,prior,N,models,svdcut,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,None)        


#####################################################################################################
            
    return()
main()
