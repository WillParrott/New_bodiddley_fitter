import gvar as gv
import corrfitter as cf
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
matplotlib.use('Agg')
plt.rc("font",**{"size":18})
import datetime
import os
import pickle
import copy
#from plotting import *
import lsqfit
lsqfit.nonlinear_fit.set(fitter='gsl_multifit',alg='subspace2D',scaler='more',solver='cholesky')#,solver='cholesky')
#################################### 
maxiter=5000
#######################################################################################################

def read_setup(setup):
    #Reads in setups, and strips out currents, parents and daughters, as well as which is which
    daughters = []
    currents = []
    parents = []
    for element in setup:
        lab = element.split('-')
        daughters.append(lab[0])
        currents.append(lab[1])
        parents.append(lab[2])
    return(daughters,currents,parents)
######################################################################################################

def strip_list(l): #Strips elemenst from list l
    stripped = ''
    for element in l:
        stripped = '{0}{1}'.format(stripped,element)
    return(stripped)

######################################################################################################


def make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents):
    #Removes things we do not want to fit, specified by FitMasses, FitTwists, FitTs assumes parents have varing mass and daughters varing twist
    j = 0
    for i in range(len(Fit['masses'])):
        if i not in FitMasses:
            del Fit['masses'][i-j]
            for element in set(parents):
                del Fit['tmaxes{0}'.format(element)][i-j]
            j += 1
    j = 0
    for i in range(len(Fit['twists'])):
        if i not in FitTwists:
            del Fit['twists'][i-j]
            for element in set(daughters):
                del Fit['tmaxes{0}'.format(element)][i-j]
            j += 1
    j = 0
    for i in range(len(Fit['Ts'])):
        if i not in FitTs:
            del Fit['Ts'][i-j]
            j += 1
    return()

#######################################################################################################

def make_data(filename,binsize):
    # Reads in filename.gpl, checks all keys have same configuration numbers, returns averaged data
    print('Reading data, binsize = ', binsize) 
    dset = cf.read_dataset(filename,binsize=binsize)
    sizes = []
    for key in dset:
        #print(key,np.shape(dset[key]))
        sizes.append(np.shape(dset[key]))
    if len(set(sizes)) != 1:
        print('Not all elements of gpl the same size')
        for key in dset:
            print(key,np.shape(dset[key]))
    return(gv.dataset.avg_data(dset))

######################################################################################################

def make_pdata(filename,models,binsize):
    # Reads in filename.gpl, checks all keys have same configuration numbers, returns averaged data 
    print('Reading data, binsize = ', binsize)
    dset = cf.read_dataset(filename,binsize=binsize)
    sizes = []
    for key in dset:
        #print(key,np.shape(dset[key]))
        sizes.append(np.shape(dset[key]))
    if len(set(sizes)) != 1:
        print('Not all elements of gpl the same size')
        for key in dset:
            print(key,np.shape(dset[key]))
    return(cf.process_dataset(dset, models))

#######################################################################################################

def effective_mass_calc(tag,correlator,tp):
    #finds the effective mass and amplitude of a two point correlator
    M_effs = []
    for t in range(2,len(correlator)-2):
        thing  = (correlator[t-2] + correlator[t+2])/(2*correlator[t]) 
        if thing >= 1:
            M_effs.append(gv.arccosh(thing)/2)
    #M_effs is all positive masses, we now take a rolling average of 4, and find where this changes the least
    rav = []
    for i in range(len(M_effs)-4):
        rav.append((M_effs[i] + M_effs[i+1] + M_effs[i+2] + M_effs[i+3])/4)
    M_eff = rav[0]
    diff = abs((rav[1] - rav[0]).mean)
    for i in range(1,len(rav)-1):
        if abs((rav[i+1]-rav[i]).mean) < diff:
            diff = abs((rav[i+1]-rav[i]).mean)
            M_eff = (rav[i] + rav[i+1])/2 
    return(M_eff)

######################################################################################################

def effective_amplitude_calc(tag,correlator,tp,M_eff,Fit,corr):
    #finds the effective mass and amplitude of a two point correlator
    tmin = Fit['tmin{0}'.format(corr)]
    A_effs = []
    if len(correlator) == tp:
        tmin = 0
    for t in range(tmin,tmin+len(correlator)):
        numerator = correlator[t-tmin]
        if numerator >= 0:
            A_effs.append( gv.sqrt(numerator/(gv.exp(-M_eff*t)+gv.exp(-M_eff*(tp-t)))))
    rav = []
    for i in range(len(A_effs)-4):
        rav.append((A_effs[i] + A_effs[i+1] + A_effs[i+2] + A_effs[i+3])/4)
    A_eff = rav[0]
    diff = abs((rav[1] - rav[0]).mean)
    for i in range(1,len(rav)-1):
        if abs((rav[i+1]-rav[i]).mean) < diff:
            diff = abs((rav[i+1]-rav[i]).mean)
            A_eff = (rav[i] + rav[i+1])/2
    an = gv.gvar(Fit['an'])
    if A_eff.sdev/A_eff.mean > 0.5:
        print('Replaced A_eff for {0} {1} -> {2}'.format(tag,A_eff,an))
        A_eff = an
    return(A_eff)

########################################################################################

def effective_V_calc(corr,daughter,parent,correlator,dcorr,pcorr,Fit,mass,twist,pA_eff,dA_eff):
    #finds the effective V_nn[0][0]
    tp = Fit['tp']
    T = Fit['Ts'][-1]
    dtmin = Fit['tmin{0}'.format(daughter)]
    ptmin = Fit['tmin{0}'.format(parent)]
    Vtmin = Fit['{0}tmin'.format(corr)]
    dcorr2 = []
    pcorr2 = []
    Vcorr2 = []
    V_effs = []
    
    #print(corr,daughter,parent,mass,twist)
    for i in range(dtmin):
        dcorr2.append(0)
    dcorr2.extend(dcorr)
    for i in range(int(tp/2)-len(dcorr2)+1):
        dcorr2.append(0)
    #print(dcorr2)

    for i in range(ptmin):
        pcorr2.append(0)
    pcorr2.extend(pcorr)
    for i in range(int(tp/2)-len(pcorr2)+1):
        pcorr2.append(0)
    #print(pcorr2)

    for i in range(Vtmin):
        Vcorr2.append(0)
    Vcorr2.extend(correlator)
    for i in range(T-len(Vcorr2)+1):
        Vcorr2.append(0)
    #print(Vcorr2)
    
    for t in range(T):
        numerator = Vcorr2[t]*pA_eff*dA_eff
        denominator = dcorr2[t]*pcorr2[T-t]
        if numerator != 0 and denominator !=0:
            V_effs.append(numerator/denominator)
    rav = []
    for i in range(len(V_effs)-4):
        rav.append((V_effs[i] + V_effs[i+1] + V_effs[i+2] + V_effs[i+3])/4)
    V_eff = rav[0]
    diff = abs((rav[1] - rav[0]).mean)
    for i in range(1,len(rav)-1):
        if abs((rav[i+1]-rav[i]).mean) < diff:
            diff = abs((rav[i+1]-rav[i]).mean)
            if (rav[i] + rav[i+1]) > 0:
                V_eff = (rav[i] + rav[i+1])/2
    V = gv.gvar(Fit['{0}Vnn0'.format(corr)])
    if abs((V_eff.mean-V).mean/(V_eff.mean-V).sdev) > 1:
        print('Replaced V_eff for {0} m {1} tw {2}: {3} --> {4}'.format(corr,mass,twist,V_eff,V))
        V_eff = V
    return(V_eff)

#######################################################################################################

def SVD_diagnosis(Fit,models,corrs,svdfac,currents):
    binsize = Fit['binsize']
    #Feed models and corrs (list of corrs in this SVD cut)
    if list(set(corrs).intersection(currents)) ==[]:
        filename = 'SVD/{0}{1}{2}{3}{4}{5}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(corrs),binsize)
    else:
        filename = 'SVD/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(corrs),strip_list(Fit['Ts']),binsize)
    #print(filename)
    for corr in corrs:
       if 'tmin{0}'.format(corr) in Fit:
           filename += '{0}'.format(Fit['tmin{0}'.format(corr)])
           for element in Fit['tmaxes{0}'.format(corr)]:
               filename += '{0}'.format(element)
       if '{0}tmin'.format(corr) in Fit:
           filename += '{0}'.format(Fit['{0}tmin'.format(corr)])
 
    #print(filename)
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        pickle_off = open(filename,"rb")
        svd = pickle.load(pickle_off)
        print('Loaded SVD for {0} : {1:.2g} x {2} = {3:.2g}'.format(corrs,svd,svdfac,svd*svdfac))
        pickle_off.close()
    else:
        print('Calculating SVD for {0}'.format(corrs))
        s = gv.dataset.svd_diagnosis(cf.read_dataset('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),binsize=binsize), models=models, nbstrap=20)
        svd = s.svdcut
        ######## save plot ##########################
        plt.figure()
        x = s.val / s.val[-1]
        ratio = s.bsval / s.val
        idx = x > s.mincut
        ratio = ratio[idx]
        x = x[idx]
        y = gv.mean(ratio)
        yerr = gv.sdev(ratio)
        plt.errorbar(x=x, y=y, yerr=yerr, fmt='+', color='b')
        sig = (2. / len(s.val)) ** 0.5
        plt.plot([x[0], x[-1]], [1. - sig, 1. - sig], 'k:')
        plt.axhline(1,ls='--',color='k')
        plt.axvline(s.svdcut,ls=':',color='g')
        #plt.axvline(0.013,ls='--',color='g')
        plt.xscale('log')
        plt.savefig('svd_plots/{0}.pdf'.format(filename.split('/')[1]))
        ###############################################
        pickle_on = open(filename,"wb")
        print('Calculated SVD for {0} : {1:.2g} x {2} = {3:.2g}'.format(corrs,svd,svdfac,svd*svdfac))
        pickle.dump(svd,pickle_on)
    return(svd*svdfac)

#######################################################################################################

def make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,svdfac,Chained,allcorrs,links,parrlinks):
    #several forms [(A,B,C,D)],[(A,B),(C),(D)],[(A,B),[(C),(D)]]
    #First make all models and then stick them into the correct chain
    models = collections.OrderedDict()
    tp = Fit['tp']
    for corr in set(parents):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for i,mass in enumerate(Fit['masses']):
                tag = Fit['{0}-Tag'.format(corr)].format(mass)
                models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1,-1)))

    for corr in set(daughters):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for i,twist in enumerate(Fit['twists']):
                tag = Fit['{0}-Tag'.format(corr)].format(twist)
                if twist == '0' and corr in notwist0:
                    pass
                elif twist == '0' and corr in non_oscillating:
                    models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag)), b=('{0}:a'.format(tag)), dE=('dE:{0}'.format(tag))))
                else:
                    models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1,-1)))

    for i,corr in enumerate(currents):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for  mass in Fit['masses']:
                for twist in Fit['twists']:
                    for T in Fit['Ts']:
                        tag = Fit['threePtTag{0}'.format(corr)].format(T,Fit['m_s'],mass,Fit['m_l'],twist)
                        ptag = Fit['{0}-Tag'.format(parents[i])].format(mass)
                        dtag = Fit['{0}-Tag'.format(daughters[i])].format(twist)
                        if twist == '0' and corr in notwist0:
                            pass
                        elif twist == '0' and daughters[i] in non_oscillating:
                            models['{0}'.format(corr)].append(cf.Corr3(datatag=tag, T=T, tmin=Fit['{0}tmin'.format(corr)], a=('{0}:a'.format(dtag)), dEa=('dE:{0}'.format(dtag)), b=('{0}:a'.format(ptag), 'o{0}:a'.format(ptag)), dEb=('dE:{0}'.format(ptag), 'dE:o{0}'.format(ptag)), sb=(1,-1), Vnn='{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist), Vno='{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)))

                        else:
                            models['{0}'.format(corr)].append(cf.Corr3(datatag=tag, T=T, tmin=Fit['{0}tmin'.format(corr)], a=('{0}:a'.format(dtag), 'o{0}:a'.format(dtag)), dEa=('dE:{0}'.format(dtag), 'dE:o{0}'.format(dtag)), sa=(1,-1), b=('{0}:a'.format(ptag), 'o{0}:a'.format(ptag)), dEb=('dE:{0}'.format(ptag), 'dE:o{0}'.format(ptag)), sb=(1,-1), Vnn='{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist), Vno='{0}Vno_m{1}_tw{2}'.format(corr,mass,twist),Von='{0}Von_m{1}_tw{2}'.format(corr,mass,twist),Voo='{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)))
    #Now we make these models into our chain calculating an svd cut for each. We make them in two halves so we can sndwich a marginalisation term if we like later
    
    if Chained:
        finalmodelsA = []
        finalmodelsB = []
        intermediate = []
        for key in links:
            link = [] #link is models in link
            for corr in links[key]:
                link.extend(models['{0}'.format(corr)]) 
            svd = SVD_diagnosis(Fit,link,links[key],svdfac,currents)
            finalmodelsA.append({'svdcut':svd})
            finalmodelsA.append(tuple(link))
        for key in parrlinks:
            link = [] #link is models in link
            for corr in parrlinks[key]:
                link.extend(models['{0}'.format(corr)]) 
            svd = SVD_diagnosis(Fit,link,parrlinks[key],svdfac,currents)
            intermediate.append({'svdcut':svd})
            intermediate.append(tuple(link))
        finalmodelsB.append(intermediate)
        return(finalmodelsA,finalmodelsB)
    else:
        finalmodels = []
        for corr in allcorrs:
            finalmodels.extend(models['{0}'.format(corr)])
        svd = SVD_diagnosis(Fit,finalmodels,allcorrs,svdfac,currents)
        return(tuple(finalmodels),svd)                
    

#######################################################################################################

def elements_in_FitCorrs(a):
    # reads [A,[B,C],[[D,E],F]] and interprets which elements will be chained and how. Returns alphabetical list of all elements, links in chain and links in parallell chain
    allcorrs = []
    links = collections.OrderedDict()
    parrlinks = collections.OrderedDict()
    for i in range(np.shape(a)[0]):
        links[i] =[]
        if len(np.shape(a[i])) == 0: #deals with one corr in chain 
            #print(a[i],i,'fit alone in chain')
            allcorrs.append(a[i])
            links[i].append(a[i])
        elif len(np.shape(a[i][0])) == 0 : #deals with multiple elements in chain 
            for j in range(len(a[i])):
                #print(a[i][j],i,'fit together in chain')
                allcorrs.append(a[i][j])
                links[i].append(a[i][j])
        else:
            del links[i]  #don't need thi key if it is in paralell
            for j in range(np.shape(a[i])[0]):
                parrlinks[j] = []
                if len(np.shape(a[i][j])) == 0: #deals with one corr in parr chain 
                    allcorrs.append(a[i][j])
                    parrlinks[j].append(a[i][j])
                else:                           # deals with multiple elements in parralell chain
                    for k in range(len(a[i][j])):
                        allcorrs.append(a[i][j][k])
                        parrlinks[j].append(a[i][j][k])
    return(sorted(allcorrs),links,parrlinks)

######################################################################################################

def make_prior(Fit,N,allcorrs,currents,daughters,parents,loosener,data,notwist0,non_oscillating):
    No = N  # number of oscillating exponentials
    prior =  gv.BufferDict()
    tw_corr = True
    otw_corr = True
    if len(daughters) != 0 and '0' in Fit['twists'] and tw_corr:
        for corr in set(daughters).intersection(allcorrs):
            prior['d2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
            prior['c2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
        print('Daughter twists correlated')
    if len(daughters) != 0 and '0' in Fit['twists'] and otw_corr:
        for corr in set(daughters).intersection(allcorrs):
            prior['oc2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
        print('Daughter oscillating twists correlated')
    
    tp = Fit['tp']
    En = '{0}({1})'.format(0.5*Fit['a'],0.25*Fit['a']*loosener) #Lambda with error of half
    an = '{0}({1})'.format(gv.gvar(Fit['an']).mean,gv.gvar(Fit['an']).sdev*loosener)
    aon = '{0}({1})'.format(gv.gvar(Fit['aon']).mean,gv.gvar(Fit['aon']).sdev*loosener)
    for corr in allcorrs:
        if corr in parents:
            for mass in Fit['masses']:
                tag = Fit['{0}-Tag'.format(corr)].format(mass)
                M_eff = effective_mass_calc(tag,data[tag],tp)
                a_eff = effective_amplitude_calc(tag,data[tag],tp,M_eff,Fit,corr)
                # Parent
                prior['log({0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                prior['log({0}:a)'.format(tag)][0] = gv.log(gv.gvar(a_eff.mean,loosener*Fit['loosener']*a_eff.mean))
                prior['log(dE:{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean,loosener*Fit['Mloosener']*M_eff.mean))
                # Parent -- oscillating part
                prior['log(o{0}:a)'.format(tag)] = gv.log(gv.gvar(No * [an]))
                prior['log(dE:o{0})'.format(tag)] = gv.log(gv.gvar(No * [En]))
                prior['log(dE:o{0})'.format(tag)][0] = gv.log(gv.gvar((M_eff+gv.gvar(En)*(4/5)).mean,loosener*Fit['oMloosener']*((M_eff+gv.gvar(En)*(4/5)).mean)))
                
                
        if corr in daughters:
            for twist in Fit['twists']:
                if twist =='0' and corr in notwist0:
                    pass
                else:
                    ap2 = 3*(np.pi*float(twist)/Fit['L'])**2
                    #print(twist,ap2)
                    tag0 = Fit['{0}-Tag'.format(corr)].format('0')
                    M_eff = np.sqrt(effective_mass_calc(tag0,data[tag0],tp)**2 +  ap2)   #from dispersion relation
                    tag = Fit['{0}-Tag'.format(corr)].format(twist)
                    a_eff = effective_amplitude_calc(tag,data[tag],tp,M_eff,Fit,corr)
                    # Daughter
                    prior['log({0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                    prior['log(dE:{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                    #prior['log(dE:{0})'.format(tag)][1] = gv.log(gv.gvar(gv.gvar(En).mean,0.01*gv.gvar(En).mean))
                    if twist !='0' and '0' in Fit['twists'] and 'log(dE:{0})'.format(tag0) in prior and tw_corr:
                        prior['log(dE:{0})'.format(tag)][0] = gv.log(gv.sqrt(prior['dE:{0}'.format(tag0)][0]**2 + ap2) * (1 + prior['c2_{0}'.format(corr)]*ap2/(np.pi)**2) )
                        prior['log({0}:a)'.format(tag)][0] = gv.log((prior['{0}:a'.format(tag0)][0]/gv.sqrt(gv.sqrt(1 + ap2/(prior['dE:{0}'.format(tag0)][0])**2))) * (1 + prior['d2_{0}'.format(corr)]*ap2/(np.pi)**2) )
                        
                    else: 
                        prior['log(dE:{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean,loosener*Fit['Mloosener']*M_eff.mean))
                        prior['log({0}:a)'.format(tag)][0] = gv.log(gv.gvar(a_eff.mean,loosener*Fit['loosener']*a_eff.mean))
                    # Daughter -- oscillating part
                    if twist =='0' and corr in non_oscillating:
                        pass
                    else:
                        newaon = aon
                        if twist == '0':
                            newaon = '{0}({1})'.format(gv.gvar(aon).mean/4,gv.gvar(aon).mean/2) #v small in the case of tw0
                        prior['log(o{0}:a)'.format(tag)] = gv.log(gv.gvar(No * [newaon]))
                        prior['log(dE:o{0})'.format(tag)] = gv.log(gv.gvar(No * [En]))
                        if twist !='0' and '0' in Fit['twists'] and 'log(dE:o{0})'.format(tag0) in prior and otw_corr:
                            prior['log(dE:o{0})'.format(tag)][0] = gv.log(gv.sqrt(prior['dE:o{0}'.format(tag0)][0]**2 + ap2) * (1 + prior['oc2_{0}'.format(corr)]*ap2/(np.pi)**2) )
                            #prior['log(o{0}:a)'.format(tag)][0] = gv.log((prior['o{0}:a'.format(tag0)][0]/gv.sqrt(1 + ap2/(prior['dE:o{0}'.format(tag0)][0])**2)) * (1 + prior['od2']*ap2/(np.pi)**2) )
                            prior['log(o{0}:a)'.format(tag)][0] = gv.log(gv.gvar(gv.gvar(newaon).mean,loosener*Fit['oloosener']*gv.gvar(newaon).mean))
                        else:
                            prior['log(dE:o{0})'.format(tag)][0] = gv.log(gv.gvar((M_eff+gv.gvar(En)/2).mean,loosener*Fit['oMloosener']*((M_eff+gv.gvar(En)/2).mean))) # kaon splitting 
                            #prior['log(dE:o{0})'.format(tag)][0] = gv.log(prior['dE:{0}'.format(tag)][0] + gv.gvar(En))
                            prior['log(o{0}:a)'.format(tag)][0] = gv.log(gv.gvar(gv.gvar(newaon).mean,loosener*Fit['oloosener']*gv.gvar(newaon).mean))
                        
        if corr in currents:
            for mass in Fit['masses']:
                for twist in Fit['twists']:
                    if twist =='0' and corr in notwist0:
                        pass
                    else: 
                        daughter=daughters[currents.index(corr)]
                        parent=parents[currents.index(corr)]
                        dcorr = data[Fit['{0}-Tag'.format(daughter)].format(twist)]
                        pcorr = data[Fit['{0}-Tag'.format(parent)].format(mass)]
                        correlator = data[Fit['threePtTag{0}'.format(corr)].format(Fit['Ts'][-1],Fit['m_s'],mass,Fit['m_l'],twist)]
                        ptag = Fit['{0}-Tag'.format(parent)].format(mass)
                        pM_eff = effective_mass_calc(ptag,data[ptag],tp)
                        pa_eff = effective_amplitude_calc(ptag,data[ptag],tp,pM_eff,Fit,parent)

                        dtag = Fit['{0}-Tag'.format(daughter)].format(twist)
                        dM_eff = effective_mass_calc(dtag,data[dtag],tp)
                        da_eff = effective_amplitude_calc(dtag,data[dtag],tp,dM_eff,Fit,daughter)

                        V_eff = effective_V_calc(corr,daughter,parent,correlator,dcorr,pcorr,Fit,mass,twist,da_eff,pa_eff)
                        if V_eff.mean != gv.gvar(Fit['{0}Vnn0'.format(corr)]).mean:
                            Vnn0 = '{0}({1})'.format(V_eff.mean,loosener*V_eff.mean*Fit['Vloosener'])
                        else:
                            Vnn0 = '{0}({1})'.format(V_eff.mean,loosener*V_eff.sdev)
                        Vn = '{0}({1})'.format(gv.gvar(Fit['{0}Vn'.format(corr)]).mean,loosener*gv.gvar(Fit['{0}Vn'.format(corr)]).sdev)
                        V0 = '{0}({1})'.format(gv.gvar(Fit['{0}V0'.format(corr)]).mean,loosener*gv.gvar(Fit['{0}V0'.format(corr)]).sdev)
                    if twist =='0' and corr in notwist0:
                        pass
                    elif twist =='0' and daughters[currents.index(corr)] in non_oscillating :
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [No* [Vn]])
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)     
                    else:
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [No * [Vn]])
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(No * [No * [Vn]])
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(No * [N * [Vn]])
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
           # for key in prior: 
           #     if key[0] == corr:
           #         for i in range(1,N):
           #             for j in range(1,N):
           #                 prior[key][i][j] = gv.gvar('0.0(5)')
    
                   
    #print(prior)
    return(prior)
            
######################################################################################################

def get_p0(Fit,fittype,Nexp,allcorrs,prior,FitCorrs):
    # We want to take in several scenarios in this order, choosing the highest in preference. 
    # 1) This exact fit has been done before, modulo priors, svds t0s etc
    # 2) Same but different type of fit, eg marginalised 
    # 3) This fit has been done before with Nexp+1
    # 4) This fit has been done beofore with Nexp-1
    # 5a) Some elemnts have bene fitted to Nexp before,
    # 5b) Some elements of the fit have been fitted in other combinations before

    
    filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),FitCorrs,strip_list(Fit['Ts']),fittype,Nexp)
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),Nexp)
    filename3 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),FitCorrs,strip_list(Fit['Ts']),fittype,Nexp+1)
    filename4 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),FitCorrs,strip_list(Fit['Ts']),fittype,Nexp-1)
    filename5a = 'p0/{0}{1}{2}'.format(Fit['conf'],Fit['filename'],Nexp)
    filename5b = 'p0/{0}{1}'.format(Fit['conf'],Fit['filename'])
    #case 1
    if os.path.isfile(filename1):
        p0 = gv.load(filename1)
        print('Loaded p0 from exact fit')
    #case 2
    
    elif os.path.isfile(filename2):
        p0 = gv.load(filename2)
        print('Loaded p0 from exact fit of different type')
    #case 3    
    elif os.path.isfile(filename3):
        p0 = gv.load(filename3)
        print('Loaded p0 from exact fit Nexp+1')
        
    #case 4    
    elif os.path.isfile(filename4):
        p0 = gv.load(filename4)
        print('Loaded p0 from exact fit Nexp-1')
        
    #case 5    
    elif os.path.isfile(filename5b):
        p0 = gv.load(filename5b)
        print('Loaded global p0')
        if os.path.isfile(filename5a):
            pnexp = gv.load(filename5a)
            for key in pnexp:
                if key in prior:
                    if key not in p0:
                        print('Error: {0} in global Nexp but not in global fit'.format(key))
                        p0[key] = pnexp[key]
                    del p0[key]
                    p0[key] = pnexp[key]
                    print('Loaded {0} p0 from global Nexp'.format(key))
    
    else:
        p0 = None
    return(p0)
######################################################################################################

def update_p0(p,finalp,Fit,fittype,Nexp,allcorrs,FitCorrs,Q,marg=False):
    # We want to take in several scenarios in this order 
    # 1) This exact fit has been done before, modulo priors, svds t0s etc
    # 2) Same but different type of fit, eg marginalised 
    # 3) Global Nexp
    # 4) Global
    # 5) if Marg is True, we don't want to save anything but filename 1 as Nexp = nmarg and is not similar to if we do other fits
    filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),FitCorrs,strip_list(Fit['Ts']),fittype,Nexp)
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),Nexp)
    filename3 = 'p0/{0}{1}{2}'.format(Fit['conf'],Fit['filename'],Nexp)
    filename4 = 'p0/{0}{1}'.format(Fit['conf'],Fit['filename'])

    #case 1
    
    for element in ['c2','d2','oc2']:
        for corr in allcorrs:
            if '{0}_{1}'.format(element,corr) in p:
                del p['{0}_{1}'.format(element,corr)]
    
    for element in ['c2','d2','oc2']:
        for corr in allcorrs:
            if '{0}_{1}'.format(element,corr) in finalp:
                del finalp['{0}_{1}'.format(element,corr)]
    gv.dump(p,filename1)
    if marg == False:
        #case 2
        gv.dump(finalp,filename2)

        #case 3
        if os.path.isfile(filename3) and Q > 0.05:
            p0 = gv.load(filename3) #load exisiting global Nexp
            for key in finalp:  # key in this output
                p0[key] =  finalp[key]  #Update exisiting and add new
            gv.dump(p0,filename3)
    
        else:
            gv.dump(finalp,filename3)

        if os.path.isfile(filename4) and Q > 0.05:
            p0 = gv.load(filename4) # load existing, could be any length
            for key in finalp:  # key in new 
                if key in p0: # if 
                    if len(np.shape(p0[key])) == 1 and len(p0[key]) <= Nexp:
                        #print('shape p0[key]',np.shape(p0[key]),key)
                        del p0[key]
                        p0[key] = finalp[key]
                        print('Updated global p0 {0}'.format(key))
                    elif np.shape(p0[key])[0] <= Nexp:
                        #print('shape p0[key]',np.shape(p0[key]),key)
                        del p0[key]
                        p0[key] = finalp[key]
                        print('Updated global p0 {0}'.format(key))
                else:
                    p0[key] =  finalp[key]
                    print('Added new element to global p0 {0}'.format(key))
            gv.dump(p0,filename4)
        else:
            gv.dump(finalp,filename4)
    return()

######################################################################################################

def save_fit(fit,Fit,allcorrs,fittype,Nexp,SvdFactor,PriorLoosener,currents,smallsave):
    filename = 'Fits/{0}{1}{2}{3}{4}{5}{6}_Nexp{7}_sfac{8}_pfac{9}_Q{10:.2f}_chi{11:.3f}_sm{12}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,SvdFactor,PriorLoosener,fit.Q,fit.chi2/fit.dof,smallsave)
    for corr in allcorrs:
        if corr in currents:
            filename += '_{0}tmin{1}'.format(corr,Fit['{0}tmin'.format(corr)])
    savedict = gv.BufferDict()
    if smallsave:
        for key in fit.p:
            if key[0] == 'l':
                key2 = key.split('(')[1].split(')')[0]
                if key2.split(':')[0] =='dE' and key2.split(':')[1][0] != 'o':
                    savedict[key] = [fit.palt[key][0]]
            elif key[2] =='n' and key[3] == 'n':
                savedict[key] = [[fit.palt[key][0][0]]]
    elif smallsave == False:
        savedict = fit.p
    print('Started gv.gdump to {1}, smallsave = {0}'.format(smallsave,'{0}.pickle'.format(filename)),datetime.datetime.now())        
    gv.gdump(savedict,'{0}.pickle'.format(filename))
    print('Finished gv.gdump fit, starting save fit output',datetime.datetime.now())
    f = open('{0}.txt'.format(filename),'w')
    f.write(fit.format(pstyle='v'))
    f.close()
    print('Finished save fit output',datetime.datetime.now())
    return()
    

######################################################################################################

def do_chained_fit(data,prior,Nexp,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,save,smallsave,GBF):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with no marginalisation Nexp = NMax
    models = copy.deepcopy(modelsA)
    if len(modelsB[0]) !=0: 
        models.extend(modelsB)
    print('Models',models)
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'chained',Nexp,allcorrs,prior,FitCorrs) 
    print(30 * '=','Chained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.chained_lsqfit(data=data, prior=prior, p0=p0, noise=noise,debug=True)
    update_p0([f.pmean for f in fit.chained_fits.values()],fit.pmean,Fit,'chained',Nexp,allcorrs,FitCorrs,fit.Q) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)
        print_Z_V(fit.p,Fit,allcorrs)
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return()
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)
        print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)

######################################################################################################

def do_chained_marginalised_fit(data,prior,Nexp,modelsA,modelsB,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,save,smallsave,GBF,Marginalised):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with marginalisation nterm = nexp,nexp Nmarg=Marginalisation us in p0 bits
    models = copy.deepcopy(modelsA)
    if len(modelsB[0]) !=0:
        models.append(dict(nterm=(Nexp,Nexp))) 
        models.extend(modelsB)
    else:
        print('Marginalisation not applied as no parrallelised models')
    print('Models',models)
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'chained-marginalised_N{0}{0}'.format(Nexp),Marginalised,allcorrs,prior,FitCorrs)
    print(30 * '=','Chained-marginalised','Nexp =',Marginalised,'nterm = ({0},{0})'.format(Nexp),'Date',datetime.datetime.now())
    fit = fitter.chained_lsqfit(data=data, prior=prior, p0=p0, noise=noise,debug=True)
    update_p0([f.pmean for f in fit.chained_fits.values()],fit.pmean,Fit,'chained-marginalised_N{0}{0}'.format(Nexp),Marginalised,allcorrs,FitCorrs,fit.Q,True) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)#.format(pstyle='m'))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)
        print_Z_V(fit.p,Fit,allcorrs)
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained-marginalised_N{0}{0}'.format(Nexp),Marginalised,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return()
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit)#.format(pstyle='m'))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)
        print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained-marginalised_N{0}{0}'.format(Nexp),Marginalised,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)

######################################################################################################

def do_unchained_fit(data,prior,Nexp,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,save,smallsave,GBF):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with no marginalisation Nexp = NMax
    print('Models',models)
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'unchained',Nexp,allcorrs,prior,allcorrs) # FitCorrs = allcorrs 
    print(30 * '=','Unchained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.lsqfit(pdata=data, prior=prior, p0=p0, svdcut=svdcut, noise=noise,debug=True)
    update_p0(fit.pmean,fit.pmean,Fit,'unchained',Nexp,allcorrs,allcorrs,fit.Q) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)#,Fit)
        print_Z_V(fit.p,Fit,allcorrs)
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'unchained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return()
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior)#,Fit)
        print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up more than 1: {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'unchained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)


######################################################################################################

def print_p_p0(p,p0,prior):
    print('{0:<30}{1:<20}{2:<40}{3:<20}'.format('key','p','p0','prior'))
    for key in prior:
        if len(np.shape(p[key])) ==1 :
            for element in range(len(p[key])):
                if element == 0:
                    print('{0:<30}{1:<20}{2:<40}{3:<20}'.format(key,p[key][element],p0[key][element],prior[key][element]))
                else:
                    print('{0:>30}{1:<20}{2:<40}{3:<20}'.format('',p[key][element],p0[key][element],prior[key][element]))
    return()

#####################################################################################################

def print_results(p,prior):#,Fit):
    print(100*'-')
    print('{0:<30}{1:<15}{2:<15}{3:<15}{4}'.format('key','p','p error','prior','prior error'))
    print(100*'-')
    print('Ground state energies')
    print(100*'-')
    for key in prior:
        if key[0] == 'l':
            key = key.split('(')[1].split(')')[0]
        if key.split(':')[0] =='dE' and key.split(':')[1][0] != 'o':
            print('{0:<30}{1:<15}{2:<15.3%}{3:<15}{4:.2%}'.format(key,p[key][0],p[key][0].sdev/p[key][0].mean,prior[key][0],prior[key][0].sdev/prior[key][0].mean))
            #if '{0}'.format(key.split(':')[1]) == Fit['BG-Tag'].format(Fit['masses'][0]):
             #   print('split: ', p['dE:{0}'.format(Fit['BNG-Tag'].format(Fit['masses'][0]))][0]-p[key][0])  
    print('')
    print('Oscillating ground state energies')
    print(100*'-')
    for key in prior:
        if key[0] == 'l':
            key = key.split('(')[1].split(')')[0]
        if key.split(':')[0] =='dE' and key.split(':')[1][0] == 'o':
            print('{0:<30}{1:<15}{2:<15.3%}{3:<15}{4:.2%}'.format(key,p[key][0],p[key][0].sdev/p[key][0].mean,prior[key][0],prior[key][0].sdev/prior[key][0].mean))
    print('')
    print('V_nn[0][0]')
    print(100*'-')
    for key in prior:
        if key[1] != '2' and key[2] =='n' and key[3] == 'n':
            print('{0:<30}{1:<15}{2:<15.3%}{3:<15}{4:.2%}'.format(key,p[key][0][0],p[key][0][0].sdev/p[key][0][0].mean,prior[key][0][0],prior[key][0][0].sdev/prior[key][0][0].mean))
    print(100*'-')
    return()
#####################################################################################################

def make_Z_V(m_h,m_s,M_parent,M_daughter,S,V):
    Z_V = (m_h-m_s)/(M_parent-M_daughter) * S/V
    return(Z_V)

#####################################################################################################
# needs generalising 
#####################################################################################################
def print_Z_V(p,Fit,allcorrs):
    if 'S' in allcorrs and 'V' in allcorrs:
        print(100*'-')
        for mass in Fit['masses']:
            M_parent = p['dE:{0}'.format(Fit['{0}-Tag'.format('BG')].format(mass))][0]
            M_daughter = p['dE:{0}'.format(Fit['{0}-Tag'.format('KG')].format('0'))][0]
            S = p['SVnn_m{0}_tw0'.format(mass)][0][0]
            V = p['VVnn_m{0}_tw0'.format(mass)][0][0]
            Z_V = make_Z_V(float(mass),float(Fit['m_s']),M_parent,M_daughter,S,V)
            print("Mass = {0} Z_V = {1}".format(mass,Z_V))
        print(100*'-')
    return()

#####################################################################################################
