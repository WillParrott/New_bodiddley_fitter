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

#######################################################################################################

def unmake_gvar_vec(vec):
    #A function which extracts the mean and standard deviation of a list of gvars
    mean = []
    sdev = []
    for element in vec:
        mean.append(vec[element].mean)
        sdev.append(vec[element].sdev)
    return(mean,sdev)

#######################################################################################################

def create_single_plot(filename,location,x,y,xlabel,ylabel):
    #Plots x against y where x is a list of floats and y of gvars
    figsca = 16 
    ymean,yerr = unmake_gvar_vec(y)
    plt.figure(filename,figsize=((figsca,2*figsca/(1+np.sqrt(5)))))
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.errorbar(x,ymean,yerr=yerr,fmt='ko',ms=12)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=30)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlabel(rxlabel,fontsize=30)
    plt.ylabel(rylabel,fontsize=30)
    plt.tight_layout()
    plt.savefig('Plots/{0}/{1}'.format(location,filename))
    
#######################################################################################################

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

def make_data(filename):
    # Reads in filename.gpl, checks all keys have same configuration numbers, returns averaged data 
    dset = cf.read_dataset(filename)
    sizes = []
    for key in dset:
        sizes.append(np.shape(dset[key]))
    if len(set(sizes)) != 1:
        print('Not all elements of gpl the same size')
        for key in dset:
            print(key,np.shape(dset[key]))
    return(gv.dataset.avg_data(dset))

#######################################################################################################

def effective_mass_calc(tag,correlator,plot,tp,middle,gap):
    #finds the effective mass and amplitude of a two point correlator
    M_effs = []
    for t in range(2,tp-2):
        thing  = (correlator[t-2] + correlator[t+2])/(2*correlator[t]) 
        if thing >= 1:
            M_effs.append(gv.arccosh(thing)/2)
        else:
            M_effs.append(0)
    denom = 0
    M_eff = 0
    for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):
        if M_effs[i] != 0:
            M_eff += M_effs[i]
            denom += 1
    M_eff = M_eff/denom    
    if plot:
        create_single_plot(tag,'Meff',t,M_effs,'$t$','$M_eff$') # fix this
    return(M_eff)

######################################################################################################

def effective_amplitude_calc(tag,correlator,plot,tp,middle,gap,M_eff):
    #finds the effective mass and amplitude of a two point correlator
    A_effs = []
    for t in range(2,tp-2):
        numerator = correlator[t]
        if numerator >= 0:
            A_effs.append(gv.sqrt(numerator/(gv.exp(-M_eff*t)+gv.exp(-M_eff*(tp-t)))))
        else:
            A_effs.append(0)
    denom = 0
    A_eff = 0
    for i in list(range(int(tp*(middle-gap)),int(tp*(middle+gap))))+list(range(int(tp*(1-middle-gap)),int(tp*(1-middle+gap)))):
        if A_effs[i] != 0:
            A_eff += A_effs[i]
            denom += 1
    A_eff = A_eff/denom    
    if plot:
        create_single_plot(tag,'Aeff',t,A_effs,'$t$','$A_eff$')
    return(A_eff)

#######################################################################################################

def SVD_diagnosis(Fit,models,corrs,svdfac):
    #Feed models and corrs (list of corrs in this SVD cut)
    filename = 'SVD/{0}{1}{2}{3}{4}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],corrs)
    #print(filename)
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        pickle_off = open(filename,"rb")
        svd = pickle.load(pickle_off)
        print('Loaded SVD for {0} : {1:.2g} x {2} = {3:.2g}'.format(corrs,svd,svdfac,svd*svdfac))
        pickle_off.close()
    else:
        print('Calculating SVD for {0}'.format(corrs))
        svd = gv.dataset.svd_diagnosis(cf.read_dataset(Fit['filename']), models=models, nbstrap=20).svdcut
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
                models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1.,-1.)))

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
                    models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1.,-1.)))

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
            svd = SVD_diagnosis(Fit,link,links[key],svdfac)
            finalmodelsA.append({'svdcut':svd})
            finalmodelsA.append(tuple(link))
        for key in parrlinks:
            link = [] #link is models in link
            for corr in parrlinks[key]:
                link.extend(models['{0}'.format(corr)]) 
            svd = SVD_diagnosis(Fit,link,parrlinks[key],svdfac)
            intermediate.append({'svdcut':svd})
            intermediate.append(tuple(link))
        finalmodelsB.append(intermediate)
        return(finalmodelsA,finalmodelsB)
    else:
        finalmodels = []
        for corr in allcorrs:
            finalmodels.append(models['0'.format(corr)])
        svd = SVD_diagnosis(Fit,finalmodels,allcorrs,svdfac)
        return(finalmodels,svd)                
    

#######################################################################################################

def elements_in_FitCorrs(a):
    # reads [A,[B,C],[[D,E],F]] and interprets which elements will be chained and how. Returns list of all elements, links in chain and links in parallell chain
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
    return(allcorrs,links,parrlinks)

######################################################################################################

def make_prior(Fit,N,allcorrs,currents,daughters,parents,loosener,data,plot,middle,gap,notwist0,non_oscillating):
    tp = Fit['tp']
    prior =  gv.BufferDict()
    En = '{0}({1})'.format(0.5*Fit['a'],0.25*Fit['a']*loosener) #Lambda with error of half
    an = '{0}({1})'.format(gv.gvar(Fit['an']).mean,gv.gvar(Fit['an']).sdev*loosener)
    for corr in allcorrs:
        if corr in parents:
            for mass in Fit['masses']:
                tag = Fit['{0}-Tag'.format(corr)].format(mass)
                M_eff = effective_mass_calc(tag,data[tag],plot,tp,middle,gap)
                a_eff = effective_amplitude_calc(tag,data[tag],plot,tp,middle,gap,M_eff)
                # Parent
                prior['log({0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                prior['log({0}:a)'.format(tag)][0] = gv.log(gv.gvar(a_eff.mean,loosener*Fit['loosener']*a_eff.mean))
                prior['log(dE:{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean,loosener*Fit['Mloosener']*M_eff.mean))
                # Parent -- oscillating part
                prior['log(o{0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                prior['log(dE:o{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean+gv.gvar(En).mean,loosener*Fit['oMloosener']*(M_eff.mean+gv.gvar(En).mean)))
                
        if corr in daughters:
            for twist in Fit['twists']:
                if twist =='0' and corr in notwist0:
                    pass
                else:
                    tag = Fit['{0}-Tag'.format(corr)].format(twist)
                    M_eff = effective_mass_calc(tag,data[tag],plot,tp,middle,gap)
                    a_eff = effective_amplitude_calc(tag,data[tag],plot,tp,middle,gap,M_eff)
                    # Daughter
                    prior['log({0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                    prior['log(dE:{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                    prior['log({0}:a)'.format(tag)][0] = gv.log(gv.gvar(a_eff.mean,loosener*Fit['loosener']*a_eff.mean))
                    prior['log(dE:{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean,loosener*Fit['Mloosener']*M_eff.mean))
                    # Daughter -- oscillating part
                    if twist =='0' and corr in non_oscillating:
                        pass
                    else:
                        prior['log(o{0}:a)'.format(tag)] = gv.log(gv.gvar(N * [an]))
                        prior['log(dE:o{0})'.format(tag)] = gv.log(gv.gvar(N * [En]))
                        prior['log(dE:o{0})'.format(tag)][0] = gv.log(gv.gvar(M_eff.mean+gv.gvar(En).mean,loosener*Fit['oMloosener']*(M_eff.mean+gv.gvar(En).mean)))
        if corr in currents:
            for mass in Fit['masses']:
                for twist in Fit['twists']:
                    Vnn0 = '{0}({1})'.format(gv.gvar(Fit['{0}Vnn0'.format(corr)]).mean,loosener*gv.gvar(Fit['{0}Vnn0'.format(corr)]).sdev)
                    Vn = '{0}({1})'.format(gv.gvar(Fit['{0}Vn'.format(corr)]).mean,loosener*gv.gvar(Fit['{0}Vn'.format(corr)]).sdev)
                    V0 = '{0}({1})'.format(gv.gvar(Fit['{0}V0'.format(corr)]).mean,loosener*gv.gvar(Fit['{0}V0'.format(corr)]).sdev)
                    if twist =='0' and corr in notwist0:
                        pass
                    elif twist =='0' and daughters[currents.index(corr)] in non_oscillating :
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)     
                    else:
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0)
    
                   
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

    
    filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,FitCorrs,Fit['Ts'],fittype,Nexp)
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,Fit['Ts'],Nexp)
    filename3 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,FitCorrs,Fit['Ts'],fittype,Nexp+1)
    filename4 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,FitCorrs,Fit['Ts'],fittype,Nexp-1)
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
                    del p0[key]
                    p0[key] = pnexp[key]
                    print('Loaded {0} p0 from global Nexp'.format(key))
    
    else:
        p0 = None
    return(p0)
######################################################################################################

def update_p0(p,finalp,Fit,fittype,Nexp,allcorrs,FitCorrs):
    # We want to take in several scenarios in this order 
    # 1) This exact fit has been done before, modulo priors, svds t0s etc
    # 2) Same but different type of fit, eg marginalised 
    # 3) Global Nexp
    # 4) Global
    filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,FitCorrs,Fit['Ts'],fittype,Nexp)
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,Fit['Ts'],Nexp)
    filename3 = 'p0/{0}{1}{2}'.format(Fit['conf'],Fit['filename'],Nexp)
    filename4 = 'p0/{0}{1}'.format(Fit['conf'],Fit['filename'])

    #case 1
    gv.dump(p,filename1)
    
    #case 2
    gv.dump(finalp,filename2)

    #case 3
    if os.path.isfile(filename3):
        p0 = gv.load(filename3) #load exisiting global Nexp
        for key in finalp:  # key in this output
            p0[key] =  finalp[key]  #Update exisiting and add new
        gv.dump(p0,filename3)
    
    else:
        gv.dump(finalp,filename3)

    if os.path.isfile(filename4):
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

def save_fit(fit,Fit,allcorrs,fittype,Nexp,SvdFactor,PriorLoosener,currents):
    filename = 'Fits/{0}{1}{2}{3}{4}{5}{6}_Nexp{7}_sfac{8}_pfac{9}_Q{10:.2f}_chi{11:.3f}'.format(Fit['conf'],Fit['filename'],Fit['masses'],Fit['twists'],allcorrs,Fit['Ts'],fittype,Nexp,SvdFactor,PriorLoosener,fit.Q,fit.chi2/fit.dof)
    for corr in allcorrs:
        if corr in currents:
            filename += '{0}_tmin{1}'.format(corr,Fit['{0}tmin'.format(corr)])
    #print(filename)        
    gv.dump(fit.p,'{0}.pickle'.format(filename))
    f = open('{0}.txt'.format(filename),'w')
    f.write(fit.format(pstyle='v'))
    f.close
    return()

######################################################################################################

def do_chained_fit(data,prior,Nexp,modelsA,modelsB,Fit,svdnoise,priornoise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,save,GBF):
    #do chained fit with no marginalisation Nexp = NMax
    if len(modelsB[0]) !=0: 
        modelsA.extend(modelsB)
    models = modelsA
    print('Models',models)
    fitter = cf.CorrFitter(models=models, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'chained',Nexp,allcorrs,prior,FitCorrs)
    #print('p0',p0)
    print(30 * '=','Chained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.chained_lsqfit(data=data, prior=prior, p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)
    if fit.Q > 0.05: #threshold for a 'good' fit
        update_p0([f.pmean for f in fit.chained_fits.values()],fit.pmean,Fit,'chained',Nexp,allcorrs,FitCorrs) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents)
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
        print('log(GBF) went up {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents)
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