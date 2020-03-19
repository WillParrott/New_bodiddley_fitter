import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
matplotlib.use('Agg')
plt.rc("font",**{"size":18})
from functions import *
import io
import sys
import os
#####################################################################################################
figsca = 16 
#####################################################################################################

def unmake_gvar_vec(vec):
    #A function which extracts the mean and standard deviation of a list of gvars
    mean = []
    sdev = []
    for element in vec:
        mean.append(vec[element].mean)
        sdev.append(vec[element].sdev)
    return(mean,sdev)

######################################################################################################


def create_t_plot(location,filename,y1,y1label,y2=None,y2label=None,y3=None,y3label=None):
    #Plots x against y where x is a list of floats and y of gvars
    y1mean,y1err = unmake_gvar_vec(y1)
    x1 = np.arange(0,len(y1mean))
    if y2 != None:
        y2mean,y2err = unmake_gvar_vec(y2)
        x2 = np.arange(0,len(y2mean))
    if y3 != None:
        y3mean,y3err = unmake_gvar_vec(y3)
        x3 = np.arange(0,len(y3mean))
        
    plt.figure(filename,figsize=((figsca,2*figsca/(1+np.sqrt(5)))))
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.errorbar(x1,y1mean,yerr=y1err,fmt='ko',ms=12,label=y1label)
    if y2 != None:
        plt.errorbar(x2,y2mean,yerr=y2err,fmt='ro',ms=12,label=y2label)
    if y3 != None:
        plt.errorbar(x3,y3mean,yerr=y3err,fmt='bo',ms=12,label=y3label)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=30)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlabel('t',fontsize=30)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=18,frameon=False) 
    #plt.ylabel(r'{0}'.format(ylabel),fontsize=30)
    plt.tight_layout()
    plt.savefig('{0}/{1}'.format(location,filename))
    return()
######################################################################################################

def plots(Fit,daughters,parents,currents):
    directory = './Plots/{0}'.format(Fit['filename'])
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Creating all plots for {0}".format(Fit['filename']))
        data = make_data(Fit['filename'])
    else:
        return()
 ######### daughters #########
    for twist in Fit['twists']:
        y = collections.OrderedDict()
        ylog = collections.OrderedDict()
        for i,corr in enumerate(set(daughters)):
            y[i] = []
            tag = Fit['{0}-Tag'.format(corr)].format(twist)
            filename = '{0}_tw{1}'.format(Fit['filename'],twist)
            for t in range(1,int(tp/2+1)):
                y[i].append((data[tag][t]+data[tag][tp-t])/2)
            if len(set(daughters)) == 1:
                create_t_plot(directory,filename,y[0],set(daughters)[0])
            else:
                create_t_plot(directory,filename,y[0],set(daughters)[0],y[1],set(daughters)[1])
            ylog[i] = []
            filename = '{0}log_tw{1}'.format(Fit['filename'],twist)
            for t in range(1,int(tp/2+1)):
                ylog[i].append(gv.log((data[tag][t]+data[tag][tp-t])/2))
            if len(set(daughters)) == 1:
                create_t_plot(directory,filename,ylog[0],set(daughters)[0])
            else:
                create_t_plot(directory,filename,ylog[0],set(daughters)[0],ylog[1],set(daughters)[1])
 ######### parents #########
    for mass in Fit['masses']:
        y = collections.OrderedDict()
        ylog = collections.OrderedDict()
        for i,corr in enumerate(set(parents)):
            y[i] = []
            tag = Fit['{0}-Tag'.format(corr)].format(mass)
            filename = '{0}_m{1}'.format(Fit['filename'],mass)
            for t in range(1,int(tp/2+1)):
                y[i].append((data[tag][t]+data[tag][tp-t])/2)
            if len(set(parents)) == 1:
                create_t_plot(directory,filename,y[0],set(parents)[0])
            else:
                create_t_plot(directory,filename,y[0],set(parents)[0],y[1],set(parents)[1])
            ylog[i] = []
            filename = '{0}_m{1}_log'.format(Fit['filename'],mass)
            for t in range(1,int(tp/2+1)):
                ylog[i].append(gv.log((data[tag][t]+data[tag][tp-t])/2))
            if len(set(daughters)) == 1:
                create_t_plot(directory,filename,ylog[0],set(daughters)[0])
            else:
                create_t_plot(directory,filename,ylog[0],set(parents)[0],ylog[1],set(parents)[1])
 ########## currents ###########
    for num,corr in enumerate(currents):
        for mass in Fit['masses']:
            for twist in Fit['twists']:
                ylog = collections.OrderedDict()
                yrat = collections.OrderedDict()
                for i,T in enumerate(Fit['Ts']):
                    ylog[i] = []
                    tag = Fit['threePtTag{0}'.format(corr)].format(T,Fit['m_s'],mass,Fit['m_l'],twist)
                    filename = '{0}_m{1}_tw{2}_log'.format(corr,mass,twist)
                    for t in range(T):
                        ylog[i].append(gv.log(data[tag][i]))
                if len(Fit['Ts']) == 2:
                    create_t_plot(directory,filename,ylog[0],'T={0}'.format(Fit['Ts'][0]),ylog[1],'T={0}'.format(Fit['Ts'][1]))
                if len(Fit['Ts']) == 3:
                    create_t_plot(directory,filename,ylog[0],'T={0}'.format(Fit['Ts'][0]),ylog[1],'T={0}'.format(Fit['Ts'][1]),ylog[2],'T={0}'.format(Fit['Ts'][2]))
                    ###### divide ######
                    yrat[i] = []
                    tagp = Fit['{0}-Tag'.format(parents[num])].format(mass)
                    tagd = tag = Fit['{0}-Tag'.format(daughters[num])].format(twist)
                    filename = '{0}_m{1}_tw{2}_rat'.format(corr,mass,twist)
                    for t in range(T):
                        yrat[i].append(data[tag][i]/(data[tagd][t]*data[tagp][T-t]))
                if len(Fit['Ts']) == 2:
                    create_t_plot(directory,filename,yrat[0],'T={0}'.format(Fit['Ts'][0]),yrat[1],'T={0}'.format(Fit['Ts'][1]))
                if len(Fit['Ts']) == 3:
                    create_t_plot(directory,filename,yrat[0],'T={0}'.format(Fit['Ts'][0]),yrat[1],'T={0}'.format(Fit['Ts'][1]),yrat[2],'T={0}'.format(Fit['Ts'][2]))
    return()

######################################################################################################
plt.figure('log{0}'.format(twist))
            for t in range(1,int(tp/2+1)):                
                plt.errorbar(t, gv.log((data[TwoPts['KGtw{0}'.format(twist)]][t]+data[TwoPts['KGtw{0}'.format(twist)]][tp-t])/2).mean, yerr=gv.log((data[TwoPts['KGtw{0}'.format(twist)]][t]+data[TwoPts['KGtw{0}'.format(twist)]][tp-t])/2).sdev, fmt='ko')

                plt.errorbar(t, ((data[TwoPts['KGtw{0}'.format(twist)]][t]+data[TwoPts['KGtw{0}'.format(twist)]][tp-t])/2).mean, yerr=((data[TwoPts['KGtw{0}'.format(twist)]][t]+data[TwoPts['KGtw{0}'.format(twist)]][tp-t])/2).sdev, fmt='ko')
