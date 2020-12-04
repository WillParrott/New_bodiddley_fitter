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
        mean.append(element.mean)
        sdev.append(element.sdev)
    return(mean,sdev)

######################################################################################################


def create_t_plot(location,filename,y1,y1label,y2=None,y2label=None,y3=None,y3label=None,y4=None,y4label=None):
    #Plots x against y where x is a list of floats and y of gvars
    y1mean,y1err = unmake_gvar_vec(y1)
    x1 = np.arange(0,len(y1mean))
    if y2 != None:
        y2mean,y2err = unmake_gvar_vec(y2)
        x2 = np.arange(0,len(y2mean))
    if y3 != None:
        y3mean,y3err = unmake_gvar_vec(y3)
        x3 = np.arange(0,len(y3mean))
    if y4 != None:
        y4mean,y4err = unmake_gvar_vec(y4)
        x4 = np.arange(0,len(y4mean))    
    plt.figure(filename,figsize=((figsca,2*figsca/(1+np.sqrt(5)))))
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.errorbar(x1,y1mean,yerr=y1err,fmt='ko',ms=12,label=y1label)
    if y2 != None:
        plt.errorbar(x2,y2mean,yerr=y2err,fmt='ro',ms=12,label=y2label)
    if y3 != None:
        plt.errorbar(x3,y3mean,yerr=y3err,fmt='bo',ms=12,label=y3label)
    if y4 != None:
        plt.errorbar(x4,y4mean,yerr=y4err,fmt='go',ms=12,label=y4label)
    plt.axes().tick_params(labelright=True,which='both',width=2,labelsize=30)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.xlabel('t',fontsize=30)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles=handles,labels=labels,fontsize=18,frameon=False) 
    #plt.ylabel(r'{0}'.format(ylabel),fontsize=30)
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('{0}/{1}'.format(location,filename))
    return()
######################################################################################################

def plots(Fit,daughters,parents,currents):
    directory = './Plots/{0}'.format(Fit['filename'])
    tp = Fit['tp']
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Creating all plots for {0}".format(Fit['filename']))
        data = make_data('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),Fit['binsize'])
    else:
        print("Plots for {0} already exist".format(Fit['filename']))
        return()
 ######### daughters #########
    for twist in Fit['twists']:
        y = collections.OrderedDict()
        ylog = collections.OrderedDict()
        lab = []
        for i,corr in enumerate(sorted(set(daughters))):
            y[i] = []
            lab.append(corr)
            tag = Fit['{0}-Tag'.format(corr)].format(twist)
            filename = 'daughter_tw{0}.pdf'.format(twist)
            for t in range(1,int(tp/2+1)):
                y[i].append((data[tag][t]+data[tag][tp-t])/2)
            ylog[i] = []     
            for t in range(1,int(tp/2+1)):
                ylog[i].append(gv.log((data[tag][t]+data[tag][tp-t])/2)) 
        if len(set(daughters)) == 1:
            create_t_plot(directory,filename ,y[0],lab[0])
            filename = 'daughter_tw{0}_log.pdf'.format(twist)
            create_t_plot(directory,filename ,ylog[0],lab[0])
        elif len(set(daughters)) == 2:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1])
            filename = 'daughter_tw{0}_log.pdf'.format(twist)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1])
        elif len(set(daughters)) == 3:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1],y[2],lab[2])
            filename = 'daughter_tw{0}_log.pdf'.format(twist)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1],ylog[2],lab[2])
        elif len(set(daughters)) == 4:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1],y[3],lab[3],y[3],lab[3])
            filename = 'daughter_tw{0}_log.pdf'.format(twist)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1],ylog[3],lab[3],ylog[4],lab[4])
 ######### parents #########
    for mass in Fit['masses']:
        y = collections.OrderedDict()
        ylog = collections.OrderedDict()
        lab = []
        for i,corr in enumerate(sorted(set(parents))):
            y[i] = []
            lab.append(corr)
            tag = Fit['{0}-Tag'.format(corr)].format(mass)
            filename = 'parent_m{0}.pdf'.format(mass)
            for t in range(1,int(tp/2+1)):
                y[i].append((data[tag][t]+data[tag][tp-t])/2)
            ylog[i] = []
            
            for t in range(1,int(tp/2+1)):
                ylog[i].append(gv.log((data[tag][t]+data[tag][tp-t])/2))
        if len(set(parents)) == 1:
            create_t_plot(directory,filename,y[0],lab[0])
            filename = 'parent_m{0}_log.pdf'.format(mass)
            create_t_plot(directory,filename,ylog[0],lab[0])
        elif len(set(parents)) == 2:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1])
            filename = 'parent_m{0}_log.pdf'.format(mass)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1])
        elif len(set(parents)) == 3:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1],y[2],lab[2])
            filename = 'parent_m{0}_log.pdf'.format(mass)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1],ylog[2],lab[2])
        elif len(set(parents)) == 4:
            create_t_plot(directory,filename,y[0],lab[0],y[1],lab[1],y[2],lab[2],y[3],lab[3])
            filename = 'parent_m{0}_log.pdf'.format(mass)
            create_t_plot(directory,filename,ylog[0],lab[0],ylog[1],lab[1],ylog[2],lab[2],ylog[3],lab[3])
 ########## currents ###########
    for num,corr in enumerate(currents):
        for mass in Fit['masses']:
            for twist in Fit['twists']:
                ylog = collections.OrderedDict()
                yrat = collections.OrderedDict()
                for i,T in enumerate(Fit['Ts']):
                    ylog[i] = []
                    tag = Fit['threePtTag{0}'.format(corr)].format(T,Fit['m_s'],mass,Fit['m_l'],twist)
                    filename = '{0}_m{1}_tw{2}_log.pdf'.format(corr,mass,twist)
                    for t in range(T):
                        ylog[i].append(gv.log(data[tag][t]))
                    yrat[i] = []
                    tagp = Fit['{0}-Tag'.format(parents[num])].format(mass)
                    tagd = Fit['{0}-Tag'.format(daughters[num])].format(twist)
                    #print(tagd,corr,tagp)
                    for t in range(T):
                        yrat[i].append(data[tag][t]/(data[tagd][t]*data[tagp][T-t]))
                if len(Fit['Ts']) == 2:
                    create_t_plot(directory,filename,ylog[0],'T={0}'.format(Fit['Ts'][0]),ylog[1],'T={0}'.format(Fit['Ts'][1]))
                if len(Fit['Ts']) == 3:
                    create_t_plot(directory,filename,ylog[0],'T={0}'.format(Fit['Ts'][0]),ylog[1],'T={0}'.format(Fit['Ts'][1]),ylog[2],'T={0}'.format(Fit['Ts'][2]))
                if len(Fit['Ts']) == 4:
                    create_t_plot(directory,filename,ylog[0],'T={0}'.format(Fit['Ts'][0]),ylog[1],'T={0}'.format(Fit['Ts'][1]),ylog[2],'T={0}'.format(Fit['Ts'][2]),ylog[3],'T={0}'.format(Fit['Ts'][3]))
                    ###### divide ######
                filename = '{0}_m{1}_tw{2}_rat.pdf'.format(corr,mass,twist)
                if len(Fit['Ts']) == 2:
                    create_t_plot(directory,filename,yrat[0],'T={0}'.format(Fit['Ts'][0]),yrat[1],'T={0}'.format(Fit['Ts'][1]))
                if len(Fit['Ts']) == 3:
                    create_t_plot(directory,filename,yrat[0],'T={0}'.format(Fit['Ts'][0]),yrat[1],'T={0}'.format(Fit['Ts'][1]),yrat[2],'T={0}'.format(Fit['Ts'][2]))
                if len(Fit['Ts']) == 4:
                    create_t_plot(directory,filename,yrat[0],'T={0}'.format(Fit['Ts'][0]),yrat[1],'T={0}'.format(Fit['Ts'][1]),yrat[2],'T={0}'.format(Fit['Ts'][2]),yrat[3],'T={0}'.format(Fit['Ts'][3]))
    return()

######################################################################################################
 
