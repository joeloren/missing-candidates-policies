import glob
import sys
import numpy as np
import cPickle as cp
import math
import matplotlib.pyplot as plt
import dpopt
import re

m=10
n=100
populations=25

loc='../results/'

thick = 6

if False:
    for vr in [0, 1, 2]:
        #for dataF in ['wf','wf-same','wf-mass']:
        dataF = 'wf'
        plt.figure()
        plt.axis([0.0, 1.0, 0.0, 1.05])
        plt.axhspan( 0.0, 1.05,0.0, 1.0,linewidth=thick,fill=False)
        for i, (phi,s) in enumerate([(0.3,'-.'), (0.8,'--'), (1.0,'-')]):
        #for i, (phi,s) in enumerate([(1.0,'-')]):
            # data-m=10-n=100-disp=0.3-popsize=25-vr=0-ap=0.9-method=0-wf.dat
            data = 'data-m='+str(m)+'-n='+str(n)+'-disp='+str(phi)+'-popsize='+str(populations)+'-vr='+str(vr)+'-ap=0.9-method=0-'+dataF+'.dat'
            f = open(data,'r')
            (xv,yv) = cp.load(f)
            f.close()
            plt.plot(xv,yv,linewidth=thick,ls=s)
        #plt.legend( (r'$\phi=1.0$',), loc='lower right')
        plt.legend( (r'$\phi=0.3$',r'$\phi=0.8$',r'$\phi=1.0$'), loc='lower right',prop={'size':24})
        plt.text(0.7,0.45,dpopt.rulesDict[vr][1],fontsize=24)
        plt.xlabel('p',fontsize=24)
        plt.tick_params(axis='x', labelsize=19)
        plt.tick_params(axis='y', labelsize=19)
        #plt.ylabel('probabily of available naive winner being robust',fontsize=18)
        plt.savefig(loc+dataF+'-vr-'+str(vr)+'.pdf', format='pdf')

def min_max_average(values):
    mmin = values[-1]
    mmax = values[-1]
    msum = values[-1]*1.0
    for i in range(len(values)-1):
        mmin = min(values[i],mmin)
        mmax = max(values[i],mmax)
        msum += values[i]
    return (mmin,mmax,msum/len(values))

query_t = {}
def save_table_query(met,vr,phi,ap,data):
    global query_t
    query_t[(met,vr,phi,ap)]  =  min_max_average(data)

size_t = {}
def save_table_size(met,vr,phi,ap,data):
    global size_t
    size_t[(met,vr,phi,ap)]  =  min_max_average(data)

vr2str = {0:'Plur', 1: 'Borda',2:'Cope'}
met2str = {0:'DP', 1: 'IG', 2: 'Rnd'}
def save_table(f,t,prec,prec2):
    form = '& %.'+str(prec)+'f (%.'+str(prec2)+'f,%.'+str(prec2)+'f)'
    for met in [0, 1, 2]:
        for vr in [0, 1, 2]:
            # Plur-DP 
            print >>f, vr2str[vr]+'-'+met2str[met]
            for phi in [0.3, 0.8, 1.0]:
                for ap in [0.3, 0.5, 0.9]:
                    (mmin,mmax,mavg) = t[(met,vr,phi,ap)]
                    print >>f, form  % (mavg,mmin,mmax)
            print >>f, '\\\\'

fs=15
if True:
    for met in [0, 1, 2]:
        for vr in [0, 1, 2]:
            plt.figure()
            plt.axis([0, 10, -0.5, 9])
            # Boxplots + tablas
            data = []
            for phi in [0.3, 0.8, 1.0]:
                for ap in [0.3, 0.5, 0.9]:
                    # data-m=10-n=100-disp=0.3-popsize=25-vr=0-ap=0.9-method=0.dat
                    dataf = 'data-m='+str(m)+'-n='+str(n)+'-disp='+str(phi)+'-popsize='+str(populations)+'-vr='+str(vr)+'-ap='+str(ap)+'-method='+str(met)+'.dat'
                    f = open(dataf,'r')
                    d = cp.load(f)
                    data.append(d)
                    f.close()

                    save_table_query(met,vr,phi,ap,d)

            for pos,phi in [(2,0.3), (5,0.8), (8,1.0)]:
                plt.text(pos,-0.2, (r'$\phi='+str(phi)+'$'),horizontalalignment='center',fontsize=fs)
                for pos2,p in [(pos-1,0.3), (pos,0.5), (pos+1,0.9)]:
                    plt.text(pos2,0.4, (r'$p='+str(p)+'$'),horizontalalignment='center',fontsize=fs)
                
            plt.text(1,8,dpopt.rulesDict[vr][1] + ' - ' + dpopt.algsDict[met][1],fontsize=fs)
            plt.ylabel('Expected number of queries',fontsize=fs)
            #plt.xlabel('For same $\phi$, $ap \in \{ 0.3, 0.5, 0.9 \}$')
            plt.boxplot(data)
            plt.xticks([])
            plt.tick_params(axis='x', labelsize=fs+1)
            plt.tick_params(axis='y', labelsize=fs+1)
            pf = 'box-vr-'+str(vr)+'-met-'+str(met)+'.pdf'
            plt.savefig(loc+pf, format='pdf')
            #print '----------- PRINTED --------------',pf
    query_f = file(loc+'query.tex','w')
    print >> query_f, '% Query.tex'
    save_table(query_f,query_t,1,1)
    query_f.close()


    # TODO: change readability
if False:
    str2data = {}
    for namef in glob.glob("data*.dot"):
        name = re.findall('([^&]*)\-p[0-9]+.dot',namef)[0]
        nqueries = 0
        for l in file(namef):
            if 'label' in l and 'label=""' not in l:
                # Is a query
                nqueries += 1
        if name in str2data:
            str2data[name].append(nqueries)
        else:
            str2data[name] = [nqueries]

    for met in [0, 1, 2]:
        for vr in [0, 1, 2]:
            plt.figure()
            plt.axis([0, 10, -55, 420])
            # Boxplots + tablas
            data = []
            for phi in [0.3, 0.8, 1.0]:
                for ap in [0.3, 0.5, 0.9]:
                    # data-m=10-n=100-disp=1.0-popsize=25-vr=2-ap=0.9-method=1
                    datal = 'data-m='+str(m)+'-n='+str(n)+'-disp='+str(phi)+'-popsize='+str(populations)+'-vr='+str(vr)+'-ap='+str(ap)+'-method='+str(met)
                    data.append(str2data[datal])
                    save_table_size(met,vr,phi,ap,str2data[datal])

            for pos,phi in [(2,0.3), (5,0.8), (8,1.0)]:
                plt.text(pos,-45, (r'$\phi='+str(phi)+'$'),horizontalalignment='center')
                for pos2,p in [(pos-1,0.3), (pos,0.5), (pos+1,0.9)]:
                    plt.text(pos2,-15, (r'$p='+str(p)+'$'),horizontalalignment='center')
            plt.text(1,325,dpopt.rulesDict[vr][1] + ' - ' + dpopt.algsDict[met][1])
            #plt.xlabel('Total queries. '+dpopt.rulesDict[vr][1] + ' - ' + dpopt.algsDict[met][1]+ r'. For same $\phi$, $ap \in \{ 0.3, 0.5, 0.9 \}$')
            plt.ylabel('Total number of queries')
            plt.boxplot(data)
            plt.xticks([])
            pf = 'box-nodes-vr-'+str(vr)+'-met-'+str(met)+'.pdf'
            plt.xticks([])
            plt.savefig(loc+pf, format='pdf')
            # print '----------- PRINTED nq --------------',pf
    size_f = file(loc+'size.tex','w')
    print >> size_f, '% Size.tex'
    save_table(size_f,size_t,1,0)
    size_f.close()
    

