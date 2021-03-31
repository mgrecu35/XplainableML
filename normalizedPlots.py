import pickle


import combAlg as cmb
cmb.mainfortpy()
cmb.initp2()

import matplotlib.pyplot as plt

w_cmb=cmb.tablep2.wcj[0:286]
zku_cmb=cmb.tablep2.zkusj[0:286]
zka_cmb=cmb.tablep2.zkasj[0:286]
r_cmb=cmb.tablep2.rj[0:286]
n_cmb=0.08e5
#n_cmb2=0.04e5
import numpy as np

resKa=np.polyfit(zka_cmb-10*np.log10(n_cmb),np.log10(r_cmb/n_cmb),1)
resKu=np.polyfit(zku_cmb-10*np.log10(n_cmb),np.log10(r_cmb/n_cmb),1)

print('\u03B1(Ku)=%7.5f \u03B2(Ku)=%5.3f'%(10**resKu[1],10*resKu[0]))
print('\u03B1(Ka)=%7.5f \u03B2(Ka)=%5.3f'%(10**resKa[1],10*resKa[0]))


d=pickle.load(open('MC3E.pklz','rb'))
w_wrf=d['w']
n_wrf=d['nw']
z_wrf=d['z']
w_z=1
d=pickle.load(open('SO.pklz','rb'))
w_wrf_so=d['w']
n_wrf_so=d['nw']
z_wrf_so=d['z']
#stop
a=range(500000)
b=range(50000,100000)
if w_z==1:
    p1=plt.scatter(0.1*z_wrf[a]-np.log10(n_wrf[a]),w_wrf[a]/n_wrf[a])
    #p1=plt.scatter(0.1*z_wrf[b]-np.log10(0.5*n_wrf[b]),w_wrf[b]/(0.5*n_wrf[b]))
    p1.axes.set_yscale('log')
    p1=plt.scatter(0.1*zku_cmb-np.log10(n_cmb),w_cmb/n_cmb)
    #p1=plt.scatter(0.1*zku_cmb-np.log10(n_cmb2),w_cmb/n_cmb2)
    plt.ylabel('W/N$_W$')
    plt.xlabel('Z$_{Ku}$-10log$_{10}$(N$_W$) (dBZ)')
    plt.legend(['WRF MC3','CMB'])
    plt.savefig('normW_Z_Ku_MC3.png')

b=np.nonzero(abs(0.1*zku_cmb-np.log10(n_cmb)-0.5)<0.01)
bw=np.nonzero(abs(0.1*z_wrf-np.log10(n_wrf)-0.5)<0.01)


plt.figure(figsize=(6,8))
plt.subplot(211)
plt.hist(np.log10(n_wrf),bins=1+np.arange(14)*0.5)
plt.title('WRF MC3E')
plt.subplot(212)
plt.hist(np.log10(n_wrf_so),bins=1+np.arange(14)*0.5)
plt.title('WRF SO')
plt.xlabel('log$_{10}$(N$_W$) [log$_{10}$(mm$^{-1}$m$^{-3}$)]')
plt.savefig('Nw_Distrib.png')
