from readLinData import *
from   numpy import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator,LogFormatter
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LogNorm
import pickle

import glob
fCRSL=sorted(glob.glob('IPHEX/*CRS*'))
fKaL=sorted(glob.glob('IPHEX/*HKa*'))
fKuL=sorted(glob.glob('IPHEX/*HKu*'))

cfadz1=zeros((70,90),float)
cfadz2=zeros((70,90),float)
cfadz3=zeros((70,90),float)

def incCfad(z1L,h1,h0,k,cfad1):
    if z1L[k]>-5:
        i0=int((h1[k]-h0+1200)/125.)
        j0=int(z1L[k]/0.5)
        if i0>=0 and i0<70 and j0>=0 and j0<90:
            cfad1[i0,j0]+=1

h0=1000.

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    coords = [ix, iy]
    print(coords)
    if iy>4:
        fig.canvas.mpl_disconnect(cid)
    hbbL.append(coords)
    return coords
import pickle
igetHBB=0


if1=0
zKuRL=[]
zKaRL=[]
z1L=[]
xL=[]



fs1=['Lin/iphex_comb_radar2014503_185703-190442.nc',\
     'Lin/iphex_comb_radar2014503_190645-192513.nc',\
     'Lin/iphex_comb_radar2014512_125511-130241.nc',\
     'Lin/iphex_comb_radar2014515_140811-142600.nc',
     'Lin/iphex_comb_radar2014515_142919-144814.nc,'\
     'Lin/iphex_comb_radar2014516_132703-133315.nc',
     'Lin/iphex_comb_radar2014523_222426-223201.nc',
     'Lin/iphex_comb_radar2014524_013152-014227.nc',
     'Lin/iphex_comb_radar2014524_014641-015700.nc',
     'Lin/iphex_comb_radar2014529_210445-210937.nc']

from netCDF4 import Dataset

fs=glob.glob('Lin/iphex_co*nc')
fs=sorted(fs)
ind=[490,100,900,1700,250,290,700,1100,900,400]
ip=0
dBaseL=[]
zXL=[]
zKuL=[]
zWL=[]
iplot=1
fnameL=[]
infoL=[]
zku1L=[]
zka1L=[]
zw1L=[]
zx1L=[]
iplot=0
for f in fs[:]:
    if 'freq' in f:
        continue
    fh=Dataset(f)
    zku=fh['zku'][:,:]
    zx=fh['zx'][:,:]
    zka=fh['zka'][:,:]
    zw=fh['zw'][:,:]
    h=fh['altitude'][:]
    dist=fh['dist'][:]
    r=fh['range'][:]
    t=fh['timed'][:]
    lat=fh['lat'][:]
    lon=fh['lon'][:]
    zkum=ma.array(zku,mask=zku<-10)
    zxm=ma.array(zx,mask=zx<-10)
    zkam=ma.array(zka,mask=zka<-10)
    zwm=ma.array(zw,mask=zw<-10)
    a=nonzero(zkum[:,120:400].sum(axis=1)>1000)
    for i in a[0]:
        dBaseL.append(zka[i,80:410:3])
        zXL.append(zx[i,80:410:3])
        zKuL.append(zku[i,80:410:3])
        zWL.append(zw[i,80:410:3])
        zx1L.append(zx[i,20:440:3])
        zku1L.append(zku[i,20:440:3])
        zka1L.append(zka[i,20:440:3])
        zw1L.append(zw[i,20:440:3])
        fnameL.append(f[20:-3])
        infoL.append([t[i],lat[i],lon[i]])
    if iplot==1:
        plt.figure()
        plt.subplot(311)
        n=zxm.shape[0]
        d=arange(n)
        plt.pcolormesh(d,20-r[120:400],zkum[:,120:400].T,vmin=0,vmax=50,cmap='jet')
        plt.subplot(312)
        plt.pcolormesh(d,20-r,zkam.T,vmin=0,vmax=50,cmap='jet')
        plt.subplot(313)
        plt.pcolormesh(d,20-r,zwm.T,vmin=0,vmax=50,cmap='jet')
        stop
        #plt.figure()
    #plt.plot(zxm[ind[ip],:],20-r)
    #plt.plot(zkum[ind[ip],:],20-r)
    #plt.plot(zkam[ind[ip],:],20-r)
    #plt.plot(zwm[ind[ip],:],20-r)
    #plt.xlim(-10,50)
    ip+=1
    #    print(f)
    #plt.show()


from sklearn.cluster import MiniBatchKMeans
nc=70
h=20-r[80:410:3]
kmeans=MiniBatchKMeans(n_clusters=nc,batch_size=5000,random_state=0)
dBaseL=array(dBaseL)
dBaseL[dBaseL<-30]=-30
zXL=array(zXL)
zXL[zXL<-30]=-30
zKuL=array(zKuL)
zKuL[zKuL<-30]=-30
zWL=array(zWL)
zWL[zWL<-30]=-30
kmeans.fit(dBaseL[:,:-10])
#classes=[3,7,23,26,30,32,45,61,63,64,69]
classes=[4,8,11,12,22,23,25,36,40,53]
classes=[5,8,19,29,34,41,67,58]
import matplotlib

matplotlib.rcParams.update({'font.size': 13})
ic=1
nL=dBaseL.shape[0]
f1=0
fhtml=open('cases.html','w')
fhtml.write('<html>')

from geopy.distance import geodesic,  GeodesicDistance, Point
import math as Math
import math
def getAzimuth(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = Math.sin(dLon) * Math.cos(lat2);
    x = Math.cos(lat1)*Math.sin(lat2)-\
        Math.sin(lat1)*Math.cos(lat2)*Math.cos(dLon);
    brng = Math.atan2(y, x)/pi*180.;
    return brng

def calculate_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
for i in range(nc):
    if i+1 in classes:
        fhtml2=open('zProfiles%2.2i.html'%ic,'w')
        a=nonzero(kmeans.labels_==i)
        plt.figure()
        zc=dBaseL[a[0],:].mean(axis=0)
        zerr=dBaseL[a[0],:].std(axis=0)
        plt.errorbar(zc,h,xerr=zerr)
        #print(fnameL[a[0][0]],fnameL[a[0][-1]],i+1)
        listOfFiles=[]
        timeLoc=[]
        nlat=35.19583333
        nlon=-81.96305556
        npol=(nlat,nlon)
        dL=[]
        zxwL=[]
        zkuwL=[]
        zkawL=[]
        zwwL=[]
        for i1 in a[0]:
            d=geodesic(npol,infoL[i1][1:])
            if d.km<100:
                listOfFiles.append(fnameL[i1])
                timeLoc.append(infoL[i1])
                dL.append(d.km)
                zxwL.append(zx1L[i1])
                zkuwL.append(zku1L[i1])
                zkawL.append(zka1L[i1])
                zwwL.append(zw1L[i1])
        for f11,loc1,d,zx1,zku1,zka1,zw1 in zip(listOfFiles,timeLoc,dL,zxwL,zkuwL,zkawL,zwwL):
            print(f11,loc1[0],loc1[1],loc1[2])
            hh=int(loc1[0])
            mm=int((loc1[0]-hh)*60)
            brng=getAzimuth(nlat,nlon,loc1[1],loc1[2])
            brng=calculate_bearing((nlat,nlon), (loc1[1],loc1[2]))
            for it in range(3):
                p1 = geodesic(kilometers=d).destination(Point(nlat, nlon), brng)
                error=((p1.longitude-loc1[2])**2+(p1.latitude-loc1[1])**2)**.5
                p = geodesic(kilometers=d).destination(Point(nlat, nlon), brng+0.1)
                error1=((p.longitude-loc1[2])**2+(p.latitude-loc1[1])**2)**.5
                brng-=(error1-error)/0.1
            #top
            #p.
            #stop
            fmt="%s Time=%2.2i:%2.2i lat=%6.2f lon=%6.2f distance_to_NPOL=%5.2f km az=%6.2f<br>\n"
            fhtml.write(fmt%(f11[:7],hh,mm,loc1[1],loc1[2],d,brng))
            s='X-band '
            for ib in range(140):
                s=s+'%6.2f '%zx1[ib]
            fhtml2.write('%s <br>\n'%s)
            s='Ku-band '
            for ib in range(140):
                s=s+'%6.2f '%zku1[ib]
            fhtml2.write('%s <br>\n'%s)
            s='Ka-band '
            for ib in range(140):
                s=s+'%6.2f '%zka1[ib]
            fhtml2.write('%s <br>\n'%s)
            s='W-band '
            for ib in range(140):
                s=s+'%6.2f '%zw1[ib]
            fhtml2.write('%s <br>\n'%s)
            
        print('<img src=\"class_2%2.2i.png\">'%ic)
        fhtml.write('<img src=\"class_2%2.2i.png\">\n <br>'%ic)
        plt.plot(zXL[a[0],:].mean(axis=0),h)
        plt.plot(zKuL[a[0],:].mean(axis=0),h)
        plt.plot(zWL[a[0],:].mean(axis=0),h)
        plt.legend(['X','Ku','W','Ka'])
        plt.xlabel('dBZ')
        plt.ylabel('Height (km)')
        plt.title("Convective Class %2.2i \n freq=%6.2f%%"%(ic,len(a[0])*100./nL))
        plt.xlim(-25,55)
        f1+=len(a[0])*100./nL
        
        plt.savefig('class_2%2.2i.png'%ic)
        fhtml2.close()
        ic+=1
fhtml.write('</html>')
fhtml.close()
import pickle
dp={}
dp["kmeans"]=kmeans
dp["zX"]=zXL
dp["zKu"]=zKuL
dp["zW"]=zWL
dp["zKa"]=dBaseL
dp["h"]=h
pickle.dump(dp,open('kmeansPROFs.plkz','wb'))
