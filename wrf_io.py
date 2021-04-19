from netCDF4 import Dataset

def read_wrf(fname,it):
    f=Dataset(fname)
    qv=f['QVAPOR'][it,:,:,:]    # water vapor
    qr=f['QRAIN'][it,:,:,:]     # rain mixing ratio
    qs=f['QSNOW'][it,:,:,:]     # snow mixing ratio
    qc=f['QCLOUD'][it,:,:,:]    # cloud mixing ratio
    qg=f['QGRAUP'][it,:,:,:]   # graupel mixing ratio
    ncr=f['QNRAIN'][it,:,:,:]     # rain mixing ratio
    ncs=f['QNSNOW'][it,:,:,:]     # snow mixing ratio
    ncg=f['QNGRAUPEL'][it,:,:,:]   # graupel mixing ratio
    #z=f['z_coords'][:]/1000.             # height (km)
    th=f['T'][it,:,:,:]+300    # potential temperature (K)
    prs=f['P'][it,:,:,:]+f['PB'][it,:,:,:]  # pressure (Pa)
    T=th*(prs/100000)**0.286  # Temperature
    t2c=T-273.15
    #stop
    z=(f['PHB'][it,:,:,:]+f['PH'][it,:,:,:])/9.81/1000.
    xlat=f['XLAT'][0,:,:]
    xlong=f['XLONG'][0,:,:]
    R=287.058  #J*kg-1*K-1
    rho=prs/(R*T)
    tsk=f['TSK'][it,:,:]
    u10=f['U10'][it,:,:]
    v10=f['V10'][it,:,:]
    w_10=(u10**2+v10**2)**(0.5)
    return qr,qs,qg,ncr,ncs,ncg,qc,rho,z,T,tsk, w_10, qv, prs, z
