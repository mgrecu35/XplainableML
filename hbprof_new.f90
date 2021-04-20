module tableP2
  real :: zKuSJ(300)
  real :: zKudN(300)
  real :: zKaSJ(300)
  real :: dNT(300)
  real :: dmJ(300)
  real :: rJ(300)
  real :: wcJ(300)
  real :: logRJ(300)
  real :: attKaJ(300)
  real :: attKuJ(300)
  integer :: nJ
end module tableP2
subroutine initP2
  use tableP2
  use tables2
  implicit none
  integer :: i
  real :: f
  f=1
  do i=1,nbins
     zKuSJ(i)=zmin+(i-1)*dzbin
     zKaSJ(i)=z35Table(i,1)
     dmJ(i)=d013Table(i,1)
     attKaJ(i)=att35Table(i,1)
     attKuJ(i)=att13Table(i,1)
     rJ(i)=pr13Table(i,1)
     wcJ(i)=10**pwc13Table(i,1)
     logRJ(i)=log10(pr13Table(i,1))
     if(dmj(i).lt.0.8) then
        dnT(i)=1.5*log((0.8/dmj(i))**0.25)+0.0
     else
        dnT(i)=1.0*log((0.8/dmj(i))**0.25)+0.0
     end if
     zKudN(i)=zKuSJ(i)+10*dnT(i)
  end do
  !print*,f
  !stop
  nJ=nbins
  !print*, d013Table(200:240,1),nbins,nbinS2,nbinH
end subroutine initP2

subroutine prof1d(btop,bzd,bcf,bsfc,binBB,binBBT,zKuL,zKaL,pType,dr,n1d,eps,imu,&
     dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dnCoeff_new,dn,dsrtPIA,relSRT,srtPIAKu,relSRTPIAKu,&
     dt1,dt2)
  implicit none
  real :: dnp(n1d), dn
  integer :: btop, bzd, bcf, bsfc, n1d, imu, bb,bbt,bbb, pType
  real :: zKuL(n1d), dr, zKaL(n1d)
  real :: dnCoeff_new(2)
  real,intent(out)::dt1,dt2
  integer :: binBB,binBBT
  real,intent(out) :: dm1d(n1d)
  real,intent(out) :: epst,piaKu,piaKa
  real,intent(out) :: dn1d(n1d), rrate1d(n1d), zKuC(n1d), zKaSim(n1d)
  real :: dpia2dn(2,n1d), dzdn(n1d,n1d),dpia2dn1(2,n1d), dzdn1(n1d,n1d)
  real :: eps
  real :: dZKa(n1d), dPIA
  real :: dn1d1(n1d),dm1d1(n1d),rrate1d1(n1d),zKuC1(n1d),zKaSim1(n1d),epst1,piaKu1,piaKa1
  real :: ddn, gradZS, gradZe
  integer :: i, nbz, relSRT
  real :: sigma, dnbz(n1d), dsrtPIA,   dpiadn
  real :: srtPIAKu
  integer :: relSRTPIAKu
  real :: piaKuS,piaKaS,piaKuS1,piaKaS1
  !print*, bbb, binBB

  dnp=0
  if (pType.eq.2) then
     !print*, pType, dnCoeff_new,dn
     call iter_profcv(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
          dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,ptype,dnCoeff_new,dn,dnp,dzdn,srtPIAKu,relSRTPIAKu,&
          dt1,dt2)
     !print*, rrate1d(bcf+1), epst
  else
     if (binBB.lt.1) then
        call iter_profst_nobb(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,ptype,&
             dnCoeff_new,dn,dnp,dzdn,&
             dt1,dt2,dpia2dn,piaKuS,piaKaS)
        call iter_profst_nobb(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d1,dm1d1,rrate1d1,zKuC1,zKaSim1,epst1,piaKu1,piaKa1,ptype,&
             dnCoeff_new,dn-0.1,dnp,dzdn1,&
             dt1,dt2,dpia2dn1,piaKuS1,piaKaS1)
        !print*, piaKu,piaKu1, piaKa, piaKa1, piaKa-piaKu, dsrtPIA
        bbb=bzd+3
     else
        bbb=binBB+2
        bbt=binBBT
        bb=binBB        
        call iter_profst(btop,bzd,bb,bbt,bbb,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dn,dnCoeff_new,&
             dnp,dzdn,dpia2dn,piaKuS,piaKaS)
        call iter_profst(btop,bzd,bb,bbt,bbb,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d1,dm1d1,rrate1d1,zKuC1,zKaSim1,epst1,piaKu1,piaKa1,dn-0.1,&
             dnCoeff_new,dnp,dzdn1,dpia2dn1,piaKuS1,piaKaS1)
     end if
     gradZS=1
     gradZe=0.
     do i=bbb+3,bcf
        if(zKuL(i+1).gt.10.0.and.zKaL(i+1).gt.10) then
           gradZS=gradZS+0.125*(-(zKaSim1(i+1)-zKaSim(i+1))/0.1)**2
           gradZe=gradZe+(-(zKaSim1(i+1)-zKaSim(i+1))/0.1)*0.125*(zKaL(i+1)-zKaSim(i+1))
        endif
     enddo
     if(relSRT.eq.1) then
        dpiadn=(piaKa1-piaKu1-piaKa+piaKu)/(-0.1)
        gradZS=gradZS+dpiadn**2
        gradZe=gradZe-dpiadn*(piaKa-piaKu-dsrtPIA)
        !print*, 'dpiadn',piaKa-piaKu, dsrtPIA, dpiadn, piaKu, piaKa, binBBT
     end if

     if(bbb.lt.bcf-3) then
        sigma=1.5
        nbz=bcf-bbb
        call gauss_newton(dzdn(bbb+2:bcf+1,bbb+2:bcf+1), zKasim(bbb+2:bcf+1), zKaL(bbb+2:bcf+1), nbz, sigma,&
             dnbz)
        dnp(bbb+2:bcf+1)=0.005*dnbz(1:nbz)
     end if
     do i=bbb+2,bcf
        if(dnp(i+1).gt.1) dnp(i+1)=1
        if(dnp(i+1).lt.-1) dnp(i+1)=-1
     enddo
     !dy=transpose(dZdn)*(dY)-0.1*invCov*(xsol-xMean);
     !A=transpose(dZdn)*dZdn+0.1*invCov;
     !dn=A\dy;
     if(relSRT.eq.1) then
        ddn=0.95*gradZe/gradZS
     else
        ddn=0
     end if
     !ddn=0.0
     if(ddn.lt.-2.5) ddn=-2.5
     if(ddn.gt.2.5) ddn=2.5
     !dnp=0.0
     !print*, ddn, bbb, bcf
     if (binBB.lt.1) then
        call iter_profst_nobb(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,ptype,&
             dnCoeff_new,dn+ddn,dnp,dzdn,&
             dt1,dt2,dpia2dn,piaKuS,piaKaS)
     else
        bbb=binBB+2
        bbt=binBBT
        bb=binBB
        !dnp=0
        call iter_profst(btop,bzd,bb,bbt,bbb,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dn+ddn,&
             dnCoeff_new,dnp,dzdn,dpia2dn,piaKuS,piaKaS)
     end if
     
  end if
end subroutine prof1d

subroutine prof1d_noad(btop,bzd,bcf,bsfc,binBB,binBBT,zKuL,zKaL,pType,dr,n1d,eps,imu,&
     dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dnCoeff_new,dn,dsrtPIA,relSRT,srtPIAKu,relSRTPIAKu,&
     dt1,dt2)
  implicit none
  real :: dnp(n1d), dn
  integer :: btop, bzd, bcf, bsfc, n1d, imu, bb,bbt,bbb, pType
  real :: zKuL(n1d), dr, zKaL(n1d)
  real :: dnCoeff_new(2)
  real,intent(out)::dt1,dt2
  integer :: binBB,binBBT
  real,intent(out) :: dm1d(n1d)
  real,intent(out) :: epst,piaKu,piaKa
  real,intent(out) :: dn1d(n1d), rrate1d(n1d), zKuC(n1d), zKaSim(n1d)
  real :: dpia2dn(2,n1d), dzdn(n1d,n1d),dpia2dn1(2,n1d), dzdn1(n1d,n1d)
  real :: eps
  real :: dZKa(n1d), dPIA
  real :: dn1d1(n1d),dm1d1(n1d),rrate1d1(n1d),zKuC1(n1d),zKaSim1(n1d),epst1,piaKu1,piaKa1
  real :: ddn, gradZS, gradZe
  integer :: i, nbz, relSRT
  real :: sigma, dnbz(n1d), dsrtPIA,   dpiadn
  real :: srtPIAKu
  integer :: relSRTPIAKu
  real :: piaKuS,piaKaS
  !print*, bbb, binBB

  dnp=0
  if (pType.eq.2) then
     !print*, pType, dnCoeff_new,dn
     call iter_profcv(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
          dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,ptype,dnCoeff_new,dn,dnp,dzdn,srtPIAKu,relSRTPIAKu,&
          dt1,dt2)
     !print*, rrate1d(bcf+1), epst
  else
     if (binBB.lt.1) then
        call iter_profst_nobb(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,ptype,dnCoeff_new,dn,dnp,dzdn,&
             dt1,dt2,dpia2dn,piaKuS,piaKaS)
     else
        bbb=binBB+2
        bbt=binBBT
        bb=binBB        
        call iter_profst(btop,bzd,bb,bbt,bbb,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
             dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dn,dnCoeff_new,dnp,dzdn,dpia2dn,piaKuS,piaKaS)
     end if
  end if
end subroutine prof1d_noad

subroutine iter_profcv(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
     dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,itype,dnCoeff_new,dncv,dnp,dzdn,piaSRTKu,relPIASRTKu,&
     dt1,dt2)
  use tables2
  use tableP2
  use ran_mod
  implicit none
  integer :: btop, bzd, bcf, bsfc, n1d, imu
  real :: zKuL(n1d), dr, zKaL(n1d), dnp(n1d)
  !real :: dmCoeff(2)=(/0.027773772993636318,-0.6624076086959071/)
  real,intent(out) :: dm1d(n1d)
  real,intent(out) :: epst,piaKu,piaKa
  real,intent(out) :: dn1d(n1d), rrate1d(n1d), zKuC(n1d), zKaSim(n1d), dzdn(n1d,n1d)
  real :: dnCoeff_new(2)
  real :: dzKa(n1d), dns(n1d)
  real :: eps, dm, pia, dncv
  integer :: it, ik, k, n1, n1H, itype
  real :: ztrueS,ztrueH,ztrue,f, attKuH, attKaH, attKuS, attKaS
  real :: ftran, zetaS(31), probs(31), sumprob,  zeta1d(n1d,31), q
  real :: rrate1d_sub(n1d,31), dn_sub(n1d,31), dm_sub(n1d,31), zkuc_sub(n1d,31), piahb_sub(31)
  real :: piaKuS, piaKaS, beta, piamax, attKu, attKa, dn, dni
  real :: dnCoeff(2)=(/-0.00570364,  0.13319214/)
  real :: dmCoeff(3)= (/-2.56885571e-04,  7.18909743e-02, -1.60889523e+00/)
  !(/-4.57599678e-04,  7.852247958e-02, -1.78316499e+00/)
  !(/-4.57599678e-04,  8.52247958e-02, -1.78316499e+00/)
  real :: rn, dm_old!, dm_sub(n1d,31)
  real :: zka1,zka2,attka2
  real :: dt1, dt2, start, finish1, finish2
  integer :: kk, imc, isub
  real :: dmm
  real :: piaSRTKu, dzdn_sub2(n1d,n1d,31), dzdn_sub1(n1d,n1d,31), zkasim_sub(n1d,31), piaKa_sub(31)
  integer :: relPIASRTKu
  real :: ddn
  piaKu=0
  !dmCoeff=array([ 0.02893781, -0.69481455])
  dzdn=0.0
  !if(itype.eq.1) then
  dnCoeff=dnCoeff_new
  !end if
  zKuC=zKuL
  dns=0
  dZKa=0
  rrate1d=0
  call cpu_time(start)
  do it=1,2
     piaKu=0.0
     piaKa=0.0
     do k=btop,min(bcf,bzd-1)
        if (zKuC(k+1).gt.10) then
           if (k.lt.bzd) then
              ztrue=zKuC(k+1)
           end if
           if(ztrue.lt.40) then
              ztrueS=ztrue
              ztrueH=0.0
           else
              ztrueS=40
              ztrueH=10*log10(10**(0.1*ztrue)-10**(0.1*39.999))
           endif
           !n1,n2=bisection(zKuSJ,ztrueS-10*dns[k+1])
           call bisection2(zKuSJ,nbinS2,ztrueS-10*dns(k+1), n1)
           attKuS=att13TableS2(n1,imu)*10**dns(k+1)
           attKaS=att35TableS2(n1,imu)*10**dns(k+1)
           !n1H,n2H=bisection(zKuHJ,ztrueH)
           call bisection2(zKuSJ,nbinS2,ztrueH, n1H)
           attKuH=att13TableH(n1H,imu)
           attKaH=att35TableH(n1H,imu)
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           zKuC(k+1)=zKuL(k+1)+piaKu
           zKaSim(k+1)=10*log10(10**(0.1*z35TableS2(n1,imu)+dns(k+1))+&
                10**(0.1*z35TableH(n1H,imu)))-piaKa
           if(zKaL(k+1).gt.10) then
              dZKa(k+1)=zKaSim(k+1)-zKaL(k+1)
           end if
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           dm1d(k+1)=0.5*(d013TableS2(n1,imu)+d013TableH(n1H,imu))
           dn1d(k+1)=dns(k+1)
           rrate1d(k+1)=pr13TableS2(n1,imu)*10**dns(k+1)+pr13TableH(n1H,imu)
        end if
     end do
     if(it.eq.1) then
        do k=btop,min(bcf,bzd-1)
           dns(k+1)=dns(k+1)-0.6*dZKa(k+1)
           if(dns(k+1).gt.0.975) dns(k+1)=0.975
           if(dns(k+1).lt.-1.0) dns(k+1)=-1.0
        end do
     end if
  end do
  
  piaKuS=piaKu+0.0
  piaKaS=piaKa+0.0

  zeta1d=0
  beta=0.71
  
  piamax=54-zKuL(bcf+1)
  if(piamax<0) piamax=56-zKuL(bcf+1)
  if(piamax<0) piamax=4
  q=0.2*log(10.0)
  zKuC(bzd+1:bcf+1)=zKuL(bzd+1:bcf+1)+piaKuS
  epst=eps
  attKu=0.0
  attKa=0.0
  f=1.0
  zeta1d=0
  sumprob=0
  piaKu=0.0
  rrate1d(bzd+1:bcf+1)=0.0
  dm1d(bzd+1:bcf+1)=0.0
  dn1d(bzd+1:bcf+1)=0.0
  rrate1d_sub=0
  dm_sub=0
  dn_sub=0
  zkasim_sub=0
  dzdn_sub2=0
  dzdn_sub1=0
  zkasim(bzd+1:bcf+1)=0
  dzdn(bzd+1:bcf+1,bzd+1:bcf+1)=0
  call cpu_time(finish1)
  ddn=0.025
!!$OMP PARALLEL DO  PRIVATE(k,dn,n1,attKu,attKa,zKa1,zKa2)
  do isub=1,31,3
     zetaS(isub)=0
     do k=bzd,bcf
        if (zKuL(k+1).gt.10) then
           call bisection2(zKudN,nbins,zKuL(k+1)+piaKuS-10*(isub-16)*ddn-10*dncv, n1)
           dn_sub(k+1,isub)=(zKudN(n1)-zKuSJ(n1))/10.0+(isub-16)*ddn+dncv
           dn=dn_sub(k+1,isub)
           attKu=att13Table(n1,imu)*10**dn
           attKa=att35Table(n1,imu)*10**dn
           zetaS(isub)=zetaS(isub)+attKu*dr
           zeta1d(k+1,isub)=zetaS(isub)
        else
           attKu=0.0
           attKa=0.0
        endif
     end do
     if (q*beta*zetaS(isub).lt.0.985) then
        piaKa_sub(isub)=piaKas
        piaHB_sub(isub)=-10/beta*log10(1-q*beta*zetaS(isub))+attKu*(bsfc-bcf)*2*dr
        do k=bzd,bcf
           if(zKuL(k+1).gt.10) then
              zKuC_sub(k+1,isub)=zKuL(k+1)-10/beta*log10(1-q*beta*zeta1d(k+1,isub))
              n1=int((zKuC_sub(k+1,isub)-zmin-10*dn_sub(k+1,isub))/dzbin)
              dn=dn_sub(k+1,isub)
              attKa=att35Table(n1,imu)*10**dn
              piaKa_sub(isub)=piaKa_sub(isub)+attKa*dr
              if(n1.lt.1) n1=1
              if(n1.gt.nbins) n1=nbins
              dm_sub(k+1,isub)=d013Table(n1,imu)
              rrate1d_sub(k+1,isub)=pr13Table(n1,imu)*10**dn_sub(k+1,isub)
              zKa1=z35Table(n1,imu)+10*dn
              zkasim_sub(k+1,isub)= zKa1-piaKa_sub(isub)
              piaKa_sub(isub)=piaKa_sub(isub)+attKa*dr
              if(n1.lt.nbins-4) then
                 zKa2=z35Table(n1+4,imu)+10*(dn-0.1)
                 attKa2=att35Table(n1+4,imu)*10**(dn-0.1)
                 dzdn_sub2(k+1,k+1,isub)=dzdn_sub2(k+1,k+1,isub)+(zka2-attKa2*dr)-((zka1-attKa*dr))
                 dzdn_sub1(k+1,k+1,isub)=dzdn_sub1(k+1,k+1,isub)+zka1-attKa*dr
                 do kk=k+1,bcf
                    dzdn_sub2(kk+1,k+1,isub)=dzdn_sub2(kk+1,k+1,isub)-(attKa2*dr-attKa*dr)
                    dzdn_sub1(kk+1,k+1,isub)=dzdn_sub1(kk+1,k+1,isub)-attKa*dr
                 end do
              end if
           end if
        end do
        probs(isub)=exp(-((isub-16)*ddn)**2/0.5**2)
        if(rrate1d_sub(bcf+1,isub).gt.150) probs(isub)=probs(isub)*0.9
     else
        probs(isub)=0.0
     end if
  end do
  !!$OMP END PARALLEL DO
  piaKu=0.0
  piaKa=0.0
  sumprob=0.0
  do isub=1,31,3
     do k=bzd,bcf
        if(zkuL(k+1).gt.10) then
           rrate1d(k+1)= rrate1d(k+1)+probs(isub)*rrate1d_sub(k+1,isub)
           dm1d(k+1)=dm1d(k+1)+probs(isub)*dm_sub(k+1,isub)
           dn1d(k+1)=dn1d(k+1)+probs(isub)*dn_sub(k+1,isub)
           zkasim(k+1)=zkasim(k+1)+10**(0.1*zkasim_sub(k+1,isub))*probs(isub)
           dzdn(k+1,k+1)=dzdn(k+1,k+1)+10**(0.1*zkasim_sub(k+1,isub)+0.1*dzdn_sub2(k+1,k+1,isub))*probs(isub)
           do kk=k+1,bcf
              if(zkuL(kk+1).gt.10) then
                 dzdn(kk+1,k+1)=dzdn(kk+1,k+1)+10**(0.1*zkasim_sub(kk+1,isub)+0.1*dzdn_sub2(kk+1,k+1,isub))*probs(isub)
              end if
           end do
        end if
     end do
     piaKu=piaKu+probs(isub)*piaHB_sub(isub)
     piaKa=piaKa+probs(isub)*piaKa_sub(isub)
     sumprob=sumprob+probs(isub)
  end do
 
  piaKu=piaKu/(sumprob+1e-10)+piaKuS
  piaKa=piaKa/(sumprob+1e-10)
  do k=bzd,bcf
     if(zkuL(k+1).gt.10) then
        rrate1d(k+1)= rrate1d(k+1)/(sumprob+1e-10)
        dm1d(k+1)=dm1d(k+1)/(sumprob+1e-10)
        dn1d(k+1)=dn1d(k+1)/(sumprob+1e-10)
        zkasim(k+1)=log10(zkasim(k+1)/(sumprob+1e-10))*10.0
        dzdn(k+1,k+1)=(log10(dzdn(k+1,k+1)/(sumprob+1e-10))*10.0-zkasim(k+1))/(-0.1)
        do kk=k+1,bcf
           if(zkuL(kk+1).gt.10) then
              dzdn(kk+1,k+1)=(log10(dzdn(kk+1,k+1)/(sumprob+1e-10))*10.0-log10(zkasim(kk+1)/(sumprob+1e-9))*10)/(-0.1)
           end if
        end do
     end if
  end do
  call cpu_time(finish2)
  dt1=finish1-start
  dt2=finish2-start
  if(rrate1d(bcf+1).gt.150) then
     print*, rrate1d_sub(bcf+1,1:31:3)
     print*, 'probs=',probs(1:31:3)
     print*, 'zetaS=',zetaS(1:31:3)
  end if
end subroutine iter_profcv

subroutine iter_profst_nobb(btop,bzd,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
     dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,itype,dnCoeff_new,dncv,dnp,dzdn,&
     dt1,dt2, dpiadn, piaKuS, piaKaS)
  use tables2
  use tableP2
  use ran_mod
  implicit none
  integer :: btop, bzd, bcf, bsfc, n1d, imu
  real :: zKuL(n1d), dr, zKaL(n1d), dnp(n1d)
  !real :: dmCoeff(2)=(/0.027773772993636318,-0.6624076086959071/)
  real,intent(out) :: dm1d(n1d)
  real,intent(out) :: epst,piaKu,piaKa
  real,intent(out) :: dn1d(n1d), rrate1d(n1d), zKuC(n1d), zKaSim(n1d), dzdn(n1d,n1d), dpiadn(2,n1d)
  real :: dnCoeff_new(2)
  real :: dzKa(n1d), dns(n1d)
  real :: eps, dm, pia, dncv
  integer :: it, ik, k, n1, n1H, itype
  real :: ztrueS,ztrueH,ztrue,f, attKuH, attKaH, attKuS, attKaS
  real :: ftran, zetaS, probs(31), sumprob,  zeta1d(n1d), q
  real :: rrate1d_sub(n1d,31), dn_sub(n1d,31), dm_sub(n1d,31), zkuc_sub(n1d,31), piahb_sub(31)
  real,intent(out) :: piaKuS, piaKaS
  real :: beta, piamax, attKu, attKa, dn, dni
  real :: dnCoeff(2)=(/-0.00570364,  0.13319214/)
  real :: dmCoeff(3)= (/-2.56885571e-04,  7.18909743e-02, -1.60889523e+00/)
  !(/-4.57599678e-04,  7.852247958e-02, -1.78316499e+00/)
  !(/-4.57599678e-04,  8.52247958e-02, -1.78316499e+00/)
  real :: rn, dm_old!, dm_sub(n1d,31)
  real :: zka1,zka2,attka2, attku2
  real,intent(out) :: dt1, dt2
  real :: start, finish1, finish2
  integer :: kk, imc, isub
  real :: dmm
  real :: piaSRTKu, dzdn_sub2(n1d,n1d,31), dzdn_sub1(n1d,n1d,31), zkasim_sub(n1d,31), piaKa_sub(31)
  integer :: relPIASRTKu
  real :: ddn
  piaKu=0
  !dmCoeff=array([ 0.02893781, -0.69481455])
  dzdn=0.0
  !if(itype.eq.1) then
  dnCoeff=dnCoeff_new
  !end if
  zKuC=zKuL
  dns=0
  dpiadn=0
  dZKa=0
  rrate1d=0
  call cpu_time(start)
  do it=1,2
     piaKu=0.0
     piaKa=0.0
     do k=btop,min(bcf,bzd-1)
        if (zKuC(k+1).gt.10) then
           if (k.lt.bzd) then
              ztrue=zKuC(k+1)
           end if
           if(ztrue.lt.40) then
              ztrueS=ztrue
              ztrueH=0.0
           else
              ztrueS=40
              ztrueH=10*log10(10**(0.1*ztrue)-10**(0.1*39.999))
           endif
           !n1,n2=bisection(zKuSJ,ztrueS-10*dns[k+1])
           call bisection2(zKuSJ,nbinS2,ztrueS-10*dns(k+1), n1)
           attKuS=att13TableS2(n1,imu)*10**dns(k+1)
           attKaS=att35TableS2(n1,imu)*10**dns(k+1)
           !n1H,n2H=bisection(zKuHJ,ztrueH)
           call bisection2(zKuSJ,nbinS2,ztrueH, n1H)
           attKuH=att13TableH(n1H,imu)
           attKaH=att35TableH(n1H,imu)
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           zKuC(k+1)=zKuL(k+1)+piaKu
           zKaSim(k+1)=10*log10(10**(0.1*z35TableS2(n1,imu)+dns(k+1))+&
                10**(0.1*z35TableH(n1H,imu)))-piaKa
           if(zKaL(k+1).gt.10) then
              dZKa(k+1)=zKaSim(k+1)-zKaL(k+1)
           end if
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           dm1d(k+1)=0.5*(d013TableS2(n1,imu)+d013TableH(n1H,imu))
           dn1d(k+1)=dns(k+1)
           rrate1d(k+1)=pr13TableS2(n1,imu)*10**dns(k+1)+pr13TableH(n1H,imu)
        end if
     end do
     if(it.eq.1) then
        do k=btop,min(bcf,bzd-1)
           dns(k+1)=dns(k+1)-0.6*dZKa(k+1)
           if(dns(k+1).gt.0.975) dns(k+1)=0.975
           if(dns(k+1).lt.-1.0) dns(k+1)=-1.0
        end do
     end if
  end do
  
  piaKuS=piaKu+0.0
  piaKaS=piaKa+0.0

  zeta1d=0
  beta=0.71
  
  piamax=54-zKuL(bcf+1)
  if(piamax<0) piamax=56-zKuL(bcf+1)
  if(piamax<0) piamax=4
  q=0.2*log(10.0)
  zKuC(bzd+1:bcf+1)=zKuL(bzd+1:bcf+1)+piaKuS
  epst=eps
  attKu=0.0
  attKa=0.0
  f=1.0
  zeta1d=0
  sumprob=0
  piaKu=0.0
  rrate1d(bzd+1:bcf+1)=0.0
  dm1d(bzd+1:bcf+1)=0.0
  dn1d(bzd+1:bcf+1)=0.0
  rrate1d_sub=0
  dm_sub=0
  dn_sub=0
  zkasim_sub=0
  dzdn_sub2=0
  dzdn_sub1=0
  zkasim(bzd+1:bcf+1)=0
  dzdn(bzd+1:bcf+1,bzd+1:bcf+1)=0
  call cpu_time(finish1)
  ddn=0.025
  piaKuS=piaKu+0.0
  piaKaS=piaKa+0.0

  zeta1d=0
  beta=0.76
  
  piamax=52-zKuL(bcf+1)
  if(piamax<0) piamax=56-zKuL(bcf+1)
  if(piamax<0) piamax=10.0
  q=0.2*log(10.0)
  zKuC(bzd+1:bcf+1)=zKuL(bzd+1:bcf+1)+piaKuS
 
  attKu=0.0
  attKa=0.0
  f=1.0
  do it=1,3
     zetaS=0.0
     piaKa=piaKaS+0.0
     do k=bzd,bcf
        if(zKuC(k+1)>0) then
           ztrue=zKuC(k+1)
           ftran=1
           !dni=1.5*(dnCoeff(1)*ztrue+dnCoeff(2))+0.2*log(epst)
           dn=1.0*(dnCoeff(1)*ztrue+dnCoeff(2))+dnp(k+1)+0.2*log(epst)+dncv
           n1=(ztrue-10*dn-zmin)/dzbin
           dm=d013Table(n1,imu)
           call bisection2(zKudN(1:nbins),nbins,ztrue-10*(0.2*log(epst)+dncv+dnp(k+1)), n1)
           dm=d013Table(n1,imu)
           dn=(ztrue-z13Table(n1,imu))/10.0
           !------------------------------------------------
           !dm=exp(dmCoeff(1)*ztrue**2+dmCoeff(2)*ztrue+dmCoeff(3))
           !if(dm.gt.0.5.and.dm.lt.2.8) then
           !if (it.eq.2) then
           !   rn=normal2(0.0,1.0)
           !   dm=0.875*dm+0.125*(1.6+0.25*normal2(0.0,1.0))
           !end if
           !end if
           !if(dm<0.31) dm=0.31
           !if(dm>3) dm=3
           !call bisection2(d013Table(:,imu),nbins,dm, n1)
           !dn=(ztrue-zKuSJ(n1))/10
           !dn=0.5*(dn+dn)
           !---------------------------------------------------
           if(10*dn+ztrue.gt.zKuSJ(nbins)) dn=(zKuSJ(nbins)-ztrue)/10.0
           if(10*dn+ztrue.lt.zKuSJ(1)) dn=(zKuSJ(1)-ztrue)/10.0
           n1=((ztrue-10*dn-zmin)/dzbin)+1
           if(n1.lt.1) n1=1
           if(n1.gt.nbins) n1=nbins
           dm=d013Table(n1,imu)
           attKu=att13Table(n1,imu)*10**dn
           attKa=att35Table(n1,imu)*10**dn
           zKa1=z35Table(n1,imu)+10*dn
           if(it.eq.3) then
              if(n1.lt.nbins-4) then
                 zKa2=z35Table(n1+4,imu)+10*(dn-0.1)
                 attKa2=att35Table(n1+4,imu)*10**(dn-0.1)
                 attKu2=att13Table(n1+4,imu)*10**(dn-0.1)
                 dpiadn(2,k+1)=(attKa2-attKa)/(-0.1)*dr
                 dpiadn(1,k+1)=(attKu2-attKu)/(-0.1)*dr
                 dzdn(k+1,k+1)=dzdn(k+1,k+1)+(zka2-zka1)/(-0.1)-(attKa2-attKa)/(-0.1)*dr
                 do kk=k+1,bcf
                    dzdn(kk+1,k+1)=dzdn(kk+1,k+1)-(attKa2-attKa)/(-0.1)*dr
                 end do
              end if
           end if
           zetaS=zetaS+attKu/10**(zKuC(k+1)*0.1*beta)*&
                10**(zKuL(k+1)*0.1*beta)*dr
           rrate1d(k+1)=pr13Table(n1,imu)*10**dn
           dn1d(k+1)=dn
           dm1d(k+1)=dm
           piaKa=piaKa+attKa*dr
           zKaSim(k+1)=z35Table(n1,imu)+10*dn-piaKa
           piaKa=piaKa+attKa*dr
        else
           attKu=0.0
           attKa=0.0
        end if
        zeta1d(k+1)=zetaS
     end do
     eps=min(1.0,(1-10**(-0.1*beta*f*piamax))/(q*beta*zetaS))
     epst=epst*eps
     do k=bzd,bcf
        zKuC(k+1)=zKuL(k+1)+piaKuS-10/beta*log10(1-eps*q*beta*zeta1d(k+1))
     end do
     piaKu=(-10/beta*log10(1-eps*q*beta*zeta1d(bcf+1)))+attKu*(bsfc-bcf)*2*dr
     piaKa=piaKa+attKa*(bsfc-bcf)*2*dr
     if(isnan(piaKu).eqv..true.) then
        print*, eps, zetaS, piamax, zeta1d(bcf+1)
        stop
     end if
     if(isnan(piaKa).eqv..true.) then
        print*,dns(btop:min(bcf,bzd))
        print*, btop, bzd, bcf
        print*,'*************'
        print*,zKuC(btop:min(bcf,bzd))
        print*, 'st',eps, zetaS, piamax, zeta1d(bcf+1), piaKa, piaKas,attKa, bsfc
        stop
     end if
  end do
  call cpu_time(finish2)
  dt1=finish1-start
  dt2=finish2-start
end subroutine iter_profst_nobb
subroutine rain()
  
end subroutine rain

subroutine iter_profst(btop,bzd,bb,bbt,bbb,bcf,bsfc,zKuL,zKaL,dr,n1d,eps,imu,&
     dn1d,dm1d,rrate1d,zKuC,zKaSim,epst,piaKu,piaKa,dnst,dnCoeff_new,dnp,&
     dzdn,dpiadn,piaKuS,piaKaS)
  use tables2
  use tableP2
  use ran_mod
  implicit none
  integer :: btop, bzd, bcf, bsfc, n1d, imu, bb,bbt,bbb
  real :: zKuL(n1d), dr, zKaL(n1d)
  real, intent(out) :: dzdn(n1d,n1d), dpiadn(2,n1d)
  real :: dnCoeff_new(2)
  !real :: dmCoeff(2)=(/0.027773772993636318,-0.6624076086959071/)
  real,intent(out) :: dm1d(n1d)
  real,intent(out) :: epst,piaKu,piaKa
  real,intent(out) :: dn1d(n1d), rrate1d(n1d), zKuC(n1d), zKaSim(n1d)
  real :: dzKa(n1d), dns(n1d), dnp(n1d)
  real :: eps, dm, pia, dnst
  integer :: it, ik, k, n1, n1H
  real :: ztrueS,ztrueH,ztrue,f, attKuH, attKaH, attKuS, attKaS
  real :: ftran, zetaS, zeta1d(n1d), q
  real,intent(out) :: piaKuS, piaKaS
  real :: beta, piamax, attKu, attKa, dn, dni
  real :: dnCoeff(2)=(/-0.00570364,  0.13319214/)
  integer :: n1S,n1BB,n1b
  real :: attKuBB, attKaBB, Z1, Z2
  real :: dmCoeff(3)=(/-2.56885571e-04,  7.18909743e-02, -1.60889523e+00/)
  
  !=(/-4.57599678e-04,  7.752247958e-02, -1.78316499e+00/)
!  array([-2.56885571e-04,  7.18909743e-02, -1.60889523e+00])

  real :: rn, r1, zka1, zka2, attKa2, attKu2
  integer :: kk
  piaKu=0
  !dmCoeff=array([ 0.02893781, -0.69481455])
  dnCoeff=dnCoeff_new
  zKuC=zKuL
  dns=0
  dzdn=0
  dpiadn=0
  dZKa=0
  piaKu=0.0
  piaKa=0.0
  piaKuS=0.0
  piaKaS=0.0
  do it=1,2
     do k=btop,min(bcf,bbt-1)
        if (zKuC(k+1).gt.10) then
           if (k.lt.bzd) then
              ztrue=zKuC(k+1)
           end if
           if(ztrue.lt.40) then
              ztrueS=ztrue
              ztrueH=0.0
           else
              ztrueS=40
              ztrueH=10*log10(10**(0.1*ztrue)-10**(0.1*39.999))
           endif
           !n1,n2=bisection(zKuSJ,ztrueS-10*dns[k+1])
           call bisection2(zKuSJ,nbinS2,ztrueS-10*dns(k+1), n1)
           attKuS=att13TableS2(n1,imu)*10**dns(k+1)
           attKaS=att35TableS2(n1,imu)*10**dns(k+1)
           !n1H,n2H=bisection(zKuHJ,ztrueH)
           call bisection2(zKuSJ,nbinS2,ztrueH, n1H)
           attKuH=att13TableH(n1H,imu)
           attKaH=att35TableH(n1H,imu)
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           zKuC(k+1)=zKuL(k+1)+piaKu
           zKaSim(k+1)=10*log10(10**(0.1*z35TableS2(n1,imu)+dns(k+1))+&
                10**(0.1*z35TableH(n1H,imu)))-piaKa
           if(zKaL(k+1).gt.10) then
              dZKa(k+1)=zKaSim(k+1)-zKaL(k+1)
           end if
           piaKu=piaKu+(attKuS+attKuH)*dr
           piaKa=piaKa+(attKaS+attKaH)*dr
           dm1d(k+1)=0.5*(d013TableS2(n1,imu)+d013TableH(n1H,imu))
           dn1d(k+1)=dns(k+1)
           rrate1d(k+1)=pr13TableS2(n1,imu)*10**dns(k+1)+pr13TableH(n1H,imu)
        end if
     end do
     if(it.eq.1) then
        do k=btop,bbt-1
           if(zkuc(k+1).gt.10) then
              dns(k+1)=dns(k+1)-0.6*dZKa(k+1)
              !print*, dns(k+1), k+1, bbt
              if(dns(k+1).gt.0.975) dns(k+1)=0.975
              if(dns(k+1).lt.-1.0) dns(k+1)=-1.0
           endif
        end do
     end if
  end do
  !print*,'bbt=', dns(bbt), dZKa(bbt), bbt, bcf
  !print*, bbt, bb
  !if(btop.gt.bbt) print*, piaKuS,piaKaS
  do k=bbt,min(bb,bcf)
     if(zKuC(k+1)>10) then
        f=(k-bbt+0.0)/(bb-bbt)
        if(k.eq.bbt) then
           f=0.875
        else
           f=1
        end if
        Z1=log10((1-f)*10**(0.1*zKuC(k+1))+1e-9)*10.
        Z2=log10(f*10**(0.1*zKuC(k+1))+1e-9)*10.
        n1S=(Z1-zmin-10*dns(bbt))/dzbin+1
        n1BB=(Z2-zmin-10*dns(bbt))/dzbin+1
        if(n1S.gt.nbinS2) n1S=nbinS2
        if(n1BB.gt.nbins) n1BB=nbins
        if(n1S.lt.1) n1S=1
        if(n1BB.lt.1) n1BB=1
        !print*, n1S,n1BB,f, Z1,Z2,zKuC(k+1)
        attKuS=att13TableS2(n1S,imu)*10**dns(bbt)
        attKaS=att35TableS2(n1S,imu)*10**dns(bbt)
        attKuBB=att13TableBB(n1S,imu)*10**dns(bbt)
        attKaBB=att35TableBB(n1S,imu)*10**dns(bbt)
        piaKu=piaKu+(attKuS+attKuBB)*dr
        piaKa=piaKa+(attKaS+attKaBB)*dr
        zKuC(k+1)=zKuL(k+1)+piaKu
        zKaSim(k+1)=10*log10(10**(0.1*z35TableS2(n1S,imu)+dns(bbt))+&
             10**(0.1*z35TableBB(n1BB,imu)+dns(bbt)))-piaKa
        piaKu=piaKu+(attKuS+attKuBB)*dr
        piaKa=piaKa+(attKaS+attKaBB)*dr
        dm1d(k+1)=(1-f)*d013TableS2(n1S,imu)+f*d013TableBB(n1BB,imu)
        dn1d(k+1)=dns(bbt)
        rrate1d(k+1)=pr13TableS2(n1S,imu)*10**dns(bbt)+pr13TableBB(n1BB,imu)*10**dns(bbt)
     end if
  end do
  epst=eps
!  if(btop.gt.bbt) print*, 'bb', piaKuS,piaKaS
  !if(btop.gt.bbt) print*, 'bb1', piaKuS,piaKaS, piaKu,piaKa,btop,bbb,bcf,dns(bbt+1),bbt
  !if(btop.gt.bbt) print*, bbb, bb, epst,dnp(bb+2), dnp(bbb+1)
  !print*,piaKus,piaKas,bb+1,bbb,bb
  do k=bb+1,min(bbb-1,bcf)
     if(zKuC(k+1)>0) then
        f=(k-bb+0.0)/(bbb-bb+1e-9)
        f=0.99
        Z1=(1-f)*zKuC(k+1)
        Z2=(f)*zKuC(k+1)
        ztrue=zKuC(bbb+1)
        if(ztrue.gt.10) then
           n1=(ztrue-zmin-10*dns(bbt))/dzbin+1
           !print*, f, ztrue, dns(bbt), n1, dnp(k+1)
           !dn=1.0*(dnCoeff(1)*Z2+dnCoeff(2))+0.2*log(epst)
           dn=dns(bbt)!1.0*(dnCoeff(1)*ztrue+dnCoeff(2))+dnp(k+1)+0.2*log(epst)+dnst
           !call bisection2(zKudN(1:nbins),nbins,ztrue-10*(dnst), n1)
           !dm=d013Table(n1,imu)
           !dn=(ztrue-z13Table(n1,imu))/10.0
           !print*, 'dn=',dn
           if(10*dn+ztrue.gt.zKuSJ(nbins)) dn=(zKuSJ(nbins)-ztrue)/10.0
           if(10*dn+ztrue.lt.zKuSJ(1)) dn=(zKuSJ(1)-ztrue)/10.0
           n1b=((Z2-zmin-10*dn)/dzbin)+1
           if(n1.gt.nbins) n1=nbins
           if(n1b.gt.nbins) n1b=nbins
           if(n1.lt.1) n1=1
           if(n1b.lt.1) n1b=1
           if(btop.gt.bb) then
              !print*,'ind', z1,z2,ztrue,zkuc(k+1)
              !print*, n1,n1b,dn,dns(bbt)
           endif
           attKu=att13Table(n1b,imu)*10**dn
           attKa=att35Table(n1b,imu)*10**dn
           attKuBB=att13TableBB(n1,imu)*10**dns(bbt)
           attKaBB=att35TableBB(n1,imu)*10**dns(bbt)
           piaKu=piaKu+(attKu+attKuBB)*dr
           piaKa=piaKa+(attKa+attKaBB)*dr
           zKuC(k+1)=zKuL(k+1)+piaKu
           zKaSim(k+1)=10*log10(10**(0.1*z35TableBB(n1,imu)+dns(bbt))+&
                10**(0.1*z35Table(n1b,imu)+dn))-piaKa
           piaKu=piaKu+(attKu+attKuBB)*dr
           piaKa=piaKa+(attKa+attKaBB)*dr
           dm1d(k+1)=(1-f)*d013TableBB(n1,imu)+f*d013Table(n1b,imu)
           dn1d(k+1)=(1-f)*dns(bbt)+f*dn
           rrate1d(k+1)=pr13TableBB(n1,imu)*10**dns(bbt)+pr13Table(n1b,imu)*10**dn
        end if
        !print*, zKuC(k+1),Z1,Z2,dn,n1b,n1
     end if
  end do
  !if(btop.gt.bbt) print*, 'bb', piaKuS,piaKaS, piaKu,piaKa
  piaKuS=piaKu+0.0
  piaKaS=piaKa+0.0

  zeta1d=0
  beta=0.76
  
  piamax=52-zKuL(bcf+1)
  if(piamax<0) piamax=56-zKuL(bcf+1)
  if(piamax<0) piamax=10.0
  
  q=0.2*log(10.0)
  zKuC(bzd+1:bcf+1)=zKuL(bzd+1:bcf+1)+piaKuS
 
  attKu=0.0
  attKa=0.0
  f=1.0
  do it=1,3
     zetaS=0.0
     piaKa=piaKaS+0.0
     do k=bbb,bcf
        if(zKuC(k+1)>0) then
           ztrue=zKuC(k+1)
           ftran=1
           !dni=1.5*(dnCoeff(1)*ztrue+dnCoeff(2))+0.2*log(epst)
           dn=1.0*(dnCoeff(1)*ztrue+dnCoeff(2))+dnp(k+1)+0.2*log(epst)+dnst
           n1=(ztrue-10*dn-zmin)/dzbin
           dm=d013Table(n1,imu)
           call bisection2(zKudN(1:nbins),nbins,ztrue-10*(0.2*log(epst)+dnst+dnp(k+1)), n1)
           dm=d013Table(n1,imu)
           dn=(ztrue-z13Table(n1,imu))/10.0
           !------------------------------------------------
           !dm=exp(dmCoeff(1)*ztrue**2+dmCoeff(2)*ztrue+dmCoeff(3))
           !if(dm.gt.0.5.and.dm.lt.2.8) then
           !if (it.eq.2) then
           !   rn=normal2(0.0,1.0)
           !   dm=0.875*dm+0.125*(1.6+0.25*normal2(0.0,1.0))
           !end if
           !end if
           !if(dm<0.31) dm=0.31
           !if(dm>3) dm=3
           !call bisection2(d013Table(:,imu),nbins,dm, n1)
           !dn=(ztrue-zKuSJ(n1))/10
           !dn=0.5*(dn+dn)
           !---------------------------------------------------
           if(10*dn+ztrue.gt.zKuSJ(nbins)) dn=(zKuSJ(nbins)-ztrue)/10.0
           if(10*dn+ztrue.lt.zKuSJ(1)) dn=(zKuSJ(1)-ztrue)/10.0
           n1=((ztrue-10*dn-zmin)/dzbin)+1
           if(n1.lt.1) n1=1
           if(n1.gt.nbins) n1=nbins
           dm=d013Table(n1,imu)
           attKu=att13Table(n1,imu)*10**dn
           attKa=att35Table(n1,imu)*10**dn
           zKa1=z35Table(n1,imu)+10*dn
           if(it.eq.3) then
              if(n1.lt.nbins-4) then
                 zKa2=z35Table(n1+4,imu)+10*(dn-0.1)
                 attKa2=att35Table(n1+4,imu)*10**(dn-0.1)
                 attKu2=att13Table(n1+4,imu)*10**(dn-0.1)
                 dzdn(k+1,k+1)=dzdn(k+1,k+1)+(zka2-zka1)/(-0.1)-(attKa2-attKa)/(-0.1)*dr
                 dpiadn(2,k+1)=(attKa2-attKa)/(-0.1)*dr
                 dpiadn(1,k+1)=(attKu2-attKu)/(-0.1)*dr
                 do kk=k+1,bcf
                    dzdn(kk+1,k+1)=dzdn(kk+1,k+1)-(attKa2-attKa)/(-0.1)*dr
                 end do
              end if
           end if
           zetaS=zetaS+attKu/10**(zKuC(k+1)*0.1*beta)*&
                10**(zKuL(k+1)*0.1*beta)*dr
           rrate1d(k+1)=pr13Table(n1,imu)*10**dn
           dn1d(k+1)=dn
           dm1d(k+1)=dm
           piaKa=piaKa+attKa*dr
           zKaSim(k+1)=z35Table(n1,imu)+10*dn-piaKa
           piaKa=piaKa+attKa*dr
        else
           attKu=0.0
           attKa=0.0
        end if
        zeta1d(k+1)=zetaS
     end do
     eps=min(1.0,(1-10**(-0.1*beta*f*piamax))/(q*beta*zetaS))
     epst=epst*eps
     do k=bbb,bcf
        zKuC(k+1)=zKuL(k+1)+piaKuS-10/beta*log10(1-eps*q*beta*zeta1d(k+1))
     end do
     piaKu=(-10/beta*log10(1-eps*q*beta*zeta1d(bcf+1)))+attKu*(bsfc-bcf)*2*dr
     piaKa=piaKa+attKa*(bsfc-bcf)*2*dr
     if(isnan(piaKu).eqv..true.) then
        print*, eps, zetaS, piamax, zeta1d(bcf+1)
        stop
     end if
     if(isnan(piaKa).eqv..true.) then
        print*,dns(btop:min(bcf,bbb))
        print*, btop, bbt, bb, bbb, bcf
        print*,'*************'
        print*,zKuC(btop:min(bcf,bbb))
        print*, 'st',eps, zetaS, piamax, zeta1d(bcf+1),piaKus, piaKa, piaKas,attKa, bsfc
        stop
     end if
  end do
end subroutine iter_profst

