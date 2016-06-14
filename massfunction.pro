


pro massfunc,printfig=printfig

;reading the data in
master=mrdfits('/Users/admin/Work/IRAM/Observations/Tables/MASTER/COLDGASS_AllCO_XA.fits',1)
ind=where(master.flag_co eq 1)
m=master[ind] 
n=n_elements(m)


device,decomposed=0
loadct,0,/silent


if (printfig eq 1 ) then epsinit,'massfunc.eps',xsize=5,ysize=5,/VECTOR,/COLOR


minm=7.6
maxm=10.2
nbins=16
dm=(maxm-minm)/float(nbins)


count=intarr(nbins)
mass=fltarr(nbins)

restore,'weightedcounts.sav'



for i=0,nbins-1 do begin
   ind=where(m.logmh2 ge minm+i*dm and m.logmh2 lt minm+(i+1)*dm)
   if (n_elements(ind) eq 1) then begin
      if (ind lt 0) then count[i]=0 
      if (ind ge 0) then begin
         sel=where(m[ind].gass eq gass)
         count[i]=wc[sel]
      endif
   endif else begin
      sel=intarr(n_elements(ind))
      for k=0,n_elements(ind)-1 do begin
         sel[k]=where(m[ind[k]].gass eq gass)
      endfor
      count[i]=total(wc[sel])
   endelse
   mass[i]=minm+(i+0.5)*dm
endfor


;calculate the volume of the survey
zmin=0.025
zmax=0.05
dmin=lumdist(zmin,/silent)
dmax=lumdist(zmax,/silent)

vol=4/3.0*!pi*(dmax^3-dmin^3)*(0.48/(4.0*!pi)) ;using Tom's estimate of the solid angle of the survey

myphi=alog10(count/vol*12006.0/366.0/dm)


plot,mass,myphi,psym=8,$
     xrange=[7.5,10.5],yrange=[-5,-1],xstyle=1,ystyle=1,$
     position=[0.15,0.15,0.95,0.95],xtitle=TeXtoIDL('log M_{H2} [M_{sun}]'),$
     ytitle=TeXtoIDL('log \phi_{H2} [Mpc^{-3} dex^{-1}]'),$
     xthick=3,ythick=3,charthick=3,charsize=1.5,/nodata


mst=2.81E9/0.7^2
alpha=-1.18
phist=0.0089*0.7^3
x=10D^(7.5+findgen(200)/200.0*3.5)
y=alog(10.0)*phist*(x/mst)^(alpha+1)*exp(-1*x/mst)

oplot,alog10(x),alog10(y),thick=7,linestyle=2

loadct,13,/silent
plotsym,8,thick=6
oplot,mass,myphi,psym=8,symsize=1.5,color=150




;Adding the non-detections 
master=mrdfits('/Users/admin/Work/IRAM/Observations/Tables/MASTER/COLDGASS_AllCO_XA.fits',1)
ind=where(master.flag_co eq 2)
m=master[ind] 
n=n_elements(m)

countnd=intarr(nbins)
massnd=fltarr(nbins)

restore,'weightedcounts.sav'

for i=0,nbins-1 do begin
   ind=where(m.logmh2lim ge minm+i*dm and m.logmh2lim lt minm+(i+1)*dm)
   if (n_elements(ind) eq 1) then begin
      if (ind lt 0) then countnd[i]=0 
      if (ind ge 0) then begin
         sel=where(m[ind].gass eq gass)
         countnd[i]=wc[sel]
      endif
   endif else begin
      sel=intarr(n_elements(ind))
      for k=0,n_elements(ind)-1 do begin
         sel[k]=where(m[ind[k]].gass eq gass)
      endfor
      countnd[i]=total(wc[sel])
   endelse
   massnd[i]=minm+(i+0.5)*dm
endfor

restore,'/Users/admin/Work/IRAM/LowMass/Master/LOWMASS_MASTER.sav'
m=master
ind=where(m.flag_co10 eq 2 and m.mass lt 10.0)
m=m[ind]

restore,'weightedcountslow.sav'
for i=0,nbins-1 do begin
   ind=where(m.limlogmh2 ge minm+i*dm and m.limlogmh2 lt minm+(i+1)*dm)
   if (n_elements(ind) eq 1) then begin
      ;if (ind lt 0) then count2[i]=0 
      if (ind ge 0) then begin
         sel=where(m[ind].id eq gass)
         countnd[i]=countnd[i]+wc[sel]
      endif
   endif else begin
      sel=intarr(n_elements(ind))
      for k=0,n_elements(ind)-1 do begin
         sel[k]=where(m[ind[k]].id eq gass)
      endfor
      countnd[i]=countnd[i]+total(wc[sel])
   endelse
endfor




myphind=alog10(countnd/vol*12006.0/366.0/dm)
oplot,massnd,myphind,psym=8,symsize=1.5,color=220


;-----------------------------------------------------
;Adding the low mass galaxies
restore,'/Users/admin/Work/IRAM/LowMass/Master/LOWMASS_MASTER.sav'
m=master
ind=where(m.flag_co10 eq 1 and m.mass lt 10.0)
m=m[ind]

restore,'weightedcountslow.sav'

count2=intarr(nbins)
mass2=fltarr(nbins)

for i=0,nbins-1 do begin
   ind=where(m.logmh2 ge minm+i*dm and m.logmh2 lt minm+(i+1)*dm)
   if (n_elements(ind) eq 1) then begin
      if (ind lt 0) then count2[i]=0 
      if (ind ge 0) then begin
         sel=where(m[ind].id eq gass)
         count2[i]=wc[sel]
      endif
   endif else begin
      sel=intarr(n_elements(ind))
      for k=0,n_elements(ind)-1 do begin
         sel[k]=where(m[ind[k]].id eq gass)
      endfor
      count2[i]=total(wc[sel])
   endelse
   mass2[i]=minm+(i+0.5)*dm
endfor



;calculate the volume of the survey
zmin=0.01
zmax=0.02
dmin=lumdist(zmin,/silent)
dmax=lumdist(zmax,/silent)
vol=4/3.0*!pi*(dmax^3-dmin^3)*(0.48/(4.0*!pi)) ;using Tom's estimate of the solid angle of the survey
myphi2=alog10(count2/vol*764.0/89.0/dm)


oplot,mass2,myphi2,psym=8,symsize=1.5,color=80
plotsym,0,/FILL
phitot=alog10(10D^myphi+10D^myphi2+10D^myphind)
oplot,mass2,phitot,psym=8,symsize=1.5,color=255





;FIT and PLOT the Schechter funtion to the combined mass function
ind=where(mass2 gt 8.9 and phitot ne 0.0 and FINITE(phitot))
start=[0.0089*0.7^3,2.81D9/0.7^2,-1D]
res = MPFITFUN('SCHECHFIT', 10D^mass2[ind], 10D^phitot[ind], 1.0 , start)
y=alog(10.0)*res[0]*(x/res[1])^(res[2]+1)*exp(-1*x/res[1])
loadct,0,/silent
oplot,alog10(x),alog10(y),linestyle=0,thick=6,color=120

if (printfig eq 1 ) then epsterm

ending:

loadct,0,/silent
plotsym,0,/FILL
device,decomposed=1
!p.multi=0

end



pro mkweights

master=mrdfits('/Users/admin/Work/IRAM/Observations/Tables/MASTER/COLDGASS_AllCO_XA.fits',1)
ind=where(master.flag_co eq 1 or master.flag_co eq 2)
m=master[ind] 
n=n_elements(m)



;reading the parent sample, and making the weights
binsize=0.2
minm=10.0
maxm=11.8
   ps=mrdfits('/Users/admin/Work/IRAM/Observations/Tables/GASS/PS_100701.fits',1,hdr)
   nps=n_elements(ps)
   masses=ps.mass_p50
   hstPS=histogram(masses,bin=binsize,min=minm,max=maxm,locations=xweights)
   hstCG=histogram(m.mass_p50,bin=binsize,min=minm,max=maxm,locations=xweights)
   weights=float(hstPS)/float(hstCG)
   ind=where(~FINITE(weights))
   weights[ind]=0.0
   save,xweights,weights,filename='weights_masses.sav'

device,decomposed=0

plothist,masses,bin=binsize,xrange=[9.8,11.8],xstyle=1,$
         xtitle='!5 log M*',thick=8,xthick=3,ythick=3,charthick=3
loadct,39,/silent
plothist,m.mass_p50,bin=binsize,color=88,peak=1500,/overplot,$
         /FILL,/FLINE,FCOLOR=88,thick=4
loadct,0,/silent

device,decomposed=1

ind=where(m.flag_co eq 1 or m.flag_co eq 2)
x=m[ind].mass_p50
y=m[ind].logmh2ms
xm=m[ind].mass_p50
nn=n_elements(y)
;make a weights array
warr=fltarr(nn)
nb=n_elements(xweights)
for i=0,nb-2 do begin
     ind=where(xm gt xweights[i] and xm le xweights[i+1])
     if (n_elements(ind) eq 1) then begin
         if (ind lt 0) then continue
      endif
     warr[ind]=replicate(weights[i],n_elements(ind))
  endfor
;ycalc=total((10^y)*warr)/total(warr)
;val=mean(10D^y)
;err=robust_sigma(10D^y)
;print,'all  - ',val,ycalc,err
print,'=========================='

wc=warr/total(warr)*n_elements(warr)
print,total(wc)
print,min(wc),max(wc)

gass=m.gass

save,gass,wc,filename='weightedcounts.sav'
end




pro mkweightslow

restore,'/Users/admin/Work/IRAM/LowMass/Master/LOWMASS_MASTER.sav'
ind=where(master.flag_co10 eq 1 or master.flag_co10 eq 2 and master.mass lt 10.0)
m=master[ind]

;reading the parent sample, and making the weights
binsize=0.2
minm=9.0
maxm=10.0
   nps=n_elements(master)
   masses=master.mass
   hstPS=histogram(masses,bin=binsize,min=minm,max=maxm,locations=xweights)
   hstCG=histogram(m.mass,bin=binsize,min=minm,max=maxm,locations=xweights)
   weights=float(hstPS)/float(hstCG)
   ind=where(~FINITE(weights))
   weights[ind]=0.0
   save,xweights,weights,filename='weights_masses_low.sav'

device,decomposed=0

plothist,masses,bin=binsize,xrange=[8.8,10.2],xstyle=1,$
         xtitle='!5 log M*',thick=8,xthick=3,ythick=3,charthick=3
loadct,39,/silent
plothist,m.mass,bin=binsize,color=88,peak=70,/overplot,$
         /FILL,/FLINE,FCOLOR=88,thick=4
loadct,0,/silent

device,decomposed=1

ind=where(m.flag_co10 eq 1 or m.flag_co10 eq 2)
x=m[ind].mass
y=m[ind].logmh2ms
xm=m[ind].mass
nn=n_elements(y)
;make a weights array
warr=fltarr(nn)
nb=n_elements(xweights)
for i=0,nb-2 do begin
     ind=where(xm gt xweights[i] and xm le xweights[i+1])
     if (n_elements(ind) eq 1) then begin
         if (ind lt 0) then continue
      endif
     warr[ind]=replicate(weights[i],n_elements(ind))
  endfor
;ycalc=total((10^y)*warr)/total(warr)
;val=mean(10D^y)
;err=robust_sigma(10D^y)
;print,'all  - ',val,ycalc,err
print,'=========================='

wc=warr/total(warr)*n_elements(warr)
print,total(wc)
print,min(wc),max(wc)

gass=m.id

save,gass,wc,filename='weightedcountslow.sav'
end
