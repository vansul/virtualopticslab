import os; os.makedirs('fig',exist_ok=True)
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim import *
import mc
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'axes.titlesize':9,'axes.titleweight':'bold','figure.dpi':100})
NAVY='#1f3a8a';OR='#e8590c';GR='#6b7a8c'
def rgbw(w):
    r=g=b=0.
    if w<440:r=(440-w)/60;b=1
    elif w<490:g=(w-440)/50;b=1
    elif w<510:g=1;b=(510-w)/20
    elif w<580:r=(w-510)/70;g=1
    elif w<645:r=1;g=(645-w)/65
    else:r=1
    f=.3+.7*(w-380)/40 if w<420 else (.3+.7*(780-w)/80 if w>700 else 1)
    return np.array([r*f,g*f,b*f])
gm=lambda v:np.clip(v,0,1)**.45
def strip(ax,xs,I,lam,ext_=None):
    c=rgbw(lam); img=gm(I)[None,:,None]*c[None,None,:]; img=np.repeat(img,12,0)
    ax.imshow(img,aspect='auto',extent=[xs[0],xs[-1],0,1]);ax.set_yticks([]);ax.grid(False);ax.set_xticks([])
    for s in ax.spines.values():s.set_visible(False)
def two(w=6.3,h=3.3):
    f=plt.figure(figsize=(w,h),dpi=220);gs=f.add_gridspec(2,1,height_ratios=[1,3.2],hspace=.08);return f,f.add_subplot(gs[0]),f.add_subplot(gs[1])
def save(f,n):f.savefig(f'fig/{n}.png',bbox_inches='tight',dpi=220);plt.close(f)
# F1 double slit case1
lam,d,a,D=550,.5,.05,1.;b=lam*1e-9*D/(d*1e-3)*1e3;xs=np.linspace(-8*b,8*b,2000);k=xs*1e-3/(lam*1e-9*D)
I=np.cos(PI*d*1e-3*k)**2*sinc(PI*a*1e-3*k)**2;sim_b,_=young(lam,d,a,D)
f,a1,a2=two();strip(a1,xs,I,lam);a2.plot(xs,I,color=NAVY,lw=1.2,label='Intensity (interference × envelope)')
a2.plot(xs,sinc(PI*a*1e-3*k)**2,'--',color=GR,lw=.9,label='Single-slit envelope');px,py=ext(xs,I,1)
m=py>.1;a2.plot(px[m],py[m],'o',ms=2.5,color=OR,label='Detected peaks');a2.set_xlabel('y (mm)');a2.set_ylabel('I / I₀')
a1.set_title(f'β simulated = {sim_b:.4f} mm,  β theory = {b:.4f} mm',loc='left',pad=4);a2.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.26),ncol=3);save(f,'f1')
# F2 single slit
lam,a,D=550,.1,1;y1=lam*1e-9*D/(a*1e-3)*1e3;xs=np.linspace(-4.5*y1,4.5*y1,2000);I=sinc(PI*a*1e-3*xs*1e-3/(lam*1e-9*D))**2;s1,_=single(lam,a,D)
f,a1,a2=two();strip(a1,xs,I,lam);a2.plot(xs,I,color=NAVY,lw=1.2);a2.axvline(s1,color=OR,ls='--',lw=.9);a2.axvline(-s1,color=OR,ls='--',lw=.9)
a2.annotate('',xy=(s1,.55),xytext=(-s1,.55),arrowprops=dict(arrowstyle='<->',color=OR));a2.text(0,.58,f'2y₁ = {2*s1:.2f} mm',ha='center',color=OR)
a2.set_xlabel('y (mm)');a2.set_ylabel('I / I₀');a1.set_title(f'a = 0.1 mm, D = 1 m, λ = 550 nm:  y₁ = {s1:.4f} mm (theory {y1:.4f} mm)',loc='left',pad=4);save(f,'f2')
# F3 grating
f,axs=plt.subplots(2,1,figsize=(6.3,4.3),dpi=220,sharex=True);x=np.linspace(-60,60,6000)
for ax,fr in zip(axs,(.4,.5)):
    y=gI(x,550,600,12,fr);ax.plot(x,y,color=NAVY,lw=.9);ax.plot(x,sinc(PI*fr*1e-3/600*np.sin(rad(x))/(550e-9))**2,'--',color=GR,lw=.8)
    for m_,xx,th,_ in grat(550,600,12,fr):
        ax.axvline(th,color=OR,lw=.5,alpha=.7);ax.axvline(-th,color=OR,lw=.5,alpha=.7)
        ax.text(th,1.03,f'm={m_}'+(' (missing)' if xx is None else ''),ha='center',fontsize=7,color=OR)
    ax.set_ylim(0,1.18);ax.set_ylabel('I / I₀');ax.set_title(f'a/d = {fr}',loc='left',pad=10)
axs[1].set_xlabel('θ (°)');axs[0].text(-59,.9,'dashed: single-slit envelope',fontsize=7,color=GR);f.subplots_adjust(hspace=.5);save(f,'f3')
# F4 biprism case 1
s,t,l,dd,W=bi(550,20,80,30,1.5);L=1e-2*100;xs=np.linspace(-1.6*W,1.6*W,3000);I=np.where(np.abs(xs)<W,np.cos(PI*dd*1e-3*xs*1e-3/(550e-9*L))**2,.5)
f,a1,a2=two();strip(a1,xs,I,550);a2.plot(xs,I,color=NAVY,lw=1);a2.axvspan(-W,W,color=OR,alpha=.08);a2.axhline(.5,color=GR,ls='--',lw=.8)
a2.text(1.58*W,.52,'single-beam background (½)',ha='right',fontsize=7,color=GR);a2.text(0,1.04,f'overlap region ±W = ±{W:.2f} mm',ha='center',color=OR,fontsize=8)
a2.set_ylim(0,1.12);a2.set_xlabel('y (mm)');a2.set_ylabel('I / I₀');a1.set_title(f'β = {s:.5f} mm;  recovered λ = {l:.1f} nm',loc='left',pad=4);save(f,'f4')
# F5 Newton mono + white
f,axs=plt.subplots(1,2,figsize=(6.3,3.4),dpi=220);N_=360;W_=[(400+20*i) for i in range(16)];S=sum(rgbw(w_) for w_ in W_)
for ax,ttl,white in zip(axs,('Monochromatic, λ = 589 nm','White light (400–700 nm)'),(0,1)):
    r=np.linspace(-1,1,N_);X,Y=np.meshgrid(r,r);R=np.hypot(X,Y);rm=np.sqrt(20*589e-9*1.)*1e3;rr=R*rm*1e-3
    if white: img=sum(rgbw(w_)*(np.sin(PI*rr**2/(w_*1e-9*1.))**2)[...,None] for w_ in W_)/S
    else: img=rgbw(589)*(np.sin(PI*rr**2/(589e-9))**2)[...,None]
    img=gm(img);img[R>1]=0;ax.imshow(img,extent=[-rm,rm,-rm,rm]);ax.set_title(ttl);ax.grid(False);ax.set_xlabel('x (mm)')
axs[1].set_yticks([]);f.suptitle('Newton’s rings, R = 1 m (20 rings shown)',fontsize=9,y=.98);save(f,'f5')
# F5b D^2 vs n
a5,b5,l5,(xs5,I5,mn)=newton(589,1.);n=np.arange(1,len(mn)+1);D2=(2*mn)**2
f,ax=plt.subplots(figsize=(4.6,2.9),dpi=220);ax.plot(n,D2,'o',color=OR,ms=3.5,label='Simulated dark rings');ax.plot(n,4*n*589e-9*1.*1e6,'-',color=NAVY,lw=1,label='Theory: Dₙ² = 4nλR')
ax.set_xlabel('ring number n');ax.set_ylabel('Dₙ² (mm²)');ax.set_title(f'Slope gives λ = {l5:.1f} nm',loc='left');ax.legend(frameon=False);save(f,'f5b')
# F6 polarisation
rs=np.random.default_rng(7);Z=rs.standard_normal(90);c_,L_,sg=10,2,2;th=66.5*L_*c_/100;ph=2*np.arange(90.);yy=np.cos(rad(ph-th))**2+.02+sg/100*Z
A=2/90*np.sum(yy*np.cos(rad(2*ph)));B=2/90*np.sum(yy*np.sin(rad(2*ph)));fit=.5*np.degrees(np.arctan2(B,A))
f,ax=plt.subplots(figsize=(6.3,2.9),dpi=220);xx=np.linspace(0,180,500);ax.plot(xx,np.cos(rad(xx-th))**2,color=NAVY,lw=1.2,label='Malus: cos²(φ − θ)');ax.plot(ph,yy,'o',ms=2.5,color=OR,label='90 noisy samples (σ = 2%)')
ax.axvline(fit,color='g',ls='--',lw=.9,label=f'Fitted θ = {fit:.3f}° (true {th:.2f}°)');ax.set_xlabel('analyzer angle φ (°)');ax.set_ylabel('I / I₀');ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.55,1.02));ax.set_ylim(0,1.25);ax.set_xlim(0,180);save(f,'f6')
# ---- MC
def seeded(fn,**kw): mc.rng=np.random.default_rng(2026); return fn(**kw)
def sd(fn,**kw): return seeded(fn,**kw).std(ddof=1)
def pan(name,fn,an,srcs,l0,fnm,col=NAVY,**kw):
    mc.rng=np.random.default_rng(2026);al=fn(**kw);mu,s=al.mean(),al.std(ddof=1);per=[sd(fn,k=tuple(int(i==j) for j in range(len(srcs))),**kw) for i in range(len(srcs))]
    v=np.array(per)**2;f,(a1,a2)=plt.subplots(1,2,figsize=(6.3,2.7),dpi=220,gridspec_kw={'width_ratios':[1.25,1]})
    a1.hist(al,bins=60,density=True,color=NAVY,alpha=.8);xg=np.linspace(mu-4*s,mu+4*s,300);a1.plot(xg,np.exp(-(xg-mu)**2/2/s**2)/s/np.sqrt(2*PI),color=OR,lw=1.2,label='Gaussian, same mean and SD')
    a1.axvline(l0,color='k',ls='--',lw=.8,label='true λ');a1.set_xlabel('λ measured (nm)');a1.set_ylabel('probability density');a1.set_title(f'{len(al):,} virtual experiments',loc='left');a1.legend(frameon=False,fontsize=6.5,loc='upper left')
    a2.barh(range(len(srcs)),per,color=OR);a2.set_yticks(range(len(srcs)));a2.set_yticklabels(srcs,fontsize=7);a2.invert_yaxis();a2.set_xlabel('SD from this source alone (nm)')
    for i,p in enumerate(per):a2.text(p,i,f' {p:.2f} nm ({100*v[i]/v.sum():.0f}%)',va='center',fontsize=6.8)
    a2.set_xlim(0,max(per)*1.7);a2.set_title(f'Total SD {s:.2f} nm ({s/mu*100:.2f}%)',loc='left');f.tight_layout();save(f,fnm)
    return mu,s,per,v/v.sum()
R={}
R['N']=pan('N',mc.newt,mc.newt_an,['Microscope readings','Lens radius R'],589,'f7');R['B']=pan('B',mc.bip,mc.bip_an,['Fringe width','Image separations','Distance D'],589,'f8');R['G']=pan('G',mc.gr,mc.gr_an,['Angle readings','Lines per mm'],546.1,'f9')
for k,v in R.items():print(k,'mean %.1f sd %.2f (%.2f%%)'%(v[0],v[1],v[1]/v[0]*100),np.round(v[2],2),np.round(v[3]*100))
# sweeps (common random numbers)
P=[1,2,3,5,7,10,15,20,25];sp=[sd(mc.newt,p=p)/589*100 for p in P];an=[np.hypot(*mc.newt_an(p=p))/589*100 for p in P]
print('p',[round(sd(mc.newt,p=p)/589*100,2) for p in (1,2,5,10,20)]);print('n1,n20',[round(sd(mc.newt,n=n)/589*100,2) for n in (1,20)])
NN=[5,10,20,40];MM=[1.5,2,3,5];print('N',[round(sd(mc.bip,N=n)/589*100,2) for n in NN]);print('m',[round(sd(mc.bip,m=m)/589*100,2) for m in MM])
print('grat n',[round(sd(mc.gr,n=n)/546.1*100,3) for n in (1,2,3)],'angle only',sd(mc.gr,k=(1,0)),sd(mc.gr,k=(1,0))/546.1*100)
f,(a1,a2,a3)=plt.subplots(1,3,figsize=(6.5,2.5),dpi=220)
a1.plot(P,sp,'o',color=OR,ms=3,label='Monte Carlo');a1.plot(P,an,'-',color=NAVY,lw=1,label='Analytic');a1.axhline(1,color=GR,ls='--',lw=.8);a1.text(25,1.1,'radius floor 1%',ha='right',fontsize=6.5,color=GR);a1.set_xlabel('rings between, p');a1.set_ylabel('spread of λ (%)');a1.set_title('Newton’s rings',loc='left');a1.legend(frameon=False,fontsize=6.5,loc='upper left');a1.set_ylim(0,7)
Nn=np.arange(3,41);a2.plot(Nn,[np.hypot(*(np.array(mc.bip_an(N=n))*1))/589*100 if False else np.sqrt(np.sum(np.square(mc.bip_an(N=n))))/589*100 for n in Nn],color=NAVY,lw=1);a2.plot(NN,[sd(mc.bip,N=n)/589*100 for n in NN],'o',color=OR,ms=3);a2.set_xlabel('fringes counted, N');a2.set_title('Biprism: N',loc='left')
Mm=np.linspace(1.5,5,30);a3.plot(Mm,[np.sqrt(np.sum(np.square(mc.bip_an(m=m))))/589*100 for m in Mm],color=NAVY,lw=1);a3.plot(MM,[sd(mc.bip,m=m)/589*100 for m in MM],'o',color=OR,ms=3);a3.set_xlabel('image magnification, m');a3.set_title('Biprism: m',loc='left')
f.tight_layout();save(f,'f10')
# F11 polarimeter noise study
pr=np.random.default_rng(11);S_=[0,1,2,5,10];Mx=2000;thv=66.5*2*.1;res=[]
for sg in S_:
    e=[]
    for _ in range(Mx):
        z=pr.standard_normal(90);y=np.cos(rad(ph-thv))**2+.02+sg/100*z;a_=2/90*np.sum(y*np.cos(rad(2*ph)));b_=2/90*np.sum(y*np.sin(rad(2*ph)));t=.5*np.degrees(np.arctan2(b_,a_));e.append(t-thv)
    e=np.array(e);res.append((sg,np.sqrt(np.mean(e**2)),np.abs(e).max(),np.degrees(np.sqrt(2/90))*sg/100,np.sqrt(np.mean(e**2))/(66.5*2)*100))
for r_ in res:print('pol',[round(x,3) for x in r_])
f,ax=plt.subplots(figsize=(4.6,2.9),dpi=220);ax.plot(S_,[r_[1] for r_ in res],'o',color=OR,label=f'RMS error ({Mx:,} fits each)');xl=np.linspace(0,10,50);ax.plot(xl,np.degrees(np.sqrt(2/90))*xl/100,color=NAVY,label='Prediction √(2/N)·σ')
ax.set_xlabel('detector noise σ (%)');ax.set_ylabel('error in θ (°)');ax.legend(frameon=False);save(f,'f11')
