import numpy as np
rng=np.random.default_rng(2026); T=40000
def rn(): return rng.standard_normal(T)
def newt(n=5,p=5,sx=10,sr=1,k=(1,1),l0=589):
    kk=np.sqrt(l0*1e-9); a=2*kk*np.sqrt(n); b=2*kk*np.sqrt(n+p); s=np.sqrt(2)*sx*1e-6
    x=a+s*rn()*k[0]; y=b+s*rn()*k[0]; R=1+sr/100*rn()*k[1]; return (y*y-x*x)/(4*p*R)*1e9
def newt_an(n=5,p=5,sx=10,sr=1,l0=589):
    kk=np.sqrt(l0*1e-9); a=2*kk*np.sqrt(n); b=2*kk*np.sqrt(n+p); s=np.sqrt(2)*sx*1e-6
    return l0*2*s*np.hypot(a,b)/(b*b-a*a), l0*sr/100
def bip(N=10,m=2,sx=10,sD=2,k=(1,1,1),l0=589):
    d=1.745;D=1000;b=l0*1e-6*D/d;d1=m*d;d2=d/m;s=np.sqrt(2)*sx*1e-3
    bb=b+s/N*rn()*k[0]; e1=d1+s*rn()*k[1]; e2=d2+s*rn()*k[1]; DD=D+sD*rn()*k[2]; return bb*np.sqrt(e1*e2)/DD*1e6
def bip_an(N=10,m=2,sx=10,sD=2,l0=589):
    d=1.745;D=1000;b=l0*1e-6*D/d;d1=m*d;d2=d/m;s=np.sqrt(2)*sx*1e-3
    return l0*s/N/b, l0*.5*np.hypot(s/d1,s/d2), l0*sD/D
def gr(n=1,N=600,sa=1,sN=.5,k=(1,1),l0=546.1):
    th0=np.arcsin(min(.999,n*l0*1e-9*N*1e3)); th=th0+np.sqrt(.5)*sa/60*np.pi/180*rn()*k[0]; NN=N*1e3*(1+sN/100*rn()*k[1]); return np.sin(th)/(n*NN)*1e9
def gr_an(n=1,N=600,sa=1,sN=.5,l0=546.1):
    th0=np.arcsin(n*l0*1e-9*N*1e3); return l0*np.sqrt(.5)*sa/60*np.pi/180/np.tan(th0), l0*sN/100
def rep(name,f,an,nsrc,**kw):
    al=f(**kw); per=[f(k=tuple(int(i==j) for j in range(nsrc)),**kw).std(ddof=1) for i in range(nsrc)]
    a=an(**kw); tot=float(np.sqrt(np.sum(np.square(a)))); v=np.array(per)**2
    print(name,kw,'mean %.2f sd %.2f (%.2f%%) analytic %.2f'%(al.mean(),al.std(ddof=1),al.std(ddof=1)/al.mean()*100,tot),'per',np.round(per,2),'an',np.round(a,2),'share',np.round(v/v.sum()*100))
    return al.std(ddof=1)/al.mean()*100
if __name__=='__main__':
    rep('N',newt,newt_an,2); rep('B',bip,bip_an,3); rep('G',gr,gr_an,2)
    print('p sweep',[round(rep('N',newt,newt_an,2,p=p) if False else newt(p=p).std()/589*100,2) for p in (1,2,5,10,20)])
    print('n sweep p5',[round(newt(n=n).std()/589*100,2) for n in (1,20)])
    print('N sweep',[round(bip(N=n).std()/589*100,2) for n in (5,10,20,40)])
    print('m sweep',[round(bip(m=m).std()/589*100,2) for m in (1.5,2,3,5)])
    for n in (1,2,3): print('grat n',n,round(gr(n=n).std()/546.1*100,3))
    print('grat angle only',gr(k=(1,0)).std(), gr(k=(1,0)).std()/546.1*100)
