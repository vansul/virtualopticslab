import numpy as np
PI=np.pi; rad=np.radians
def sinc(v): return np.where(np.abs(v)<1e-9,1.0,np.sin(v)/np.where(v==0,1,v))
def ext(xs,ys,mx):
    a,b,c=ys[:-2],ys[1:-1],ys[2:]
    ok=((b>a)&(b>=c)) if mx else ((b<a)&(b<=c))
    i=np.where(ok)[0]+1; A,B,C=ys[i-1],ys[i],ys[i+1]; d=A-2*B+C
    dx=xs[1]-xs[0]; off=np.where(d!=0,.5*(A-C)/np.where(d==0,1,d),0)
    return xs[i]+off*dx, B
def gap(x): return None if len(x)<3 else (x[-1]-x[0])/(len(x)-1)
# double slit
def young(lam,d,a,D):
    b=lam*1e-9*D/(d*1e-3)*1e3; xs=np.linspace(-8*b,8*b,2000); k=xs*1e-3/(lam*1e-9*D); aa=min(a,.9*d)
    I=np.cos(PI*d*1e-3*k)**2*sinc(PI*aa*1e-3*k)**2; px,py=ext(xs,I,1); return gap(px[py>.1]),b
def single(lam,a,D):
    y=lam*1e-9*D/(a*1e-3)*1e3; xs=np.linspace(-4.5*y,4.5*y,2000); I=sinc(PI*a*1e-3*xs*1e-3/(lam*1e-9*D))**2
    px,py=ext(xs,I,0); m=(py<1e-3)&(px>0); return px[m][0],y
def gI(x,lam,lpm,N,f):
    s=np.sin(rad(x)); d=1e-3/lpm; dl=PI*d*s/(lam*1e-9); sn=np.sin(dl)
    F=np.where(np.abs(sn)<1e-9,N*N,(np.sin(N*dl)/np.where(sn==0,1,sn))**2)
    return sinc(PI*f*d*s/(lam*1e-9))**2*F/(N*N)
def grat(lam,lpm,N,f):
    d=1e-3/lpm;out=[]
    for m in (1,2,3):
        s=m*lam*1e-9/d
        if s>=1:break
        th=np.degrees(np.arcsin(s)); x=np.arange(th-2,th+2+1e-9,.005); y=gI(x,lam,lpm,N,f); j=np.argmax(y)
        out.append((m,None if y[j]<.02 else x[j],th,y[j]))
    return out
def bi(lam,a,b,al,n):
    alr=al/60*PI/180; d=2*a*1e-2*(n-1)*alr; L=(a+b)*1e-2; W=b*1e-2*(n-1)*alr*1e3
    xs=np.linspace(-1.6*W,1.6*W,3000); I=np.where(np.abs(xs)<W,np.cos(PI*d*xs*1e-3/(lam*1e-9*L))**2,.5)
    px,py=ext(xs,I,1); m=(np.abs(px)<W*.97)&(py>.8); s=gap(px[m]); return s,lam*1e-9*L/d*1e3,s*1e-3*d/L*1e9,d*1e3,W
def newton(lam,R):
    xs=np.linspace(0,np.sqrt(20*lam*1e-9*R)*1e3,4000); I=np.sin(PI*(xs*1e-3)**2/(lam*1e-9*R))**2
    mn,_=ext(xs,I,0); n=len(mn); k=np.arange(1,n+1); D=(2*mn)**2; sl=np.polyfit(k,D,1)[0]
    return 2*mn[4],2*np.sqrt(5*lam*1e-9*R)*1e3,sl*1e-6/(4*R)*1e9,(xs,I,mn)
