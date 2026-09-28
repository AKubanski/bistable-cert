"""Numerical section for the Opuscula submission. Fully reproducible:
the graph is specified by an explicit edge list printed in the paper."""
import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

BETA = 2.5

a_star = lambda b: np.arccosh(np.sqrt(b)) / b
f      = lambda s, b: np.tanh(b*s) - s
s_SN   = lambda b: np.sqrt(1 - 1/b)
u_SN   = lambda b: s_SN(b) - a_star(b)
C_beta = lambda b: u_SN(b) / (1 + a_star(b))
s_plus = lambda b: brentq(lambda s: f(s, b), a_star(b)+1e-12, 1-1e-15)
k_a    = lambda a, b: b / np.cosh(b*a)**2
gamma_a= lambda a, b: 1 - k_a(a, b)

print("="*70); print("SCALAR CONSTANTS, beta =", BETA); print("="*70)
print(f"  a_*     = {a_star(BETA):.6f}")
print(f"  s_SN    = {s_SN(BETA):.6f}")
print(f"  u_SN    = {u_SN(BETA):.6f}")
print(f"  C_beta  = {C_beta(BETA):.6f}")
print(f"  s_+     = {s_plus(BETA):.6f}")

# ---------------- explicit 7-node weighted graph (printed in the paper) -------
EDGES = [(1,2,0.70),(1,3,0.60),(1,5,0.50),(2,3,0.40),(2,4,0.30),(2,5,0.60),
         (2,6,0.20),(3,4,0.50),(3,5,0.30),(3,6,0.40),(4,7,0.30),(5,6,0.50),
         (5,7,0.40),(6,7,0.60)]
N = 7
W = np.zeros((N,N))
for u,v,w in EDGES:
    W[u-1,v-1] = W[v-1,u-1] = w
L = np.diag(W.sum(1)) - W
assert len(EDGES)==14

sigma  = np.array([-1.,1.,1.,1.,-1.,-1.,-1.])
i0     = 0                      # write node 1 (0-indexed)
sigma2 = sigma.copy(); sigma2[i0] = -sigma[i0]
tau    = sigma2[i0]

def c_vec(sig):
    return np.array([W[k][sig != sig[k]].sum() for k in range(N)])
def q_load(sig): return c_vec(sig).max()

qs, qs2 = q_load(sigma), q_load(sigma2)
Q = max(qs, qs2)
EPS = 0.10
A_MARGIN = 0.65

print("\n" + "="*70); print("GRAPH AND ENDPOINT CERTIFICATES"); print("="*70)
print("  edges:", len(EDGES), " degrees:", np.round(W.sum(1),3))
print(f"  c(sigma)  = {np.round(c_vec(sigma),3)}   q(sigma)  = {qs:.4f}")
print(f"  c(sigma') = {np.round(c_vec(sigma2),3)}   q(sigma') = {qs2:.4f}")
print(f"  Q = {Q:.4f}")
lhs = EPS*(1+A_MARGIN)*Q; rhs = f(A_MARGIN,BETA)
print(f"  eps(1+a)Q = {lhs:.6f}  <  f_beta(a) = {rhs:.6f}   -> {lhs<rhs}")
print(f"  eps*q(sigma)  = {EPS*qs:.6f} < C_beta = {C_beta(BETA):.6f} -> {EPS*qs<C_beta(BETA)}")
print(f"  eps*q(sigma') = {EPS*qs2:.6f} < C_beta -> {EPS*qs2<C_beta(BETA)}")
ga = gamma_a(A_MARGIN,BETA)
print(f"  gamma_a = {ga:.6f}   k_a = {k_a(A_MARGIN,BETA):.6f}")
print(f"  admissible a interval: ({a_star(BETA):.4f}, {s_plus(BETA):.4f})")

# ---------------- pulse margin -----------------------------------------------
d_par  = W[i0][sigma ==  tau].sum()
d_perp = W[i0][sigma == -tau].sum()
d_i    = W[i0].sum()
print(f"\n  d_par = {d_par:.4f}  d_perp = {d_perp:.4f}  d_i = {d_i:.4f}")

def g_i(r,U,a=A_MARGIN,b=BETA,eps=EPS):
    return np.tanh(b*(r+U)) - (1+eps*d_i)*r + eps*(a*d_par - d_perp)
def m_i(U,a=A_MARGIN):
    r = np.linspace(-1,a,400001)
    v = g_i(r,U,a); k = int(np.argmin(v))
    return v[k], r[k]

U = 0.40
m_val, r_arg = m_i(U)
T_cert = (1+A_MARGIN)/m_val
print(f"  U = {U:.2f}   m_1(U,a) = {m_val:.6f} at r = {r_arg:.6f}")
print(f"  certified duration T >= (1+a)/m = {T_cert:.6f}")
m_inf = 1 - A_MARGIN - EPS*(1+A_MARGIN)*d_perp
print(f"  m^inf(a) = 1-a-eps(1+a)d_perp = {m_inf:.6f} > 1-tanh(beta a) = {1-np.tanh(BETA*A_MARGIN):.6f}")

# ---------------- dynamics ----------------------------------------------------
def rhs(t,Z,u):
    x,y = Z[:N],Z[N:]; ut = u(t)
    return np.concatenate([-x+np.tanh(BETA*(y+ut))-EPS*(L@x),
                           -y+np.tanh(BETA*(x+ut))-EPS*(L@y)])
zero_u = lambda t: np.zeros(N)
def pulse_u(T):
    return lambda t: np.where(np.arange(N)==i0, tau*U*(t<=T), 0.0)

KW = dict(rtol=1e-11, atol=1e-13, dense_output=True, max_step=1e-2)

def equilibrium(sig):
    Z = solve_ivp(rhs,[0,300],np.concatenate([sig*0.99]*2),args=(zero_u,),
                  rtol=1e-12,atol=1e-14).y[:,-1]
    return Z[:N]
p_src = equilibrium(sigma); p_tgt = equilibrium(sigma2)
print(f"\n  P_sigma  (diag) = {np.round(p_src,5)}")
print(f"  P_sigma' (diag) = {np.round(p_tgt,5)}")
print(f"  interior check: min |sigma_j p_j| - a = {np.min(sigma*p_src)-A_MARGIN:+.5f}")

J = [j for j in range(N) if j != i0]
L_i = L[np.ix_(J,J)]; w_i = W[J,i0]
A_i = ga*np.eye(N-1) + EPS*L_i
R_i = EPS*np.linalg.solve(A_i, w_i)
print(f"  R_1 = {np.round(R_i,6)}")
print(f"  2R_1 = {np.round(2*R_i,6)}    max 2R/(1-a) = {(2*R_i/(1-A_MARGIN)).max():.4f}")
print(f"  identity check ||R - (1 - gamma_a A^-1 1)||_inf ="
      f" {np.abs(R_i-(1-ga*np.linalg.solve(A_i,np.ones(N-1)))).max():.2e}")

RESULTS = {}

def run_write(Z0, label, Tend_extra=25.0):
    T = T_cert
    sW = solve_ivp(rhs,[0,T+Tend_extra],Z0,args=(pulse_u(T),),**KW)
    sS = solve_ivp(rhs,[0,T+Tend_extra],Z0,args=(zero_u,),**KW)
    tt = np.linspace(0,T,20001); Z = sW.sol(tt); X,Y = Z[:N],Z[N:]
    # B*1 non-target margin during pulse
    nt = min(float((sigma[j]*np.r_[X[j],Y[j]]).min()) for j in J)
    # B*2 target boundedness
    tb = float(max(np.abs(X[i0]).max(), np.abs(Y[i0]).max()))
    # B*3 hitting time on v = min(tau x_i, tau y_i)   [correct variable]
    v = np.minimum(tau*X[i0], tau*Y[i0])
    hit = tt[np.argmax(v>=A_MARGIN)] if (v>=A_MARGIN).any() else np.nan
    capt = float(v[-1])
    # post-pulse convergence
    Zend = sW.sol(T+Tend_extra); res = float(np.abs(Zend[:N]-p_tgt).max())
    # crosstalk: rho (matched no-write reference) and eta (source equilibrium)
    t2 = np.linspace(0,T+Tend_extra,30001)
    ZW,ZS = sW.sol(t2), sS.sol(t2)
    rho = np.maximum(np.abs(ZW[J]-ZS[J]),
                     np.abs(ZW[[j+N for j in J]]-ZS[[j+N for j in J]])).max(1)
    eta = np.maximum(np.abs(ZW[J]-p_src[J][:,None]),
                     np.abs(ZW[[j+N for j in J]]-p_src[J][:,None])).max(1)
    RESULTS[label] = dict(T=T, nontarget_min=nt, target_bound=tb, t_hit=float(hit),
                          frac_used=float(hit/T), capture=capt, post_residual=res,
                          rho=rho.tolist(), eta=eta.tolist(),
                          rho_over_2R=(rho/(2*R_i)).tolist(),
                          eta_over_2R=(eta/(2*R_i)).tolist())
    print(f"\n--- {label} ---")
    print(f"  min_j,t sigma_j s_j during pulse = {nt:.6f}  (floor a = {A_MARGIN})  margin +{nt-A_MARGIN:.6f}")
    print(f"  max |target coord| during pulse  = {tb:.6f}  (<= 1)")
    print(f"  t_hit = {hit:.6f}   t_hit/T = {hit/T:.4f}   capture v(T) = {capt:.6f} >= a")
    print(f"  post-pulse residual at t=T+{Tend_extra:g}: {res:.3e}")
    print(f"  rho_j          = {np.round(rho,6)}")
    print(f"  rho_j / 2R_j   = {np.round(rho/(2*R_i),4)}   max {float((rho/(2*R_i)).max()):.4f}")
    print(f"  eta_j          = {np.round(eta,6)}")
    print(f"  eta_j / 2R_j   = {np.round(eta/(2*R_i),4)}   max {float((eta/(2*R_i)).max()):.4f}")
    return sW, sS

# (i) classical write from the stored source equilibrium (Theorem C regime)
Z_eq = np.concatenate([p_src, p_src])
sW_eq, sS_eq = run_write(Z_eq, "start at P_sigma (diagonal, equilibrium)")

# (ii) full-box adversarial corner start (Theorem B* regime), x and y opposite corners
lo = np.where(sigma>0, A_MARGIN, -1.0); hi = np.where(sigma>0, 1.0, -A_MARGIN)
x0 = lo.copy(); y0 = hi.copy()
x0[i0] = lo[i0] if tau>0 else hi[i0]      # target at its worst corner
y0[i0] = x0[i0]
Z_box = np.concatenate([x0, y0])
sW_bx, sS_bx = run_write(Z_box, "full-box corner start (off-diagonal)")

# ---------------- storage-certificate conservativeness ------------------------
print("\n" + "="*70); print("STORAGE CERTIFICATE VS REDUCED BENCHMARKS"); print("="*70)
# balanced K_8, unit weights, 4+/4- split  => q = 4, reduced branch f(r) = 8 eps r
eps_cert_K8 = C_beta(BETA)/4
def branch_K8(eps):
    # largest r>0 solving f(r) = 8 eps r ; tangency = loss of the symmetric branch
    g = lambda r: f(r,BETA) - 8*eps*r
    rs = np.linspace(1e-6, s_plus(BETA), 200001); v = g(rs)
    sgn = np.sign(v); idx = np.where(np.diff(sgn)!=0)[0]
    return len(idx)
lo_e, hi_e = 0.0, 0.5
for _ in range(200):
    mid = 0.5*(lo_e+hi_e)
    if branch_K8(mid) >= 1: lo_e = mid
    else: hi_e = mid
eps_stab_K8 = 0.5*(lo_e+hi_e)
print(f"  balanced K_8: q = 4,  eps_cert = C_beta/4 = {eps_cert_K8:.6f}")
print(f"                reduced-branch stability loss eps_stab = {eps_stab_K8:.6f}")
print(f"                certificate reaches {100*eps_cert_K8/eps_stab_K8:.1f}% of it")
# large-degree star: hub opposite all leaves, eps = lambda/d, d -> infinity
# hub equation: f(h) = lambda (h + leaf), leaf -> s_+ as d -> infinity
lam_cert = C_beta(BETA)
def star_fold(lam):
    sp = s_plus(BETA)
    g = lambda h: f(h,BETA) - lam*(h + sp)
    hs = np.linspace(1e-6, sp, 200001); v = g(hs)
    return len(np.where(np.diff(np.sign(v))!=0)[0])
lo_l, hi_l = 0.0, 1.0
for _ in range(200):
    mid = 0.5*(lo_l+hi_l)
    if star_fold(mid) >= 1: lo_l = mid
    else: hi_l = mid
lam_fold = 0.5*(lo_l+hi_l)
print(f"  large-degree star: lambda_cert = C_beta = {lam_cert:.6f}")
print(f"                     limiting hub fold lambda_fold = {lam_fold:.6f}")
print(f"                     certificate reaches {100*lam_cert/lam_fold:.1f}% of it")

RESULTS['constants'] = dict(beta=BETA, a_star=a_star(BETA), s_SN=s_SN(BETA),
    u_SN=u_SN(BETA), C_beta=C_beta(BETA), s_plus=s_plus(BETA), a=A_MARGIN,
    gamma_a=ga, eps=EPS, q_sigma=qs, q_sigma2=qs2, Q=Q, U=U, m=m_val, r_arg=r_arg,
    T=T_cert, d_par=d_par, d_perp=d_perp, d_i=d_i, R=R_i.tolist(),
    p_src=p_src.tolist(), p_tgt=p_tgt.tolist(),
    eps_cert_K8=eps_cert_K8, eps_stab_K8=eps_stab_K8,
    lam_cert=lam_cert, lam_fold=lam_fold, edges=EDGES)
json.dump(RESULTS, open('paper_numerics.json','w'), indent=1)
np.save('traj_eq.npy', np.array([sW_eq.sol(np.linspace(0,T_cert+25,4000))]))
print("\nwrote paper_numerics.json")
