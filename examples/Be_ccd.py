from pyscf import gto, scf, cc
from kelvin.mp3 import mp3
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

mol = gto.M(
    verbose = 0,
    atom = 'Be 0 0 0',
    basis = 'sto-3G')

m = scf.RHF(mol)
scf.conv_tol_grad = 1e-12
m.conv_tol = 1e-12
print('SCF energy: %f' % m.scf())

sys = scf_system(m,0.0,0.0,orbtype='g')
ccsd0 = ccsd(sys,iprint=1,max_iter=14,econv=1e-10,singles=False)
ccsd0.run()

T = 5.0
mu = 0.0
sys = scf_system(m,T,mu,orbtype='g')
mp3T = mp3(sys,iprint=1,T=T,mu=mu)
E0T,E1T,E2T,E3T = mp3T.run()
print('HF energy: %.8f' % (E0T + E1T))
print('MP3 correlation energy: %.8f' % (E2T + E3T))

ccsdT = ccsd(sys,iprint=1,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=10,singles=False)
Ecctot,Ecc = ccsdT.run()

print(E2T,E2T+E3T)
print(Ecc)
