import unittest
import numpy
from pyscf import gto, scf
from kelvin.rt_ccsd import RTCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class RTCCSD1RDMTest(unittest.TestCase):
    def test_Be_rk1(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=40,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk1")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - rtccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - rtccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - rtccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - rtccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3,erroria)
        self.assertTrue(eji < 1e-3,errorji)
        self.assertTrue(eba < 1e-3,errorba)
        self.assertTrue(eai < 1e-3,errorai)

    def test_Be_rk2(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=40,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk2")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - rtccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - rtccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - rtccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - rtccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3,erroria)
        self.assertTrue(eji < 1e-3,errorji)
        self.assertTrue(eba < 1e-3,errorba)
        self.assertTrue(eai < 1e-3,errorai)

    def test_Be_rk4(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk4")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - rtccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - rtccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - rtccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - rtccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-5,erroria)
        self.assertTrue(eji < 1e-5,errorji)
        self.assertTrue(eba < 1e-5,errorba)
        self.assertTrue(eai < 1e-5,errorai)

    #def test_Be_active(self):
    #    mol = gto.M(
    #        verbose = 0,
    #        atom = 'Be 0 0 0',
    #        basis = 'sto-3G')

    #    m = scf.RHF(mol)
    #    m.conv_tol = 1e-12
    #    Escf = m.scf()
    #    T = 0.05
    #    mu = 0.0
    #    sys = scf_system(m,T,mu,orbtype='g')

    #    # compute normal-ordered 1-rdm 
    #    ccsdT = ccsd(sys,T=T,mu=mu,ngrid=100,iprint=1,damp=0.4,athresh = 1e-20)
    #    Eref,Eccref = ccsdT.run()
    #    ccsdT._g_ft_1rdm()

    #    # compute normal-order 1-rdm from propagation
    #    rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk4", athresh = 1e-20)
    #    Eout,Eccout = rtccsdT.run()
    #    print(Eout, Eccout)
    #    Etmp,Ecctmp = rtccsdT._ccsd_lambda()
    #    eia = numpy.linalg.norm(ccsdT.dia - rtccsdT.dia)
    #    eji = numpy.linalg.norm(ccsdT.dji - rtccsdT.dji)
    #    eba = numpy.linalg.norm(ccsdT.dba - rtccsdT.dba)
    #    eai = numpy.linalg.norm(ccsdT.dai - rtccsdT.dai)
    #    print(eia)
    #    print(eji)
    #    print(eba)
    #    print(eai)
    #    erroria = "Difference in pia: {}".format(eia)
    #    errorji = "Difference in pji: {}".format(eji)
    #    errorba = "Difference in pba: {}".format(eba)
    #    errorai = "Difference in pai: {}".format(eai)
    #    self.assertTrue(eia < 1e-4,erroria)
    #    self.assertTrue(eji < 1e-4,errorji)
    #    self.assertTrue(eba < 1e-4,errorba)
    #    self.assertTrue(eai < 1e-4,errorai)

if __name__ == '__main__':
    unittest.main()
