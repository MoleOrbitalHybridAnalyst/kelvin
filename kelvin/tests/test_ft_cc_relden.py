import unittest
import numpy
from pyscf import gto, scf
from cqcpy import utils
from lattice.hubbard import Hubbard1D
from kelvin.ccsd import ccsd
from kelvin.hubbard_system import HubbardSystem
from kelvin.scf_system import scf_system
from kelvin.ueg_system import ueg_system
from kelvin.ueg_scf_system import ueg_scf_system
from kelvin.pueg_system import pueg_system
from kelvin import scf_utils

class FTCCReldenTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-14

    def test_Be_gen(self):
        T = 0.8
        mu = 0.04
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0)
        cc = ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_Be_gen_prop(self):
        T = 2.0
        mu = 0.0
        thresh = 5e-7
        ngrid = 10
        damp = 0.1
        mi = 100
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        h = m.get_hcore()
        field = numpy.random.random(h.shape)
        field = 0.1*(field + field.transpose((1,0)))
        fmo = scf_utils.mo_tran_1e(m, field)
        fdiag = fmo.diagonal()
        na = fmo.shape[0]//2

        alpha = 8e-4
        m.get_hcore = lambda *args: h + alpha*field
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        ccf = ccsdT.run()

        m.get_hcore = lambda *args: h - alpha*field
        m.mo_energy -= 2.0*alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        ccb = ccsdT.run()

        ref = (ccf[0] - ccb[0])/(2*alpha)

        m.get_hcore = lambda *args: h
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()
        Dm = ccsdT.r1rdm + (ccsdT.n1rdm - numpy.diag(ccsdT.n1rdm.diagonal()))
        out = numpy.einsum('ij,ji->',Dm,fmo)
        diff = abs(out - ref)

        error = "Expected: {}  Actual: {}".format(ref,out)
        self.assertTrue(diff < thresh,error)

    def test_Be_gen_active(self):
        T = 0.02
        mu = 0.0
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=80,athresh=1e-30,iprint=0)
        cc = ccsdT.run()
        ccsdT.compute_ESN() 
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_Be_gen_prop_active(self):
        T = 0.02
        mu = 0.0
        thresh = 5e-7
        ngrid = 80
        mi = 100
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        h = m.get_hcore()
        field = numpy.random.random(h.shape)
        field = 0.1*(field + field.transpose((1,0)))
        fmo = scf_utils.mo_tran_1e(m, field)
        fdiag = fmo.diagonal()
        na = fmo.shape[0]//2

        alpha = 4e-4
        m.get_hcore = lambda *args: h + alpha*field
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        ccf = ccsdT.run()
        G0f = ccsdT.G0
        G1f = ccsdT.G1

        m.get_hcore = lambda *args: h - alpha*field
        m.mo_energy -= 2.0*alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        ccb = ccsdT.run()
        G0b = ccsdT.G0
        G1b = ccsdT.G1

        ref = (ccf[0] - ccb[0])/(2*alpha)
        Acc = (ccf[1] - ccb[1])/(2*alpha)
        A01 = (ccf[0] - ccf[1] - ccb[0] + ccb[1])/(2*alpha)
        A0 = (G0f - G0b)/(2*alpha)
        A1 = (G1f - G1b)/(2*alpha)

        m.get_hcore = lambda *args: h
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()
        Dm = ccsdT.r1rdm + (ccsdT.n1rdm - numpy.diag(ccsdT.n1rdm.diagonal()))
        out = numpy.einsum('ij,ji->',Dm,fmo)
        diff = abs(out - ref)
        error = "Expected: {}  Actual: {}".format(ref,out)
        self.assertTrue(diff < thresh,error)

    def test_pueg(self):
        T = 0.5
        mu = 0.2
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 19
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = pueg_system(T,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_ueg_gen(self):
        T = 0.5
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_ueg_scf_gen(self):
        T = 0.5
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = ueg_scf_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_ueg(self):
        T = 0.5
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        econv = 1e-12
        thresh = 1e-12
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,econv=econv,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT._grel_ft_1rdm()

        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
        uccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,econv=econv,damp=damp,ngrid=10)
        uccsdT.run()
        uccsdT._urel_ft_1rdm()

        # compare relaxed 1rdm
        daout,dbout = uccsdT.r1rdm
        naout,nbout = uccsdT.n1rdm
        dref = ccsdT.r1rdm
        nref = ccsdT.n1rdm
        daref = dref[:norb,:norb]
        dbref = dref[norb:,norb:]
        naref = nref[:norb,:norb]
        nbref = nref[norb:,norb:]
        diffa = numpy.linalg.norm(daref - daout)/numpy.sqrt(daref.size)
        diffb = numpy.linalg.norm(dbref - dbout)/numpy.sqrt(dbref.size)
        self.assertTrue(diffa < thresh, "Error in relaxed alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in relaxed beta rdm: {}".format(diffb))
        diffa = numpy.linalg.norm(naref - naout)/numpy.sqrt(naref.size)
        diffb = numpy.linalg.norm(nbref - nbout)/numpy.sqrt(nbref.size)
        self.assertTrue(diffa < thresh, "Error in normal-ordered alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in normal-ordered beta rdm: {}".format(diffb))

    def test_ueg_scf(self):
        T = 0.5
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        econv = 1e-12
        thresh = 1e-12
        ueg = ueg_scf_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,econv=econv,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT._grel_ft_1rdm()

        ueg = ueg_scf_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
        uccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,econv=econv,damp=damp,ngrid=10)
        uccsdT.run()
        uccsdT._urel_ft_1rdm()

        # compare relaxed 1rdm
        daout,dbout = uccsdT.r1rdm
        naout,nbout = uccsdT.n1rdm
        dref = ccsdT.r1rdm
        nref = ccsdT.n1rdm
        daref = dref[:norb,:norb]
        dbref = dref[norb:,norb:]
        naref = nref[:norb,:norb]
        nbref = nref[norb:,norb:]
        diffa = numpy.linalg.norm(daref - daout)/numpy.sqrt(daref.size)
        diffb = numpy.linalg.norm(dbref - dbout)/numpy.sqrt(dbref.size)
        self.assertTrue(diffa < thresh, "Error in relaxed alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in relaxed beta rdm: {}".format(diffb))
        diffa = numpy.linalg.norm(naref - naout)/numpy.sqrt(naref.size)
        diffb = numpy.linalg.norm(nbref - nbout)/numpy.sqrt(nbref.size)
        self.assertTrue(diffa < thresh, "Error in normal-ordered alpha rdm: {}".format(diffa))

    def test_Be_active(self):
        T = 0.02
        mu = 0.0
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        thresh = 1e-11
        ethresh = 1e-12
        mi = 100
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=80,athresh=1e-30,iprint=0,econv=ethresh,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()

        sys = scf_system(m,T,mu,orbtype='u')
        uccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=80,athresh=1e-30,iprint=0,econv=ethresh,max_iter=mi)
        ucc = uccsdT.run()
        uccsdT._urel_ft_1rdm()

        # compare relaxed 1rdm
        daout,dbout = uccsdT.r1rdm
        naout,nbout = uccsdT.n1rdm
        norb = daout.shape[0]
        dref = ccsdT.r1rdm
        nref = ccsdT.n1rdm
        daref = dref[:norb,:norb]
        dbref = dref[norb:,norb:]
        naref = nref[:norb,:norb]
        nbref = nref[norb:,norb:]
        diffa = numpy.linalg.norm(daref - daout)/numpy.sqrt(daref.size)
        diffb = numpy.linalg.norm(dbref - dbout)/numpy.sqrt(dbref.size)
        self.assertTrue(diffa < thresh, "Error in relaxed alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in relaxed beta rdm: {}".format(diffb))
        diffa = numpy.linalg.norm(naref - naout)/numpy.sqrt(naref.size)
        diffb = numpy.linalg.norm(nbref - nbout)/numpy.sqrt(nbref.size)
        self.assertTrue(diffa < thresh, "Error in normal-ordered alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in normal-ordered beta rdm: {}".format(diffb))

    def test_Beplus(self):
        T = 0.5
        mu = 0.0
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        mol.charge = 1
        mol.spin = 1

        thresh = 1e-11
        ethresh = 1e-12
        mi = 100
        m = scf.UHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=40,iprint=0,econv=ethresh,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()

        sys = scf_system(m,T,mu,orbtype='u')
        uccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=40,iprint=0,econv=ethresh,max_iter=mi)
        ucc = uccsdT.run()
        uccsdT._urel_ft_1rdm()

        # compare relaxed 1rdm
        daout,dbout = uccsdT.r1rdm
        naout,nbout = uccsdT.n1rdm
        norb = daout.shape[0]
        dref = ccsdT.r1rdm
        nref = ccsdT.n1rdm
        daref = dref[:norb,:norb]
        dbref = dref[norb:,norb:]
        naref = nref[:norb,:norb]
        nbref = nref[norb:,norb:]
        diffa = numpy.linalg.norm(daref - daout)/numpy.sqrt(daref.size)
        diffb = numpy.linalg.norm(dbref - dbout)/numpy.sqrt(dbref.size)
        self.assertTrue(diffa < thresh, "Error in relaxed alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in relaxed beta rdm: {}".format(diffb))
        diffa = numpy.linalg.norm(naref - naout)/numpy.sqrt(naref.size)
        diffb = numpy.linalg.norm(nbref - nbout)/numpy.sqrt(nbref.size)
        self.assertTrue(diffa < thresh, "Error in normal-ordered alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in normal-ordered beta rdm: {}".format(diffb))

    def test_hubbard(self):
        T = 0.5
        L = 2
        U = 2.0
        mu = 0.0
        damp = 0.2
        mi = 50
        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        hub = Hubbard1D(L,1.0,U,boundary='c')
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='u')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=10)
        ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._urel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm[0])
        Nout += numpy.trace(ccsdT.r1rdm[1])
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_Hubbard_gu(self):
        T = 1.0
        L = 2
        U = 2.0
        mu = 0.0
        damp = 0.2
        mi = 50
        ethresh = 1e-11
        thresh = 1e-11
        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        hub = Hubbard1D(L,1.0,U,boundary='c')
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,damp=damp,ngrid=10,econv=ethresh,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()

        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='u')
        uccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=10,iprint=0,econv=ethresh,max_iter=mi)
        ucc = uccsdT.run()
        uccsdT._urel_ft_1rdm()

        # compare relaxed 1rdm
        daout,dbout = uccsdT.r1rdm
        naout,nbout = uccsdT.n1rdm
        norb = daout.shape[0]
        dref = ccsdT.r1rdm
        nref = ccsdT.n1rdm
        daref = dref[:norb,:norb]
        dbref = dref[norb:,norb:]
        naref = nref[:norb,:norb]
        nbref = nref[norb:,norb:]
        diffa = numpy.linalg.norm(daref - daout)/numpy.sqrt(daref.size)
        diffb = numpy.linalg.norm(dbref - dbout)/numpy.sqrt(dbref.size)
        self.assertTrue(diffa < thresh, "Error in relaxed alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in relaxed beta rdm: {}".format(diffb))
        diffa = numpy.linalg.norm(naref - naout)/numpy.sqrt(naref.size)
        diffb = numpy.linalg.norm(nbref - nbout)/numpy.sqrt(nbref.size)
        self.assertTrue(diffa < thresh, "Error in normal-ordered alpha rdm: {}".format(diffa))
        self.assertTrue(diffb < thresh, "Error in normal-ordered beta rdm: {}".format(diffb))

if __name__ == '__main__':
    unittest.main()