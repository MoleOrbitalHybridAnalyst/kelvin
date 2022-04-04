import logging
import numpy
import time
from pyscf import lib
from cqcpy import cc_energy
from cqcpy import cc_equations
from cqcpy import ft_utils
from cqcpy import utils
from . import zt_mp
from . import ft_mp
from . import cc_utils
from . import ft_cc_energy
from . import ft_cc_equations
from . import quadrature

einsum = lib.einsum

class ccsd(object):
    ''' CCSD with finite temperature smearing

    Attributes:
        sys: System object
        T (float): Temperature.
        mu (float): Chemical potential.
        iprint (int): Print level.
        singles (bool): Include singles (False -> CCD).
        econv (float): Energy difference convergence threshold.
        tconv (float): Amplitude difference convergence threshold.
        max_iter (int): Max number of iterations.
        damp (float): Mixing parameter to damp iterations.
        dt (float): Time step in imag. time form for a single point
        athresh (float): Threshold for ignoring small occupations
        T1: Saved T1 amplitudes
        T2: Saved T2 amplitudes
        L1: Saved L1 amplitudes
        L2: Saved L2 amplitudes
    '''
    def __init__(self, sys, T=0.0, mu=0.0, iprint=0,
                 singles=True, econv=1e-8, tconv=None, max_iter=40,
                 damp=0.0, athresh=0.0, dt=None, degcr = 1E-12):

        self.T = T
        self.mu = mu
        assert T > 0
        self.finite_T = True
        self.iprint = iprint
        self.singles = singles
        self.econv = econv
        self.tconv = tconv if tconv is not None else 1000.0*econv
        self.max_iter = max_iter
        self.damp = damp
        self.athresh = athresh
        self.dt = dt
        self.degcr = degcr
        if not sys.verify(self.T, self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        self.beta = None
        self.beta = 1.0 / T
        self.sys = sys
        # amplitudes
        self.T1 = None
        self.T2 = None
        self.L1 = None
        self.L2 = None
        # pieces of normal-ordered 1-rdm
        self.dia = None
        self.dba = None
        self.dji = None
        self.dai = None
        # occupation number response
        self.rono = None
        self.ronv = None
        self.ron1 = None
        # orbital energy response
        self.rorbo = None
        self.rorbv = None
        # pieces of 1-rdm with ONs
        self.ndia = None
        self.ndba = None
        self.ndji = None
        self.ndai = None
        # pieces of normal-ordered 2-rdm
        self.P2 = None
        # full unrelaxed 1-rdm
        self.n1rdm = None
        # full unrelaxed 2-rdm
        self.n2rdm = None
        # ON- and OE-relaxation contribution to 1-rdm
        self.r1rdm = None

    def run(self, T1=None, T2=None, D1corr=None, D2corr=None, scaleI=True):
        if self.finite_T:
            logging.info('Running CCSD at an electronic temperature of %f K'
                % ft_utils.HtoK(self.T))
            if self.sys.has_u():
                return self._ft_uccsd(T1in=T1, T2in=T2, D1corr=D1corr, D2corr=D2corr, scaleI=scaleI)
            else:
                return self._ft_ccsd(T1in=T1, T2in=T2)
        else:
            pass

    def compute_ESN(self, L1=None, L2=None, gderiv=True):
        raise NotImplementedError()

    def _g_ft_ESN(self, L1=None, L2=None, gderiv=True):
        raise NotImplementedError()

    def _u_ft_ESN(self, L1=None, L2=None, gderiv=True):
        raise NotImplementedError()

    def _ft_ccsd(self, T1in=None, T2in=None):
        raise NotImplementedError()

    def _ft_uccsd(self, T1in=None, T2in=None, D1corr=None, D2corr=None, scaleI=True):
        '''
        CCSD with FT smearing
        '''
        if self.finite_T:
            beta = self.beta
            mu = self.mu if self.finite_T else None

            # get orbital energies (not scaled)
            ea, eb = self.sys.u_energies_tot()
            na = ea.shape[0]
            nb = eb.shape[0]
            # ev - eo
            D1a = utils.D1(ea, ea)
            D1b = utils.D1(eb, eb)
            D2aa = utils.D2(ea, ea)
            D2ab = utils.D2u(ea, eb, ea, eb)
            D2bb = utils.D2(eb, eb)

            # get 0th and 1st order contributions
            En = self.sys.const_energy()
            g0 = ft_utils.uGP0(beta, ea, eb, mu)
            E0 = ft_mp.ump0(g0[0], g0[1]) + En
            E1 = self.sys.get_mp1()
            E01 = E0 + E1

            if scaleI:
                if self.athresh > 0.0:
                    raise NotImplementedError()
                else:
                    # get FT-scaled integrals
                    Fa, Fb, Ia, Ib, Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)
            else:
                from cqcpy.ov_blocks import one_e_blocks,two_e_blocks,two_e_blocks_full
                foa = ft_utils.ff(beta, ea, mu)
                fob = ft_utils.ff(beta, eb, mu)
                maskoa = foa > self.degcr
                maskob = fob > self.degcr
                maskva = ~maskoa
                maskvb = ~maskob

                Fa, Fb = self.sys.u_fock_tot()
                Fa = Fa - numpy.diag(ea)
                Fb = Fb - numpy.diag(eb)
                Fa = one_e_blocks(None, Fa, Fa, None)
                Fb = one_e_blocks(None, Fb, Fb, None)

                eriA, eriB, eriAB = self.sys.u_aint_tot()
                Ioovv = numpy.zeros_like(eriA)
                Ivvoo = numpy.zeros_like(eriA)
                Ioovv[numpy.ix_(maskoa,maskoa,maskva,maskva)] = \
                        eriA[numpy.ix_(maskoa,maskoa,maskva,maskva)]
                Ivvoo[numpy.ix_(maskva,maskva,maskoa,maskoa)] = \
                        eriA[numpy.ix_(maskva,maskva,maskoa,maskoa)]
                Ia = two_e_blocks(
                    vvvv=None, vvvo=None, vovv=None, vvoo=Ivvoo,
                    vovo=None, oovv=Ioovv, vooo=None, ooov=None, oooo=None)
                Ioovv = numpy.zeros_like(eriB)
                Ivvoo = numpy.zeros_like(eriB)
                Ioovv[numpy.ix_(maskob,maskob,maskvb,maskvb)] = \
                        eriB[numpy.ix_(maskob,maskob,maskvb,maskvb)]
                Ivvoo[numpy.ix_(maskvb,maskvb,maskob,maskob)] = \
                        eriB[numpy.ix_(maskvb,maskvb,maskob,maskob)]
                Ib = two_e_blocks(
                    vvvv=None, vvvo=None, vovv=None, vvoo=Ivvoo,
                    vovo=None, oovv=Ioovv, vooo=None, ooov=None, oooo=None)
                Ioovv = numpy.zeros_like(eriAB)
                Ivvoo = numpy.zeros_like(eriAB)
                Ioovv[numpy.ix_(maskoa,maskob,maskva,maskvb)] = \
                        eriAB[numpy.ix_(maskoa,maskob,maskva,maskvb)]
                Ivvoo[numpy.ix_(maskva,maskvb,maskoa,maskob)] = \
                        eriAB[numpy.ix_(maskva,maskvb,maskoa,maskob)]
                Iabab = two_e_blocks_full(
                        vvvv=None, vvvo=None, vvov=None,
                        vovv=None, ovvv=None, vvoo=Ivvoo,
                        vovo=None, ovvo=None, voov=None,
                        ovov=None, oovv=Ioovv, vooo=None,
                        ovoo=None, oovo=None, ooov=None,
                        oooo=None)

        else:
            pass

        method = "CCSD" if self.singles else "CCD"
        conv_options = {
                "econv": self.econv,
                "tconv": self.tconv,
                "max_iter": self.max_iter,
                "damp": self.damp,
                "dt": self.dt}
        if True: # if self.rt_iter[0] == 'a' or T2in is not None:
            # initialize T-amplitudes
            if T1in is not None and T2in is not None:
                if self.singles:
                    T1aold = T1in[0]
                    T1bold = T1in[1]
                    T2aaold = T2in[0]
                    T2abold = T2in[1]
                    T2bbold = T2in[2]
                else:
                    # T1 should be set to zero
                    raise NotImplementedError()
            else:
                if self.singles:
                    T1aold = -Fa.vo
                    T1bold = -Fb.vo
                else:
                    raise NotImplementedError()
                logging.info("TODO: MP2 initials with deg. levels?")
                T2aaold = -Ia.vvoo
                T2abold = -Iabab.vvoo
                T2bbold = -Ib.vvoo
                def divide(x, y):
                    mask = numpy.abs(y) > self.degcr
                    d = numpy.zeros_like(x)
                    d[mask] = x[mask] / y[mask]
                    return d
                #T1aold = T1aold / D1a
                #T1bold = T1bold / D1b
                #T2aaold = T2aaold / D2aa
                #T2abold = T2abold / D2ab
                #T2bbold = T2bbold / D2bb

                # @@@@@@@
                if D1corr is not None:
                    if D1corr.ndim == 3:
                        D1a += D1corr[0]
                        D1b += D1corr[1]
                    else:
                        D1a += D1corr
                        D1b += D1corr
                if D2corr is not None:
                    if D2corr.ndim == 5:
                        D2aa += D2corr[0]
                        D2bb += D2corr[1]
                        D2ab += D2corr[2]
                    else:
                        D2aa += D2corr
                        D2bb += D2corr
                        D2ab += D2corr
                # @@@@@@@
        
                T1aold = divide(T1aold, D1a)
                T1bold = divide(T1bold, D1b)
                T2aaold = divide(T2aaold, D2aa)
                T2abold = divide(T2abold, D2ab)
                T2bbold = divide(T2bbold, D2bb)

            # MP2 energy
            E2 = ft_cc_energy.zt_ucc_energy(
                T1aold, T1bold, T2aaold, T2abold, T2bbold, Fa.ov, Fb.ov,
                Ia.oovv, Ib.oovv, Iabab.oovv, Qterm=False)
            logging.info('MP2 Energy: {:.10f}'.format(E2))

            # @@@@@ use zeros as initials
#            T1aold = numpy.zeros_like(T1aold)
#            T1bold = numpy.zeros_like(T1bold)
#            T2aaold = numpy.zeros_like(T2aaold)
#            T2abold = numpy.zeros_like(T2abold)
#            T2bbold = numpy.zeros_like(T2bbold)
            # @@@@ saving for checking
            self.Fa = Fa
            self.Fb = Fb
            self.Ia = Ia
            self.Ib = Ib
            self.Iabab = Iabab
            self.T1in = [T1aold, T1bold]
            self.T2in = [T2aaold, T2abold, T2bbold]

            # run CC iterations
            if self.max_iter < 0:
                Eccn = E2
                T1 = self.T1in
                T2 = self.T2in
            else:
                Eccn, T1, T2 = cc_utils.zt_ucc_iter(
                    method, T1aold, T1bold, T2aaold, T2abold, T2bbold,
                    Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                    self.iprint, conv_options, degcr=self.degcr)
        else:
            pass

        # @@@@@ save D for check
        self.D1 = (D1a, D1b)
        self.D2 = (D2aa, D2ab, D2bb)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2
        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn

        return (Eccn+E01, Eccn)

