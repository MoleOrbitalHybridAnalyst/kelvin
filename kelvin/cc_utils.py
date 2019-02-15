import time
import numpy
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from pyscf import lib
from . import ft_cc_energy
from . import ft_cc_equations
from . import quadrature

einsum = lib.einsum
#einsum = einsum

def form_new_ampl(method, F, I, T1old, T2old, D1, D2, ti, ng, G):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        ti (array): time grid.
        ng (int): number of time points.
        G (array): Quadrature weight matrix.
    """
    if method == "CCSD":
        T1,T2 = ft_cc_equations.ccsd_stanton(F,I,T1old,T2old,
                D1,D2,ti,ng,G)
    elif method == "CCD":
        T1 = T1old
        T2 = ft_cc_equations.ccd_simple(F,I,T2old,
                D2,ti,ng,G)
    elif method == "LCCSD":
        T1,T2 = ft_cc_equations.lccsd_simple(F,I,T1old,T2old,
                D1,D2,ti,ng,G)
    elif method == "LCCD":
        T1 = T1old
        T2 = ft_cc_equations.lccd_simple(F,I,T2old,
                D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")

    return T1,T2

def form_new_ampl_u(method, Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold, T2aaold, T2abold, T2bbold,
        D1a, D1b, D2aa, D2ab, D2bb, ti, ng, G):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        ti (array): time grid.
        ng (int): number of time points.
        G (array): Quadrature weight matrix.
    """
    if method == "CCSD":
        T1out,T2out = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)
    #elif method == "CCD":
    #    T1 = T1old
    #    T2 = ft_cc_equations.ccd_simple(F,I,T2old,
    #            D2,ti,ng,G)
    #elif method == "LCCSD":
    #    T1,T2 = ft_cc_equations.lccsd_simple(F,I,T1old,T2old,
    #            D1,D2,ti,ng,G)
    #elif method == "LCCD":
    #    T1 = T1old
    #    T2 = ft_cc_equations.lccd_simple(F,I,T2old,
    #            D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword for unrestricted calc")

    return T1out,T2out

def form_new_ampl_extrap(ig,method,F,I,T1,T2,T1bar,T2bar,D1,D2,ti,ng,G):
    if method == "CCSD":
        T1,T2 = ft_cc_equations.ccsd_stanton_single(ig,F,I,T1,T2,
                T1bar,T2bar,D1,D2,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")
    return T1,T2

def form_new_ampl_extrap_u(ig,method,Fa,Fb,Ia,Ib,Iabab,
        T1a,T1b,T2aa,T2ab,T2bb,T1bara,T1barb,T2baraa,T2barab,T2barbb,
        D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G):
    if method == "CCSD":
        T1,T2 = ft_cc_equations.uccsd_stanton_single(ig,Fa,Fb,Ia,Ib,Iabab,
                T1a,T1b,T2aa,T2ab,T2bb,T1bara,T1barb,T2baraa,T2barab,T2barbb,
                D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)
    else:
        raise Exception("Unrecognized method keyword")
    return T1,T2

def ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G, beta, ng, ti,
        iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    ethresh = conv_options["econv"]
    tthresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    Eold = 888888888.888888888
    nl1 = numpy.linalg.norm(T1old) + 0.1
    nl2 = numpy.linalg.norm(T2old) + 0.1
    while i < max_iter and not converged:
        # form new T1 and T2
        T1,T2 = form_new_ampl(method,F,I,T1old,T2old,D1,D2,ti,ng,G)

        res1 = numpy.linalg.norm(T1 - T1old) / nl1
        res2 = numpy.linalg.norm(T2 - T2old) / nl2
        # damp new T-amplitudes
        T1old = alpha*T1old + (1.0 - alpha)*T1
        T2old = alpha*T2old + (1.0 - alpha)*T2
        nl1 = numpy.linalg.norm(T1old) + 0.1
        nl2 = numpy.linalg.norm(T2old) + 0.1

        # compute energy
        E = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,g,beta)

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f   %.4E' % (i+1,E,res1+res2))
        i = i + 1
        if numpy.abs(E - Eold) < ethresh and res1+res2 < tthresh:
            converged = True
        Eold = E

    if not converged:
        print("WARNING: {} did not converge!".format(method))

    tend = time.time()
    if iprint > 0:
        print("Total {} time: {:.4f} s".format(method,(tend - tbeg)))

    return Eold,T1,T2

def ft_cc_iter_extrap(method, F, I, D1, D2, g, G, beta, ng, ti,
        iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]

    no,nv = F.ov.shape
    t1bar = numpy.zeros((ng,nv,no))
    t2bar = numpy.zeros((ng,nv,nv,no,no))
    T1new = numpy.zeros((ng,nv,no))
    T2new = numpy.zeros((ng,nv,nv,no,no))

    # loop over grid points
    for ig in range(ng):
        if ig == 0:
            t1bar[0] = -F.vo
            t2bar[0] = -I.vvoo
            continue # don't bother computing at T = inf
        elif ig == 1:
            t1bar[ig] = -F.vo
            t2bar[ig] = -I.vvoo
            T1new[ig] = quadrature.int_tbar1_single(ng,ig,t1bar,ti,D1,G)
            T2new[ig] = quadrature.int_tbar2_single(ng,ig,t2bar,ti,D2,G)
        else:
            # linear extrapolation
            T1new[ig] = T1new[ig - 1] + (T1new[ig - 2] - T1new[ig - 1])\
                    *(ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
            T2new[ig] = T2new[ig - 1] + (T2new[ig - 2] - T2new[ig - 1])\
                    *(ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
        converged = False
        nl1 = numpy.sqrt(float(T1new[ig].size))
        nl2 = numpy.sqrt(float(T2new[ig].size))
        if iprint > 0:
            print("Time point {}".format(ig))
        i = 0
        while i < max_iter and not converged:
            # form new T1 and T2
            T1,T2 = form_new_ampl_extrap(ig,method,F,I,T1new[ig],T2new[ig],
                    t1bar,t2bar,D1,D2,ti,ng,G)

            res1 = numpy.linalg.norm(T1 - T1new[ig]) / nl1
            res2 = numpy.linalg.norm(T2 - T2new[ig]) / nl2
            # damp new T-amplitudes
            T1new[ig] = alpha*T1new[ig] + (1.0 - alpha)*T1.copy()
            T2new[ig] = alpha*T2new[ig] + (1.0 - alpha)*T2.copy()

            # determine convergence
            if iprint > 0:
                print(' %2d  %.4E' % (i+1,res1+res2))
            i = i + 1
            if res1 + res2 < thresh:
                converged = True
    return T1new,T2new

def ft_ucc_iter(method, T1aold, T1bold, T2aaold, T2abold, T2bbold, Fa, Fb, Ia, Ib, Iabab,
        D1a, D1b, D2aa, D2ab, D2bb, g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    ethresh = conv_options["econv"]
    tthresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    na = D1a.shape[0]
    nb = D1b.shape[0]
    n = na + nb
    Eold = 888888888.888888888
    while i < max_iter and not converged:
        T1out,T2out = form_new_ampl_u(method,Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold, D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)

        nl1 = numpy.linalg.norm(T1aold) + 0.1
        nl1 += numpy.linalg.norm(T1bold)
        nl2 = numpy.linalg.norm(T2aaold) + 0.1
        nl2 += numpy.linalg.norm(T2abold)
        nl2 += numpy.linalg.norm(T2bbold)

        res1 = numpy.linalg.norm(T1out[0] - T1aold) / nl1
        res1 += numpy.linalg.norm(T1out[1] - T1bold) / nl1
        res2 = numpy.linalg.norm(T2out[0] - T2aaold) / nl2
        res2 += numpy.linalg.norm(T2out[1] - T2abold) / nl2
        res2 += numpy.linalg.norm(T2out[2] - T2bbold) / nl2

        # damp new T-amplitudes
        T1aold = alpha*T1aold + (1.0 - alpha)*T1out[0]
        T1bold = alpha*T1bold + (1.0 - alpha)*T1out[1]
        T2aaold = alpha*T2aaold + (1.0 - alpha)*T2out[0]
        T2abold = alpha*T2abold + (1.0 - alpha)*T2out[1]
        T2bbold = alpha*T2bbold + (1.0 - alpha)*T2out[2]

        # compute energy
        E = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,beta)

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f   %.4E' % (i+1,E,res1+res2))
        i = i + 1
        if numpy.abs(E - Eold) < ethresh and res1+res2 < tthresh:
            converged = True
        Eold = E

    if not converged:
        print("WARNING: {} did not converge!".format(method))

    tend = time.time()
    if iprint > 0:
        print("Total {} time: {:.4f} s".format(method,(tend - tbeg)))

    return Eold,(T1aold,T1bold),(T2aaold,T2abold,T2bbold)

def ft_ucc_iter_extrap(method, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        g (array): quadrature weight vector.
        G (array): quadrature weight matrix.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]

    noa,nva = Fa.ov.shape
    nob,nvb = Fb.ov.shape
    t1bara = numpy.zeros((ng,nva,noa))
    t1barb = numpy.zeros((ng,nvb,nob))
    t2baraa = numpy.zeros((ng,nva,nva,noa,noa))
    t2barab = numpy.zeros((ng,nva,nvb,noa,nob))
    t2barbb = numpy.zeros((ng,nvb,nvb,nob,nob))
    T1newa = numpy.zeros(t1bara.shape)
    T1newb = numpy.zeros(t1barb.shape)
    T2newaa = numpy.zeros(t2baraa.shape)
    T2newab = numpy.zeros(t2barab.shape)
    T2newbb = numpy.zeros(t2barbb.shape)

    # loop over grid points
    for ig in range(ng):
        if ig == 0:
            t1bara[0] = -Fa.vo
            t1barb[0] = -Fb.vo
            t2baraa[0] = -Ia.vvoo
            t2barab[0] = -Iabab.vvoo
            t2barbb[0] = -Ib.vvoo
            continue # don't bother computing at T = inf
        elif ig == 1:
            t1bara[ig] = -Fa.vo
            t1barb[ig] = -Fb.vo
            t2baraa[ig] = -Ia.vvoo
            t2barab[ig] = -Iabab.vvoo
            t2barbb[ig] = -Ib.vvoo
            T1newa[ig] = quadrature.int_tbar1_single(ng,ig,t1bara,ti,D1a,G)
            T1newb[ig] = quadrature.int_tbar1_single(ng,ig,t1barb,ti,D1b,G)
            T2newaa[ig] = quadrature.int_tbar2_single(ng,ig,t2baraa,ti,D2aa,G)
            T2newab[ig] = quadrature.int_tbar2_single(ng,ig,t2barab,ti,D2ab,G)
            T2newbb[ig] = quadrature.int_tbar2_single(ng,ig,t2barbb,ti,D2bb,G)
        else:
            # linear extrapolation
            fac = (ti[ig] - ti[ig - 1])/(ti[ig - 2] - ti[ig - 1])
            T1newa[ig] = T1newa[ig - 1] + (T1newa[ig - 2] - T1newa[ig - 1])*fac
            T1newb[ig] = T1newb[ig - 1] + (T1newb[ig - 2] - T1newb[ig - 1])*fac
            T2newaa[ig] = T2newaa[ig - 1] + (T2newaa[ig - 2] - T2newaa[ig - 1])*fac
            T2newab[ig] = T2newab[ig - 1] + (T2newab[ig - 2] - T2newab[ig - 1])*fac
            T2newbb[ig] = T2newbb[ig - 1] + (T2newbb[ig - 2] - T2newbb[ig - 1])*fac
        converged = False
        nl1 = numpy.sqrt(float(T1newa[ig].size))
        nl2 = numpy.sqrt(float(T2newaa[ig].size))
        if iprint > 0:
            print("Time point {}".format(ig))
        i = 0
        while i < max_iter and not converged:
            # form new T1 and T2
            (T1a,T1b),(T2aa,T2ab,T2bb) = form_new_ampl_extrap_u(ig,method,Fa,Fb,Ia,Ib,Iabab,
                    T1newa[ig],T1newb[ig],T2newaa[ig],T2newab[ig],T2newbb[ig],
                    t1bara,t1barb,t2baraa,t2barab,t2barbb,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)

            res1 = numpy.linalg.norm(T1a - T1newa[ig]) / nl1
            res1 += numpy.linalg.norm(T1b - T1newb[ig]) / nl1
            res2 = numpy.linalg.norm(T2aa - T2newaa[ig]) / nl2
            res2 += numpy.linalg.norm(T2ab - T2newab[ig]) / nl2
            res2 += numpy.linalg.norm(T2bb - T2newbb[ig]) / nl2
            # damp new T-amplitudes
            T1newa[ig] = alpha*T1newa[ig] + (1.0 - alpha)*T1a.copy()
            T1newb[ig] = alpha*T1newb[ig] + (1.0 - alpha)*T1b.copy()
            T2newaa[ig] = alpha*T2newaa[ig] + (1.0 - alpha)*T2aa.copy()
            T2newab[ig] = alpha*T2newab[ig] + (1.0 - alpha)*T2ab.copy()
            T2newbb[ig] = alpha*T2newbb[ig] + (1.0 - alpha)*T2bb.copy()

            # determine convergence
            if iprint > 0:
                print(' %2d  %.4E' % (i+1,res1+res2))
            i = i + 1
            if res1 + res2 < thresh:
                converged = True
    return (T1newa,T1newb),(T2newaa,T2newab,T2newbb)

def ft_lambda_iter(method, L1old, L2old, T1, T2, F, I, D1, D2,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    i = 0
    nl1 = numpy.linalg.norm(L1old) + 0.1
    nl2 = numpy.linalg.norm(L2old) + 0.1
    while i < max_iter and not converged:
        if method == "LCCSD":
            L1,L2 = ft_cc_equations.lccsd_lambda_simple(
                F,I,T1,T2,L1old,L2old,D1,D2,ti,ng,g,G,beta)
        elif method == "LCCD":
            L1 = L1old
            L2 = ft_cc_equations.lccd_lambda_simple(F,I,self.T2,
                    L2old,D2,ti,ng,g,G,beta)
        elif method == "CCSD":
            L1,L2 = ft_cc_equations.ccsd_lambda_opt(
                F,I,T1,T2,L1old,L2old,D1,D2,ti,ng,g,G,beta)
        elif method == "CCD":
            L1 = L1old
            L2 = ft_cc_equations.ccd_lambda_simple(F,I,T2,
                    L2old,D2,ti,ng,g,G,beta)
        else:
            raise Exception("Unrecognized method keyword")

        res1 = numpy.linalg.norm(L1 - L1old) / nl1
        res2 = numpy.linalg.norm(L2 - L2old) / nl2
        # compute new L-amplitudes
        L1old = alpha*L1old + (1.0 - alpha)*L1
        L2old = alpha*L2old + (1.0 - alpha)*L2
        nl1 = numpy.linalg.norm(L1old) + 0.1
        nl2 = numpy.linalg.norm(L2old) + 0.1
        L1 = None
        L2 = None

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f' % (i+1,res1 + res2))
        i = i + 1
        if res1 + res2 < thresh:
            converged = True

    if not converged:
        print("WARNING: CCSD Lambda-equations did not converge!")

    tend = time.time()
    if iprint > 0:
        print("Total CCSD Lambda time: %f s" % (tend - tbeg))

    return L1old,L2old

def ft_ulambda_iter(method, L1ain, L1bin, L2aain, L2abin, L2bbin, T1aold, T1bold,
        T2aaold, T2abold, T2bbold, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
        g, G, beta, ng, ti, iprint, conv_options):
    """Form new amplitudes.

    Arguments:
        method (str): Amplitude equation type.
        F (array): Fock matrix.
        I (array): ERI tensor.
        T1old (array): T1 amplitudes.
        T2old (array): T2 amplitudes.
        D1 (array): 1-electron denominators.
        D2 (array): 2-electron denominators.
        beta (float): inverse temperature.
        ti (array): time grid.
        ng (int): number of time points.
        iprint (int): print level.
        conv_options (dict): Convergence options.
    """
    tbeg = time.time()
    converged = False
    thresh = conv_options["tconv"]
    max_iter = conv_options["max_iter"]
    alpha = conv_options["damp"]
    na = D1a.shape[0]
    nb = D1b.shape[0]
    n = na + nb
    i = 0
    L1aold = L1ain
    L1bold = L1bin
    L2aaold = L2aain
    L2abold = L2abin
    L2bbold = L2bbin

    nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold)
    nl2 = numpy.linalg.norm(L2aaold)
    nl2 += numpy.linalg.norm(L2bbold)
    nl2 += 4*numpy.linalg.norm(L2abold)
    while i < max_iter and not converged:
        if method == "LCCSD":
            raise Exception("U-LCCSD lambda equations not implemented")
        elif method == "LCCD":
            raise Exception("U-LCCD lambda equations not implemented")
        elif method == "CCSD":
            L1a,L1b,L2aa,L2ab,L2bb = ft_cc_equations.uccsd_lambda_opt(
                Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,T2aaold,T2abold,T2bbold,
                L1aold,L1bold,L2aaold,L2abold,L2bbold,D1a,D1b,D2aa,D2ab,D2bb,
                ti,ng,g,G,beta)
        elif method == "CCD":
            raise Exception("UCCD lambda equations not implemented")
        else:
            raise Exception("Unrecognized method keyword")

        res1 = numpy.linalg.norm(L1a - L1aold) / nl1
        res1 += numpy.linalg.norm(L1b - L1bold) / nl1
        res2 = numpy.linalg.norm(L2aa - L2aaold) / nl2
        res2 += numpy.linalg.norm(L2ab - L2abold) / nl2
        res2 += numpy.linalg.norm(L2bb - L2bbold) / nl2
        # compute new L-amplitudes
        L1aold = alpha*L1aold + (1.0 - alpha)*L1a
        L1bold = alpha*L1bold + (1.0 - alpha)*L1b
        L2aaold = alpha*L2aaold + (1.0 - alpha)*L2aa
        L2abold = alpha*L2abold + (1.0 - alpha)*L2ab
        L2bbold = alpha*L2bbold + (1.0 - alpha)*L2bb
        nl1 = numpy.linalg.norm(L1aold) + numpy.linalg.norm(L1bold)
        nl2 = numpy.linalg.norm(L2aaold)
        nl2 += numpy.linalg.norm(L2bbold)
        nl2 += 4*numpy.linalg.norm(L2abold)
        L1a = None
        L1b = None
        L2aa = None
        L2ab = None
        L2bb = None

        # determine convergence
        if iprint > 0:
            print(' %2d  %.10f' % (i+1,res1 + res2))
        i = i + 1
        if res1 + res2 < thresh:
            converged = True

    if not converged:
        print("WARNING: CCSD Lambda-equations did not converge!")

    tend = time.time()
    if iprint > 0:
        print("Total CCSD Lambda time: %f s" % (tend - tbeg))

    return L1aold,L1bold,L2aaold,L2abold,L2bbold

def get_ft_integrals(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)

        # get FT fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo,sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo,sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo,sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo,sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri,sfv,sfv,sfv,sfo)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri,sfv,sfv,sfo,sfo)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,sfv,sfo)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,sfv,sfv)
        Ivooo = einsum('akij,a,k,i,j->akij',eri,sfv,sfo,sfo,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri,sfo,sfo,sfo,sfo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return F,I

def get_ft_integrals_neq(sys, en, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis
        including real-time component."""
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get FT fock matrix
        fmo = sys.g_fock_tot(direc='f')
        fmo = (fmo - numpy.diag(en)).astype(complex)

        # pre-contract with fermi factors
        Foo = einsum('ij,j->ij',fmo[0],fo)
        Fvo = einsum('ai,a,i->ai',fmo[0],fv,fo)
        Fvv = einsum('ab,a->ab',fmo[0],fv)
        F = one_e_blocks(Foo,fmo[0],Fvo,Fvv)

        Foo = einsum('yij,j->yij',fmo,fo)
        Fvo = einsum('yai,a,i->yai',fmo,fv,fo)
        Fvv = einsum('yab,a->yab',fmo,fv)
        Ff = one_e_blocks(Foo,fmo,Fvo,Fvv)

        fmo = sys.g_fock_tot(direc='b')
        fmo = (fmo - numpy.diag(en)).astype(complex)
        Foo = einsum('yij,j->yij',fmo,fo)
        Fvo = einsum('yai,a,i->yai',fmo,fv,fo)
        Fvv = einsum('yab,a->yab',fmo,fv)
        Fb = one_e_blocks(Foo,fmo,Fvo,Fvv)

        # get ERIs
        eri = sys.g_aint_tot().astype(complex)

        Ivvvv = einsum('abcd,a,b->abcd',eri,fv,fv)
        Ivvvo = einsum('abci,a,b,i->abci',eri,fv,fv,fo)
        Ivovv = einsum('aibc,a->aibc',eri,fv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri,fv,fv,fo,fo)
        Ivovo = einsum('ajbi,a,i->ajbi',eri,fv,fo)
        Ivooo = einsum('akij,a,i,j->akij',eri,fv,fo,fo)
        Iooov = einsum('jkia,i->jkia',eri,fo)
        Ioooo = einsum('klij,i,j->klij',eri,fo,fo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=eri,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return F,Ff,Fb,I

def get_uft_integrals(sys, ea, eb, beta, mu):
        """Return one and two-electron integrals in the general spin orbital basis."""
        na = ea.shape[0]
        nb = eb.shape[0]
        #en = numpy.concatenate((ea,eb))
        #fo = ft_utils.ff(beta, en, mu)
        #fv = ft_utils.ffv(beta, en, mu)
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)

        # get FT fock matrix
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)

        # pre-contract with fermi factors
        Fooa = einsum('ij,i,j->ij',fa,sfoa,sfoa)
        Fova = einsum('ia,i,a->ia',fa,sfoa,sfva)
        Fvoa = einsum('ai,a,i->ai',fa,sfva,sfoa)
        Fvva = einsum('ab,a,b->ab',fa,sfva,sfva)
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)

        Foob = einsum('ij,i,j->ij',fb,sfob,sfob)
        Fovb = einsum('ia,i,a->ia',fb,sfob,sfvb)
        Fvob = einsum('ai,a,i->ai',fb,sfvb,sfob)
        Fvvb = einsum('ab,a,b->ab',fb,sfvb,sfvb)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)

        # get ERIs
        eriA,eriB,eriAB = sys.u_aint_tot()
        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriA,sfva,sfva,sfva,sfva)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriA,sfva,sfva,sfva,sfoa)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriA,sfva,sfoa,sfva,sfva)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriA,sfva,sfva,sfoa,sfoa)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriA,sfoa,sfoa,sfva,sfva)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriA,sfva,sfoa,sfva,sfoa)
        Ivooo = einsum('akij,a,k,i,j->akij',eriA,sfva,sfoa,sfoa,sfoa)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriA,sfoa,sfoa,sfoa,sfva)
        Ioooo = einsum('klij,k,l,i,j->klij',eriA,sfoa,sfoa,sfoa,sfoa)
        Ia = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriB,sfvb,sfvb,sfvb,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriB,sfvb,sfvb,sfvb,sfob)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriB,sfvb,sfob,sfvb,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriB,sfvb,sfvb,sfob,sfob)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriB,sfob,sfob,sfvb,sfvb)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriB,sfvb,sfob,sfvb,sfob)
        Ivooo = einsum('akij,a,k,i,j->akij',eriB,sfvb,sfob,sfob,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriB,sfob,sfob,sfob,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriB,sfob,sfob,sfob,sfob)
        Ib = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eriAB,sfva,sfvb,sfva,sfvb)
        Ivvvo = einsum('abci,a,b,c,i->abci',eriAB,sfva,sfvb,sfva,sfob)
        Ivvov = einsum('abic,a,b,i,c->abic',eriAB,sfva,sfvb,sfoa,sfvb)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eriAB,sfva,sfob,sfva,sfvb)
        Iovvv = einsum('iabc,i,a,b,c->iabc',eriAB,sfoa,sfvb,sfva,sfvb)
        Ivvoo = einsum('abij,a,b,i,j->abij',eriAB,sfva,sfvb,sfoa,sfob)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eriAB,sfva,sfob,sfva,sfob)
        Iovvo = einsum('jabi,j,a,b,i->jabi',eriAB,sfoa,sfvb,sfva,sfob)
        Ivoov = einsum('ajib,a,j,i,b->ajib',eriAB,sfva,sfob,sfoa,sfvb)
        Iovov = einsum('jaib,j,a,i,b->jaib',eriAB,sfoa,sfvb,sfoa,sfvb)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eriAB,sfoa,sfob,sfva,sfvb)
        Ivooo = einsum('akij,a,k,i,j->akij',eriAB,sfva,sfob,sfoa,sfob)
        Iovoo = einsum('kaij,k,a,i,j->kaij',eriAB,sfoa,sfvb,sfoa,sfob)
        Ioovo = einsum('jkai,j,k,a,i->jkai',eriAB,sfoa,sfob,sfva,sfob)
        Iooov = einsum('jkia,j,k,i,a->jkia',eriAB,sfoa,sfob,sfoa,sfvb)
        Ioooo = einsum('klij,k,l,i,j->klij',eriAB,sfoa,sfob,sfoa,sfob)
        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)

        return Fa,Fb,Ia,Ib,Iabab

def get_ft_active_integrals(sys, en, focc, fvir, iocc, ivir):
        """Return one and two-electron integrals in the general spin orbital basis
        with small occupations excluded."""
        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)

        # get ERIs
        eri = sys.g_aint_tot()

        # pre-contract with fermi factors
        Foo = einsum('ij,i,j->ij',fmo[numpy.ix_(iocc,iocc)],sfo,sfo)
        Fov = einsum('ia,i,a->ia',fmo[numpy.ix_(iocc,ivir)],sfo,sfv)
        Fvo = einsum('ai,a,i->ai',fmo[numpy.ix_(ivir,iocc)],sfv,sfo)
        Fvv = einsum('ab,a,b->ab',fmo[numpy.ix_(ivir,ivir)],sfv,sfv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)

        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri[numpy.ix_(ivir,ivir,ivir,ivir)],sfv,sfv,sfv,sfv)
        Ivvvo = einsum('abci,a,b,c,i->abci',eri[numpy.ix_(ivir,ivir,ivir,iocc)],sfv,sfv,sfv,sfo)
        Ivovv = einsum('aibc,a,i,b,c->aibc',eri[numpy.ix_(ivir,iocc,ivir,ivir)],sfv,sfo,sfv,sfv)
        Ivvoo = einsum('abij,a,b,i,j->abij',eri[numpy.ix_(ivir,ivir,iocc,iocc)],sfv,sfv,sfo,sfo)
        Ioovv = einsum('ijab,i,j,a,b->ijab',eri[numpy.ix_(iocc,iocc,ivir,ivir)],sfo,sfo,sfv,sfv)
        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri[numpy.ix_(ivir,iocc,ivir,iocc)],sfv,sfo,sfv,sfo)
        Ivooo = einsum('akij,a,k,i,j->akij',eri[numpy.ix_(ivir,iocc,iocc,iocc)],sfv,sfo,sfo,sfo)
        Iooov = einsum('jkia,j,k,i,a->jkia',eri[numpy.ix_(iocc,iocc,iocc,ivir)],sfo,sfo,sfo,sfv)
        Ioooo = einsum('klij,k,l,i,j->klij',eri[numpy.ix_(iocc,iocc,iocc,iocc)],sfo,sfo,sfo,sfo)
        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

        return F,I

def _form_ft_d_eris(eri, sfo, sfv, dso, dsv):
        Ivvvv = einsum('abcd,a,b,c,d->abcd',eri,dsv,sfv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,dsv,sfv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,dsv,sfv)\
              + einsum('abcd,a,b,c,d->abcd',eri,sfv,sfv,sfv,dsv)

        Ivvvo = einsum('abci,a,b,c,i->abci',eri,dsv,sfv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,dsv,sfv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,sfv,dsv,sfo)\
              + einsum('abci,a,b,c,i->abci',eri,sfv,sfv,sfv,dso)

        Ivovv = einsum('aibc,a,i,b,c->aibc',eri,dsv,sfo,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,dso,sfv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,dsv,sfv)\
              + einsum('aibc,a,i,b,c->aibc',eri,sfv,sfo,sfv,dsv)

        Ivvoo = einsum('abij,a,b,i,j->abij',eri,dsv,sfv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,dsv,sfo,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,sfv,dso,sfo)\
              + einsum('abij,a,b,i,j->abij',eri,sfv,sfv,sfo,dso)

        Ivovo = einsum('ajbi,a,j,b,i->ajbi',eri,dsv,sfo,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,dso,sfv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,dsv,sfo)\
              + einsum('ajbi,a,j,b,i->ajbi',eri,sfv,sfo,sfv,dso)

        Ioovv = einsum('ijab,i,j,a,b->ijab',eri,dso,sfo,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,dso,sfv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,dsv,sfv)\
              + einsum('ijab,i,j,a,b->ijab',eri,sfo,sfo,sfv,dsv)

        Ivooo = einsum('akij,a,k,i,j->akij',eri,dsv,sfo,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,dso,sfo,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,sfo,dso,sfo)\
              + einsum('akij,a,k,i,j->akij',eri,sfv,sfo,sfo,dso)

        Iooov = einsum('jkia,j,k,i,a->jkia',eri,dso,sfo,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,dso,sfo,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,dso,sfv)\
              + einsum('jkia,j,k,i,a->jkia',eri,sfo,sfo,sfo,dsv)

        Ioooo = einsum('klij,k,l,i,j->klij',eri,dso,sfo,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,dso,sfo,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,sfo,dso,sfo)\
              + einsum('klij,k,l,i,j->klij',eri,sfo,sfo,sfo,dso)

        I = two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
                vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)
        return I

def get_ft_d_integrals(sys, en, fo, fv, dvec):
        """form integrals contracted with derivatives of occupation numbers in the
        spin-orbital basis."""

        # get FT Fock matrix
        fmo = sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)
        fd = sys.g_fock_d_tot(dvec)

        # get ERIs
        eri = sys.g_aint_tot()
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)
        dso = -0.5*sfo*fv*dvec
        dsv = +0.5*sfv*fo*dvec

        # form derivative integrals
        Foo = einsum('ij,i,j->ij',fd,sfo,sfo)\
                + einsum('ij,i,j->ij',fmo,dso,sfo)\
                + einsum('ij,i,j->ij',fmo,sfo,dso)
        Fov = einsum('ia,i,a->ia',fd,sfo,sfv)\
                + einsum('ia,i,a->ia',fmo,dso,sfv)\
                + einsum('ia,i,a->ia',fmo,sfo,dsv)
        Fvo = einsum('ai,a,i->ai',fd,sfv,sfo)\
                + einsum('ai,a,i->ai',fmo,dsv,sfo)\
                + einsum('ai,a,i->ai',fmo,sfv,dso)
        Fvv = einsum('ab,a,b->ab',fd,sfv,sfv)\
                + einsum('ab,a,b->ab',fmo,dsv,sfv)\
                + einsum('ab,a,b->ab',fmo,sfv,dsv)
        F = one_e_blocks(Foo,Fov,Fvo,Fvv)
        
        I = _form_ft_d_eris(eri,sfo,sfv,dso,dsv)
        return F,I

def u_ft_d_integrals(sys, ea, eb, foa, fob, fva, fvb, dveca, dvecb):
        """form unrestricted integrals contracted with derivatives of occupation numbers."""
        na = ea.shape[0]
        nb = eb.shape[0]

        # get FT Fock matrices
        fa,fb = sys.u_fock_tot()
        fa = fa - numpy.diag(ea)
        fb = fb - numpy.diag(eb)
        fda,fdb = sys.u_fock_d_tot(dveca,dvecb)

        sfoa = numpy.sqrt(foa)
        sfva = numpy.sqrt(fva)
        dsoa = -0.5*sfoa*fva*dveca
        dsva = +0.5*sfva*foa*dveca

        sfob = numpy.sqrt(fob)
        sfvb = numpy.sqrt(fvb)
        dsob = -0.5*sfob*fvb*dvecb
        dsvb = +0.5*sfvb*fob*dvecb

        Fooa = einsum('ij,i,j->ij',fda,sfoa,sfoa)\
                + einsum('ij,i,j->ij',fa,dsoa,sfoa)\
                + einsum('ij,i,j->ij',fa,sfoa,dsoa)
        Fova = einsum('ia,i,a->ia',fda,sfoa,sfva)\
                + einsum('ia,i,a->ia',fa,dsoa,sfva)\
                + einsum('ia,i,a->ia',fa,sfoa,dsva)
        Fvoa = einsum('ai,a,i->ai',fda,sfva,sfoa)\
                + einsum('ai,a,i->ai',fa,dsva,sfoa)\
                + einsum('ai,a,i->ai',fa,sfva,dsoa)
        Fvva = einsum('ab,a,b->ab',fda,sfva,sfva)\
                + einsum('ab,a,b->ab',fa,dsva,sfva)\
                + einsum('ab,a,b->ab',fa,sfva,dsva)
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)

        Foob = einsum('ij,i,j->ij',fdb,sfob,sfob)\
                + einsum('ij,i,j->ij',fb,dsob,sfob)\
                + einsum('ij,i,j->ij',fb,sfob,dsob)
        Fovb = einsum('ia,i,a->ia',fdb,sfob,sfvb)\
                + einsum('ia,i,a->ia',fb,dsob,sfvb)\
                + einsum('ia,i,a->ia',fb,sfob,dsvb)
        Fvob = einsum('ai,a,i->ai',fdb,sfvb,sfob)\
                + einsum('ai,a,i->ai',fb,dsvb,sfob)\
                + einsum('ai,a,i->ai',fb,sfvb,dsob)
        Fvvb = einsum('ab,a,b->ab',fdb,sfvb,sfvb)\
                + einsum('ab,a,b->ab',fb,dsvb,sfvb)\
                + einsum('ab,a,b->ab',fb,sfvb,dsvb)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)

        # form derivative integrals
        #fova = dveca*foa*fva
        #fovb = dvecb*fob*fvb

        #Fooa = einsum('ij,j->ij',fa,fova) + einsum('ij,j->ij',fda,foa)
        #Fvoa = einsum('ai,a,i->ai',fda,fva,foa) + einsum('ai,a,i->ai',fa,fva,fova) \
        #        - einsum('ai,a,i->ai',fa,fova,foa)
        #Fvva = einsum('ab,a->ab',fda,fva) - einsum('ab,a->ab',fa,fova)
        #Foob = einsum('ij,j->ij',fa,fova) + einsum('ij,j->ij',fda,foa)
        #Fvob = einsum('ai,a,i->ai',fdb,fvb,fob) + einsum('ai,a,i->ai',fb,fvb,fovb) \
        #        - einsum('ai,a,i->ai',fb,fovb,fob)
        #Fvvb = einsum('ab,a->ab',fdb,fvb) - einsum('ab,a->ab',fb,fovb)
        #Fa = one_e_blocks(Fooa,fda,Fvoa,Fvva)
        #Fb = one_e_blocks(Foob,fdb,Fvob,Fvvb)

        # get ERIs
        Ia,Ib,Iabab = sys.u_aint_tot()

        #Iavvvv = - einsum('abcd,a,b->abcd',Ia,fova,fva)\
        #        - einsum('abcd,a,b->abcd',Ia,fva,fova)
        #Iavvvo = - einsum('abci,a,b,i->abci',Ia,fova,fva,foa)\
        #        - einsum('abci,a,b,i->abci',Ia,fva,fova,foa)\
        #        + einsum('abci,a,b,i->abci',Ia,fva,fva,fova)
        #Iavovv = - einsum('aibc,a->aibc',Ia,fova)
        #Iavvoo = - einsum('abij,a,b,i,j->abij',Ia,fova,fva,foa,foa)\
        #        - einsum('abij,a,b,i,j->abij',Ia,fva,fova,foa,foa)\
        #        + einsum('abij,a,b,i,j->abij',Ia,fva,fva,fova,foa)\
        #        + einsum('abij,a,b,i,j->abij',Ia,fva,fva,foa,fova)
        #Iavovo = - einsum('ajbi,a,i->ajbi',Ia,fova,foa) \
        #        + einsum('ajbi,a,i->ajbi',Ia,fva,fova)
        #Iavooo = -einsum('akij,a,i,j->akij',Ia,fova,foa,foa)\
        #        + einsum('akij,a,i,j->akij',Ia,fva,fova,foa)\
        #        + einsum('akij,a,i,j->akij',Ia,fva,foa,fova)
        #Iaooov = einsum('jkia,i->jkia',Ia,fova)
        #Iaoooo = einsum('klij,i,j->klij',Ia,fova,foa) \
        #        + einsum('klij,i,j->klij',Ia,foa,fova)
        #Iaoovv = numpy.zeros(Ia.shape)
        #Ia = two_e_blocks(vvvv=Iavvvv,vvvo=Iavvvo,vovv=Iavovv,vvoo=Iavvoo,
        #        vovo=Iavovo,oovv=Iaoovv,vooo=Iavooo,ooov=Iaooov,oooo=Iaoooo)
        Ia = _form_ft_d_eris(Ia,sfoa,sfva,dsoa,dsva)
        Ib = _form_ft_d_eris(Ib,sfob,sfvb,dsob,dsvb)

        #Ibvvvv = -einsum('abcd,a,b->abcd',Ib,fovb,fvb)\
        #        - einsum('abcd,a,b->abcd',Ib,fvb,fovb)
        #Ibvvvo = -einsum('abci,a,b,i->abci',Ib,fovb,fvb,fob)\
        #        - einsum('abci,a,b,i->abci',Ib,fvb,fovb,fob)\
        #        + einsum('abci,a,b,i->abci',Ib,fvb,fvb,fovb)
        #Ibvovv = -einsum('aibc,a->aibc',Ib,fovb)
        #Ibvvoo = -einsum('abij,a,b,i,j->abij',Ib,fovb,fvb,fob,fob)\
        #        - einsum('abij,a,b,i,j->abij',Ib,fvb,fovb,fob,fob)\
        #        + einsum('abij,a,b,i,j->abij',Ib,fvb,fvb,fovb,fob)\
        #        + einsum('abij,a,b,i,j->abij',Ib,fvb,fvb,fob,fovb)
        #Ibvovo = -einsum('ajbi,a,i->ajbi',Ib,fovb,fob) \
        #        + einsum('ajbi,a,i->ajbi',Ib,fvb,fovb)
        #Ibvooo = -einsum('akij,a,i,j->akij',Ib,fovb,fob,fob)\
        #        + einsum('akij,a,i,j->akij',Ib,fvb,fovb,fob)\
        #        + einsum('akij,a,i,j->akij',Ib,fvb,fob,fovb)
        #Ibooov = einsum('jkia,i->jkia',Ib,fovb)
        #Iboooo = einsum('klij,i,j->klij',Ib,fovb,fob) \
        #        + einsum('klij,i,j->klij',Ib,fob,fovb)
        #Iboovv = numpy.zeros(Ib.shape)
        #Ib = two_e_blocks(vvvv=Ibvvvv,vvvo=Ibvvvo,vovv=Ibvovv,vvoo=Ibvvoo,
        #        vovo=Ibvovo,oovv=Iboovv,vooo=Ibvooo,ooov=Ibooov,oooo=Iboooo)

        #I2vvvv = -einsum('abcd,a,b->abcd',Iabab,fova,fvb)\
        #        - einsum('abcd,a,b->abcd',Iabab,fva,fovb)
        #I2vvvo = -einsum('abci,a,b,i->abci',Iabab,fova,fvb,fob)\
        #        - einsum('abci,a,b,i->abci',Iabab,fva,fovb,fob)\
        #        + einsum('abci,a,b,i->abci',Iabab,fva,fvb,fovb)
        #I2vvov = -einsum('abic,a,b,i->abic',Iabab,fova,fvb,foa)\
        #        - einsum('abic,a,b,i->abic',Iabab,fva,fovb,foa)\
        #        + einsum('abic,a,b,i->abic',Iabab,fva,fvb,fova)
        #I2vovv = -einsum('aibc,a->aibc',Iabab,fova)
        #I2ovvv = -einsum('iabc,a->iabc',Iabab,fovb)
        #I2vvoo = -einsum('abij,a,b,i,j->abij',Iabab,fova,fvb,foa,fob)\
        #        - einsum('abij,a,b,i,j->abij',Iabab,fva,fovb,foa,fob)\
        #        + einsum('abij,a,b,i,j->abij',Iabab,fva,fvb,fova,fob)\
        #        + einsum('abij,a,b,i,j->abij',Iabab,fva,fvb,foa,fovb)
        #I2vovo = -einsum('ajbi,a,i->ajbi',Iabab,fova,fob) \
        #        + einsum('ajbi,a,i->ajbi',Iabab,fva,fovb)
        #I2ovvo = -einsum('jabi,a,i->jabi',Iabab,fovb,fob) \
        #        + einsum('jabi,a,i->jabi',Iabab,fvb,fovb)
        #I2voov = -einsum('ajib,a,i->ajib',Iabab,fova,foa) \
        #        + einsum('ajib,a,i->ajib',Iabab,fva,fova)
        #I2ovov = -einsum('jaib,a,i->jaib',Iabab,fovb,foa) \
        #        + einsum('jaib,a,i->jaib',Iabab,fvb,fova)
        #I2vooo = -einsum('akij,a,i,j->akij',Iabab,fova,foa,fob)\
        #        + einsum('akij,a,i,j->akij',Iabab,fva,fova,fob)\
        #        + einsum('akij,a,i,j->akij',Iabab,fva,foa,fovb)
        #I2ovoo = -einsum('kaij,a,i,j->kaij',Iabab,fovb,foa,fob)\
        #        + einsum('kaij,a,i,j->kaij',Iabab,fvb,fova,fob)\
        #        + einsum('kaij,a,i,j->kaij',Iabab,fvb,foa,fovb)
        #I2ooov = einsum('jkia,i->jkia',Iabab,fova)
        #I2oovo = einsum('jkai,i->jkai',Iabab,fovb)
        #I2oooo = einsum('klij,i,j->klij',Iabab,fova,fob) \
        #        + einsum('klij,i,j->klij',Iabab,foa,fovb)
        #I2oovv = numpy.zeros(Iabab.shape)
        Ivvvv =  einsum('abcd,a,b,c,d->abcd',Iabab,dsva,sfvb,sfva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,dsvb,sfva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,sfvb,dsva,sfvb)
        Ivvvv += einsum('abcd,a,b,c,d->abcd',Iabab,sfva,sfvb,sfva,dsvb)

        Ivvvo =  einsum('abci,a,b,c,i->abci',Iabab,dsva,sfvb,sfva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,dsvb,sfva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,sfvb,dsva,sfob)
        Ivvvo += einsum('abci,a,b,c,i->abci',Iabab,sfva,sfvb,sfva,dsob)

        Ivvov =  einsum('abic,a,b,i,c->abic',Iabab,dsva,sfvb,sfoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,dsvb,sfoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,sfvb,dsoa,sfvb)
        Ivvov += einsum('abic,a,b,i,c->abic',Iabab,sfva,sfvb,sfoa,dsvb)

        Ivovv =  einsum('aibc,a,i,b,c->aibc',Iabab,dsva,sfob,sfva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,dsob,sfva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,sfob,dsva,sfvb)
        Ivovv += einsum('aibc,a,i,b,c->aibc',Iabab,sfva,sfob,sfva,dsvb)

        Iovvv =  einsum('iabc,i,a,b,c->iabc',Iabab,dsoa,sfvb,sfva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,dsvb,sfva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,sfvb,dsva,sfvb)
        Iovvv += einsum('iabc,i,a,b,c->iabc',Iabab,sfoa,sfvb,sfva,dsvb)

        Ivvoo =  einsum('abij,a,b,i,j->abij',Iabab,dsva,sfvb,sfoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,dsvb,sfoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,sfvb,dsoa,sfob)
        Ivvoo += einsum('abij,a,b,i,j->abij',Iabab,sfva,sfvb,sfoa,dsob)

        Ivovo =  einsum('ajbi,a,j,b,i->ajbi',Iabab,dsva,sfob,sfva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,dsob,sfva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,sfob,dsva,sfob)
        Ivovo += einsum('ajbi,a,j,b,i->ajbi',Iabab,sfva,sfob,sfva,dsob)

        Iovvo =  einsum('jabi,j,a,b,i->jabi',Iabab,dsoa,sfvb,sfva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,dsvb,sfva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,sfvb,dsva,sfob)
        Iovvo += einsum('jabi,j,a,b,i->jabi',Iabab,sfoa,sfvb,sfva,dsob)

        Ivoov =  einsum('ajib,a,j,i,b->ajib',Iabab,dsva,sfob,sfoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,dsob,sfoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,sfob,dsoa,sfvb)
        Ivoov += einsum('ajib,a,j,i,b->ajib',Iabab,sfva,sfob,sfoa,dsvb)

        Iovov =  einsum('jaib,j,a,i,b->jaib',Iabab,dsoa,sfvb,sfoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,dsvb,sfoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,sfvb,dsoa,sfvb)
        Iovov += einsum('jaib,j,a,i,b->jaib',Iabab,sfoa,sfvb,sfoa,dsvb)

        Ioovv =  einsum('ijab,i,j,a,b->ijab',Iabab,dsoa,sfob,sfva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,dsob,sfva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,sfob,dsva,sfvb)
        Ioovv += einsum('ijab,i,j,a,b->ijab',Iabab,sfoa,sfob,sfva,dsvb)

        Ivooo =  einsum('akij,a,k,i,j->akij',Iabab,dsva,sfob,sfoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,dsob,sfoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,sfob,dsoa,sfob)
        Ivooo += einsum('akij,a,k,i,j->akij',Iabab,sfva,sfob,sfoa,dsob)

        Iovoo =  einsum('kaij,k,a,i,j->kaij',Iabab,dsoa,sfvb,sfoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,dsvb,sfoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,sfvb,dsoa,sfob)
        Iovoo += einsum('kaij,k,a,i,j->kaij',Iabab,sfoa,sfvb,sfoa,dsob)

        Ioovo =  einsum('jkai,j,k,a,i->jkai',Iabab,dsoa,sfob,sfva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,dsob,sfva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,sfob,dsva,sfob)
        Ioovo += einsum('jkai,j,k,a,i->jkai',Iabab,sfoa,sfob,sfva,dsob)

        Iooov =  einsum('jkia,j,k,i,a->jkia',Iabab,dsoa,sfob,sfoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,dsob,sfoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,sfob,dsoa,sfvb)
        Iooov += einsum('jkia,j,k,i,a->jkia',Iabab,sfoa,sfob,sfoa,dsvb)

        Ioooo =  einsum('klij,k,l,i,j->klij',Iabab,dsoa,sfob,sfoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,dsob,sfoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,sfob,dsoa,sfob)
        Ioooo += einsum('klij,k,l,i,j->klij',Iabab,sfoa,sfob,sfoa,dsob)

        Iabab = two_e_blocks_full(vvvv=Ivvvv,
                vvvo=Ivvvo,vvov=Ivvov,
                vovv=Ivovv,ovvv=Iovvv,
                vvoo=Ivvoo,vovo=Ivovo,
                ovvo=Iovvo,voov=Ivoov,
                ovov=Iovov,oovv=Ioovv,
                vooo=Ivooo,ovoo=Iovoo,
                oovo=Ioovo,ooov=Iooov,
                oooo=Ioooo)


        return Fa,Fb,Ia,Ib,Iabab
