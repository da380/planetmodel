"""planetmodel.loading -- the quasi-static spheroidal loading problem.

Assembles and solves, degree by degree, the reduced weak form of the
elastostatic loading problem for a spherically symmetric, self-
gravitating, hydrostatically pre-stressed earth model, following
Al-Attar & Tromp (2014, GJI 196, 34-77), Appendix D: the bilinear form
of eq. (D76) with the surface load force of eq. (D79).  Dropping the
viscoelastic memory terms of the full theory leaves exactly this
system, which is also the operator inverted at every step of a future
viscoelastic time-stepper.

Unknowns per degree l (k^2 = l(l+1)): on solid nodes the radial and
tangential displacement scalars U, V (generalized-spherical-harmonic
convention of the paper) and the potential perturbation phi; on fluid
nodes phi alone, the deformation there being characterized entirely by
the potential (Dahlen 1974; Crossley & Gubbins 1975), with the
stratification term g^-1 (d rho/dr) phi phi' r^2 and explicit
fluid-solid interface couplings.  The exterior is closed by the
Dirichlet-to-Neumann term (l+1) a phi phi' / 4 pi G at the surface and,
when the domain is truncated for high degrees, by the interior
continuation term l r phi phi' / 4 pi G at the bottom.  For l = 1 the
surface potential dof is removed (centre-of-mass frame: the exterior
degree-one field vanishes) and k is reported as zero.

Elasticity is transversely isotropic by default, using the model's
A, C, F, L, N directly (ti_moduli) in the TI radial weak form of the
DSpecM1D static operator (Myhill & Al-Attar), whose isotropic limit
is exactly eq. (D76); pass moduli=voigt_moduli for the Voigt-averaged
isotropic collapse instead.

Love numbers follow the dimensional convention u_lm = h_l sigma_lm,
v_lm = l_l sigma_lm, phi_lm = k_l sigma_lm for a unit surface-density
load coefficient; conversions to the conventional dimensionless load
numbers (h', l', k') are provided.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded

from .mesh1d import G_NEWTON, RadialMesh

__all__ = ["voigt_moduli", "ti_moduli", "solve_degree", "love_numbers",
           "love_number_table", "reciprocity_residual",
           "write_love_numbers"]


# ---------------------------------------------------------------------------
# material moduli
# ---------------------------------------------------------------------------

def ti_moduli(mesh: RadialMesh) -> tuple:
    """Nodal transversely isotropic moduli (A, C, F, L, N) of the model.

    The moduli are what a model stores -- deck readers convert
    velocities on read -- so this is a lookup rather than a conversion.
    Models predating that, or built by hand without moduli, still work:
    the fallback rebuilds them from the velocity columns exactly as
    before.

    This is the default constitutive input of the solver, whose elastic
    kernel is the transversely isotropic radial weak form (as in the
    DSpecM1D static operator of Myhill & Al-Attar), reducing to
    eq. (D76) of Al-Attar & Tromp (2014) in the isotropic limit.  Fluid
    layers give A = C = F = rho vp^2 and L = N = 0, the correct mu = 0
    limit used at degree 0.
    """
    model = mesh.model
    if all(name in model for name in ("A", "C", "F", "L", "N")):
        return tuple(mesh.nodal(name) for name in ("A", "C", "F", "L", "N"))

    rho = mesh.nodal("rho")
    A = rho * mesh.nodal("vph") ** 2
    C = rho * mesh.nodal("vpv") ** 2
    Lm = rho * mesh.nodal("vsv") ** 2
    N = rho * mesh.nodal("vsh") ** 2
    F = mesh.nodal("eta") * (A - 2.0 * Lm)
    return A, C, F, Lm, N


def voigt_moduli(mesh: RadialMesh) -> tuple[np.ndarray, np.ndarray]:
    """Nodal isotropic (kappa, mu), Voigt-averaged from the TI moduli.

        kappa = (4A + C + 4F - 4N)/9,  mu = (A + C - 2F + 5N + 6L)/15,

    which reduce exactly to rho (vp^2 - 4 vs^2 / 3) and rho vs^2 where
    the model is isotropic.  Shapes (nspec, ngll).
    """
    A, C, F, L, N = ti_moduli(mesh)
    kappa = (4.0 * A + C + 4.0 * F - 4.0 * N) / 9.0
    mu = (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0
    return kappa, mu


def _as_five_moduli(mesh: RadialMesh, moduli):
    """Resolve constitutive input to nodal (A, C, F, L, N) arrays.

    Accepts None (full TI from the model via ti_moduli), a callable of
    the mesh returning either form, a 2-tuple (kappa, mu) of nodal
    arrays -- mapped isotropically as A = C = kappa + 4 mu / 3,
    F = kappa - 2 mu / 3, L = N = mu -- or a 5-tuple (A, C, F, L, N).
    """
    if moduli is None:
        moduli = ti_moduli
    if callable(moduli):
        moduli = moduli(mesh)
    moduli = tuple(moduli)
    if len(moduli) == 2:
        kap, mu = moduli
        A = kap + 4.0 * mu / 3.0
        return A, A, kap - 2.0 * mu / 3.0, mu, mu
    if len(moduli) == 5:
        return moduli
    raise ValueError("moduli must resolve to (kappa, mu) or (A, C, F, L, N)")


# ---------------------------------------------------------------------------
# element matrices (transcribed from eq. D76; dr measure, k^2 = l(l+1))
# ---------------------------------------------------------------------------

def _solid_local(l: int, r, rho, g, A, C, F, Lm, N, w, jac: float, D,
                 G: float) -> np.ndarray:
    """Dense local matrix of a solid element; dofs (U, V, phi) per node.

    The elastic block is the transversely isotropic radial weak form
    (the DSpecM1D static operator): with X = r d_r U, Y = 2U - k^2 V
    and S = U + r d_r V - V,

        C X X' + (A - N) Y Y' + F (X Y' + Y X')
          + k^2 L S S' + k^2 (k^2 - 2) N V V',

    which reduces exactly to the isotropic terms of eq. (D76) under
    A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3, L = N = mu.  The
    gravitational, coupling and potential terms follow eq. (D76)
    unchanged.
    """
    n = r.size
    k2 = l * (l + 1.0)
    iU = 3 * np.arange(n)
    iV = iU + 1
    iP = iU + 2
    B = r[:, None] * D / jac          # B[q, i] = r_q l_i'(x_q) / jac
    I = np.eye(n)
    Wq = w * jac
    L = np.zeros((3 * n, 3 * n))

    def quad(P, coef):
        """Add int coef * P(x) P'(x) dr for the strain-like row map P."""
        return (P * (coef * Wq)[:, None]).T @ P

    X = np.zeros((n, 3 * n))
    X[:, iU] = B                                  # r d_r U
    Y = np.zeros((n, 3 * n))
    Y[:, iU] = 2.0 * I                            # 2U - k^2 V
    Y[:, iV] = -k2 * I
    S = np.zeros((n, 3 * n))
    S[:, iU] = I                                  # U + r d_r V - V
    S[:, iV] = B - I
    L += quad(X, C)
    L += quad(Y, A - N)
    XF = (X * (F * Wq)[:, None]).T @ Y
    L += XF + XF.T
    L += k2 * quad(S, Lm)
    # k^2 (k^2 - 2) N V V'
    L[iV, iV] += k2 * (k2 - 2.0) * N * Wq
    # 4 rho (pi G rho r - g) U U' r
    L[iU, iU] += 4.0 * rho * (np.pi * G * rho * r - g) * r * Wq
    # k^2 rho g (V U' + U V') r
    cUV = k2 * rho * g * r * Wq
    L[iU, iV] += cUV
    L[iV, iU] += cUV
    # k^2 rho (V phi' + phi V') r
    cVP = k2 * rho * r * Wq
    L[iV, iP] += cVP
    L[iP, iV] += cVP
    # rho (phi' U' ... ) : rho (d_r phi U' + U d_r phi') r^2
    M = (rho * r * r * w)[:, None] * D
    L[np.ix_(iU, iP)] += M
    L[np.ix_(iP, iU)] += M.T
    # (1 / 4 pi G) (r^2 d_r phi d_r phi' + k^2 phi phi')
    ifpg = 1.0 / (4.0 * np.pi * G)
    L[np.ix_(iP, iP)] += ifpg * ((D * (w * r * r)[:, None]).T @ D) / jac
    L[iP, iP] += ifpg * k2 * Wq
    return L


def _fluid_local(l: int, r, drho, g, w, jac: float, D,
                 G: float) -> np.ndarray:
    """Dense local matrix of a fluid element; only the phi dofs are live.

    Comprises the potential terms of eq. (D76) plus the fluid
    stratification term g^-1 (d rho / dr) phi phi' r^2.
    """
    n = r.size
    k2 = l * (l + 1.0)
    iP = 3 * np.arange(n) + 2
    ifpg = 1.0 / (4.0 * np.pi * G)
    Wq = w * jac
    L = np.zeros((3 * n, 3 * n))
    L[np.ix_(iP, iP)] += ifpg * ((D * (w * r * r)[:, None]).T @ D) / jac
    L[iP, iP] += ifpg * k2 * Wq
    ok = g > 1e-8
    strat = np.where(ok, r * r * drho / np.where(ok, g, 1.0), 0.0)
    L[iP, iP] += strat * Wq
    return L


# ---------------------------------------------------------------------------
# dof numbering and assembly
# ---------------------------------------------------------------------------

def _dof_map(mesh: RadialMesh, e0: int, l: int) -> tuple[np.ndarray, int]:
    """Global dof numbers per (node, component), -1 where absent.

    Components are (0, 1, 2) = (U, V, phi).  Nodes touched by any
    active solid element carry all three; purely fluid nodes carry phi
    only.  Numbering is node-major from the truncation node upward.
    For l = 1 the surface phi dof is removed (phi(a) = 0 in the
    centre-of-mass frame).
    """
    nglob = mesh.nglob
    g0 = int(mesh.gmap[e0, 0])
    solid = np.zeros(nglob, dtype=bool)
    for e in range(e0, mesh.nspec):
        if l == 0 or not mesh.is_fluid[e]:
            solid[mesh.gmap[e]] = True
    comps = (0, 2) if l == 0 else (0, 1, 2)
    dof = -np.ones((nglob, 3), dtype=int)
    nd = 0
    for gn in range(g0, nglob):
        for c in (comps if solid[gn] else (2,)):
            dof[gn, c] = nd
            nd += 1
    if l == 1:
        drop = dof[nglob - 1, 2]
        dof[nglob - 1, 2] = -1
        dof[dof > drop] -= 1
        nd -= 1
    if l == 0 and float(mesh.left[e0]) == 0.0:
        # A degree-0 radial displacement must vanish at the centre: a
        # 'hedgehog' U(0) != 0 has finite energy in the reduced measure
        # but is not an admissible 3-D field, so impose U(0) = 0.
        drop = dof[g0, 0]
        dof[g0, 0] = -1
        dof[dof > drop] -= 1
        nd -= 1
    return dof, nd


def _assemble(mesh: RadialMesh, l: int, e0: int, moduli5,
              G: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Assemble the degree-l system in symmetric upper band storage.

    Returns (ab, dof, kd, ndim) with ab[kd + i - j, j] = A[i, j] for
    i <= j and bandwidth kd = 3 (ngll - 1) + 2.
    """
    A5, C5, F5, L5, N5 = moduli5
    dof, nd = _dof_map(mesh, e0, l)
    kd = 3 * (mesh.ngll - 1) + 2
    ab = np.zeros((kd + 1, nd))
    rows, cols, vals = [], [], []

    r_all = mesh.r
    rho_all = mesh.nodal("rho")
    g_all = mesh.nodal_gravity(G=G)
    drho_all = mesh.nodal_derivative("rho")
    D, w = mesh.deriv, mesh.w

    for e in range(e0, mesh.nspec):
        if mesh.is_fluid[e] and l > 0:
            L = _fluid_local(l, r_all[e], drho_all[e], g_all[e],
                             w, float(mesh.jac[e]), D, G)
        else:
            L = _solid_local(l, r_all[e], rho_all[e], g_all[e],
                             A5[e], C5[e], F5[e], L5[e], N5[e],
                             w, float(mesh.jac[e]), D, G)
        gd = dof[mesh.gmap[e]].reshape(-1)
        GA, GB = gd[:, None], gd[None, :]
        sel = (GA >= 0) & (GB >= 0) & (GA <= GB) & (L != 0.0)
        rows.append((kd + GA - GB)[sel] * 0 + (kd + GA - GB)[sel])
        cols.append(np.broadcast_to(GB, L.shape)[sel])
        vals.append(L[sel])

    def point(i: int, j: int, v: float) -> None:
        """Queue a single symmetric-upper entry."""
        if i < 0 or j < 0:
            return
        if i > j:
            i, j = j, i
        rows.append(np.array([kd + i - j]))
        cols.append(np.array([j]))
        vals.append(np.array([v]))

    # fluid-solid interface terms: +/- rho_f (g U U' + phi U' + U phi') r^2
    flu = mesh.is_fluid
    for e in range(e0, mesh.nspec - 1):
        if l == 0 or flu[e] == flu[e + 1]:
            continue
        gn = int(mesh.gmap[e, -1])
        rt, gt = float(r_all[e, -1]), float(g_all[e, -1])
        if flu[e]:                       # fluid below, solid above (e.g. CMB)
            rf, sgn = float(rho_all[e, -1]), +1.0
        else:                            # solid below, fluid above (e.g. ICB)
            rf, sgn = float(rho_all[e + 1, 0]), -1.0
        point(dof[gn, 0], dof[gn, 0], sgn * rf * gt * rt * rt)
        point(dof[gn, 0], dof[gn, 2], sgn * rf * rt * rt)

    # Dirichlet-to-Neumann closures
    ifpg = 1.0 / (4.0 * np.pi * G)
    top = int(mesh.gmap[-1, -1])
    if l != 1:
        point(dof[top, 2], dof[top, 2], (l + 1) * float(r_all[-1, -1]) * ifpg)
    if mesh.left[e0] > 0.0:
        bot = int(mesh.gmap[e0, 0])
        point(dof[bot, 2], dof[bot, 2], l * float(mesh.left[e0]) * ifpg)

    np.add.at(ab, (np.concatenate(rows), np.concatenate(cols)),
              np.concatenate(vals))
    return ab, dof, kd, nd


def _solve(ab: np.ndarray, kd: int, b: np.ndarray) -> tuple[np.ndarray, str]:
    """Solve the banded symmetric system, preferring Cholesky.

    A symmetric diagonal (Jacobi) scaling equalizes the mixed physical
    units of the U/V and phi dofs before factorization; on a Cholesky
    failure the solver falls back to a general banded LU.
    """
    n = ab.shape[1]
    b2 = b if b.ndim == 2 else b[:, None]
    d = ab[kd].copy()
    if np.all(d > 0.0):
        s = 1.0 / np.sqrt(d)
        abs_ = np.zeros_like(ab)
        for m in range(kd + 1):
            u = kd - m                    # superdiagonal offset of band row m
            if u < n:
                js = np.arange(u, n)
                abs_[m, u:] = ab[m, u:] * s[js] * s[js - u]
        try:
            c = cholesky_banded(abs_, lower=False)
            x = s[:, None] * cho_solve_banded((c, False), s[:, None] * b2)
            return (x if b.ndim == 2 else x[:, 0]), "cholesky"
        except np.linalg.LinAlgError:
            pass
    full = np.zeros((2 * kd + 1, n))
    full[:kd + 1] = ab
    for m in range(kd):
        u = kd - m
        if u < n:
            full[kd + u, :n - u] = ab[m, u:]
    x = solve_banded((kd, kd), full, b2)
    return (x if b.ndim == 2 else x[:, 0]), "banded-lu"


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def _rhs_load(mesh, dof, nd: int, l: int, part: str, G: float) -> "np.ndarray":
    """Surface-load force vector (eq. D79) for a unit sigma_lm.

    part selects the channel: "force" applies only the mechanical piece
    -g a^2 on U'(a), "potential" only the attraction piece -a^2 on
    phi'(a), and "both" the full load.  For l = 1 the potential piece
    vanishes with the removed surface dof.
    """
    b = np.zeros(nd)
    top = int(mesh.gmap[-1, -1])
    a = float(mesh.r[-1, -1])
    if part in ("force", "both"):
        b[dof[top, 0]] = -float(mesh.nodal_gravity(G=G)[-1, -1]) * a * a
    if part in ("potential", "both") and dof[top, 2] >= 0:
        b[dof[top, 2]] = -a * a
    return b


def _rhs_tide(mesh, dof, nd: int, l: int, e0: int, G: float) -> "np.ndarray":
    """Force vector for a unit external tidal potential psi = (r/a)^l.

    Obtained by writing the total potential as phi + psi and moving
    the known psi terms to the right-hand side.  The result is the
    radial reduction of the forced weak form of Yu et al. (2025),
    eq. (A2), term by term:

        int_MS rho u'.grad(psi)      -> -rho d_r psi U' r^2
                                        and -k^2 rho psi V' r,
        int_MF g^-1 d_r rho psi phi' -> -g^-1 (d rho/dr) psi phi' r^2,
        int_Sigma_FS rho^- psi n.u'  -> -rho_f psi U' r^2  (fluid below),
        int_Sigma_SF rho^+ psi n.u'  -> +rho_f psi U' r^2  (fluid above),

    the fluid terms following Dahlen (1974) as extended by Bagheri
    et al. (2019), for whom the fluid density perturbation responds to
    the total potential phi + psi.  Degree 0 returns zero, a uniform
    external potential exerting no force; degree 1 does not, the
    centre-of-mass constraint phi(a) = 0 leaving a non-trivial
    response.
    """
    b = np.zeros(nd)
    if l == 0:
        return b
    a = float(mesh.r[-1, -1])
    k2 = l * (l + 1.0)
    rho_all = mesh.nodal("rho")
    g_all = mesh.nodal_gravity(G=G)
    drho_all = mesh.nodal_derivative("rho")
    for e in range(e0, mesh.nspec):
        r = mesh.r[e]
        Wq = mesh.w * float(mesh.jac[e])
        psi = (r / a) ** l
        dpsi = (l / a) * (r / a) ** (l - 1)
        gd = dof[mesh.gmap[e]]
        if mesh.is_fluid[e]:
            ok = g_all[e] > 1e-8
            strat = np.where(ok, r * r * drho_all[e]
                             / np.where(ok, g_all[e], 1.0), 0.0)
            b[gd[:, 2]] -= Wq * strat * psi
        else:
            b[gd[:, 0]] -= Wq * rho_all[e] * r * r * dpsi
            b[gd[:, 1]] -= k2 * Wq * rho_all[e] * r * psi
    flu = mesh.is_fluid
    for e in range(e0, mesh.nspec - 1):
        if l == 0 or flu[e] == flu[e + 1]:
            continue
        gn = int(mesh.gmap[e, -1])
        rt = float(mesh.r[e, -1])
        if flu[e]:                       # fluid below, solid above
            rf, sgn = float(rho_all[e, -1]), +1.0
        else:                            # solid below, fluid above
            rf, sgn = float(rho_all[e + 1, 0]), -1.0
        b[dof[gn, 0]] -= sgn * rf * rt * rt * (rt / a) ** l
    return b


def solve_degree(mesh: RadialMesh, l: int, moduli=None,
                 G: float = G_NEWTON, eps: float = 1e-8,
                 forcing: str = "load"):
    """Solve degree l >= 0 for one forcing; return (h, lv, k).

    forcing is "load" (unit surface density, both channels),
    "load_force" / "load_potential" (its mechanical / attraction
    pieces separately), or "tide" (unit external potential
    psi = (r/a)^l).  The returned values are the surface U, V, phi per
    unit forcing; components without a dof (V at l = 0, phi at l = 1)
    are zero, as is the whole tidal response for l < 2.  The loaded
    surface must be solid (strip a fluid ocean with the mesh's rmax).
    """
    if l < 0:
        raise ValueError("degree must be non-negative")
    if mesh.is_fluid[-1]:
        raise ValueError("loaded surface is fluid; rebuild the mesh with "
                         "rmax at the solid surface (e.g. 6368e3 for PREM)")
    m5 = _as_five_moduli(mesh, moduli)
    e0 = mesh.first_element_for(l, eps=eps) if l > 0 else 0
    ab, dof, kd, nd = _assemble(mesh, l, e0, m5, G)
    if forcing == "load":
        b = _rhs_load(mesh, dof, nd, l, "both", G)
    elif forcing == "load_force":
        b = _rhs_load(mesh, dof, nd, l, "force", G)
    elif forcing == "load_potential":
        b = _rhs_load(mesh, dof, nd, l, "potential", G)
    elif forcing == "tide":
        b = _rhs_tide(mesh, dof, nd, l, e0, G)
    else:
        raise ValueError(f"unknown forcing {forcing!r}")
    x, _ = _solve(ab, kd, b)
    top = int(mesh.gmap[-1, -1])
    h = float(x[dof[top, 0]])
    lv = float(x[dof[top, 1]]) if dof[top, 1] >= 0 else 0.0
    k = float(x[dof[top, 2]]) if dof[top, 2] >= 0 else 0.0
    return h, lv, k


def love_numbers(model, lmax: int, *, lmin: int = 1, ngll: int = 5,
                 rmax: float | None = None, mesh: RadialMesh | None = None,
                 eps: float = 1e-8, G: float = G_NEWTON,
                 moduli=None) -> dict:
    """Load Love numbers for degrees lmin..lmax on one shared mesh.

    Returns a dict of arrays: 'l'; the dimensional 'h', 'v', 'k'
    (u_lm = h_l sigma_lm etc.); and the conventional dimensionless
    load numbers 'hp', 'lp', 'kp' defined through the load potential
    +4 pi G a sigma / (2l+1):

        h' = h g (2l+1) / (4 pi G a),
        l' = v g (2l+1) / (4 pi G a),
        k' = -k (2l+1) / (4 pi G a) - 1,

    V being directly the coefficient of grad_1 Y in the tangential
    displacement (verified against published centre-of-mass degree-one
    values), and the sign of k' reflecting phi's physical-potential
    convention (negative near added mass).  For l = 1 the values are
    centre-of-mass frame numbers with k' = -1 identically.  The mesh
    is built with the drmax rule for lmax unless one is supplied;
    'mesh' is included in the result for reuse.
    """
    if mesh is None:
        mesh = RadialMesh(model, ngll=ngll, lmax=lmax, rmax=rmax)
    m5 = _as_five_moduli(mesh, moduli)
    ls = np.arange(lmin, lmax + 1)
    h = np.empty(ls.size)
    v = np.empty(ls.size)
    k = np.empty(ls.size)
    for i, l in enumerate(ls):
        h[i], v[i], k[i] = solve_degree(mesh, int(l), m5, G=G, eps=eps)
    a = float(mesh.r[-1, -1])
    ga = float(mesh.nodal_gravity(G=G)[-1, -1])
    fac = ga * (2.0 * ls + 1.0) / (4.0 * np.pi * G * a)
    hp = h * fac
    lp = v * fac
    kp = -k * (2.0 * ls + 1.0) / (4.0 * np.pi * G * a) - 1.0
    return {"l": ls, "h": h, "v": v, "k": k,
            "hp": hp, "lp": lp, "kp": kp, "mesh": mesh}


def love_number_table(model, lmax: int, *, ngll: int = 5,
                      rmax: float | None = None,
                      mesh: RadialMesh | None = None, eps: float = 1e-8,
                      G: float = G_NEWTON, moduli=None) -> dict:
    """The pyslfp-style Love number table for degrees 0..lmax.

    These are the generalized loading Love numbers of Al-Attar et al.
    (2024), eq. (62): for the generalized fingerprint problem the load
    splits into a traction-like term t = zeta_u grad(Phi) and a
    potential-source term zeta_phi, so that

        u_lm   = h_l sigma_lm + h^u_l zeta^u_lm + h^phi_l zeta^phi_lm,
        phi_lm = k_l sigma_lm + k^u_l zeta^u_lm + k^phi_l zeta^phi_lm.

    A true surface load acts through both channels at once, whence
    h_l = h^u_l + h^phi_l and k_l = k^u_l + k^phi_l (their eq. 63).
    Calculation differs only in the boundary terms applied, which here
    means only which surface entries of the force vector are set.

    For every degree the system is assembled and factorized once and
    solved for three right-hand sides, giving arrays (length lmax + 1):

        h_u, k_u      h^u_l, k^u_l: surface U and phi per unit
                      traction-like load zeta_u;
        h_phi, k_phi  h^phi_l, k^phi_l: the same per unit
                      potential-like load zeta_phi, so the sums are
                      the ordinary load Love numbers of love_numbers();
        h_t, k_t      surface U and phi per unit external potential
                      psi = (r/a)^l -- the rotational-feedback channel
                      of Yu et al. (2025), eq. (A6).  k_t is the
                      dimensionless ratio phi(a)/psi(a) and h_t has
                      units of length per potential; both vanish at
                      l = 0, and k_t also at l = 1 where the surface
                      potential dof is removed.  In the classical
                      geodetic normalization
                      k_2^T = k_t and h_2^T = -g h_t.

    By the symmetry of the bilinear form these six numbers satisfy
    g h^phi_l = k^u_l (their eq. 64, "a useful check within numerical
    calculations"); see reciprocity_residual().

    All values are dimensional SI in the physical-potential sign
    convention (phi negative near added mass), which is what the
    pyslfp reader expects; the conventional geodetic tidal numbers are
    k_2^T = k_t and h_2^T = -g h_t.  `moduli` selects the constitutive
    input (see _as_five_moduli): the default is the full transversely
    isotropic ti_moduli, and moduli=voigt_moduli recovers the
    isotropic Voigt collapse.

    At degree 0 the potential-only fluid reduction is invalid (it
    discards the radial compression that governs a fluid's l = 0
    response), so fluid regions are treated as mu = 0 elastic media
    and the essential condition U(0) = 0 is imposed; the resulting
    total k_0 satisfies the mass-conservation identity
    k_u + k_phi = -4 pi G a to solver precision.  Degree 1 is in the
    centre-of-mass frame with the potential channel identically zero.
    """
    if mesh is None:
        mesh = RadialMesh(model, ngll=ngll, lmax=max(lmax, 1), rmax=rmax)
    if mesh.is_fluid[-1]:
        raise ValueError("loaded surface is fluid; rebuild the mesh with "
                         "rmax at the solid surface (e.g. 6368e3 for PREM)")
    m5 = _as_five_moduli(mesh, moduli)
    out = {name: np.zeros(lmax + 1)
           for name in ("h_u", "k_u", "h_phi", "k_phi", "h_t", "k_t")}
    top = int(mesh.gmap[-1, -1])
    for l in range(lmax + 1):
        e0 = mesh.first_element_for(l, eps=eps) if l > 0 else 0
        ab, dof, kd, nd = _assemble(mesh, l, e0, m5, G)
        B = np.column_stack([_rhs_load(mesh, dof, nd, l, "force", G),
                             _rhs_load(mesh, dof, nd, l, "potential", G),
                             _rhs_tide(mesh, dof, nd, l, e0, G)])
        X, _ = _solve(ab, kd, B)
        iu, ip = dof[top, 0], dof[top, 2]
        out["h_u"][l] = X[iu, 0]
        out["h_phi"][l] = X[iu, 1]
        out["h_t"][l] = X[iu, 2]
        if ip >= 0:
            out["k_u"][l] = X[ip, 0]
            out["k_phi"][l] = X[ip, 1]
            out["k_t"][l] = X[ip, 2]
    out["l"] = np.arange(lmax + 1)
    out["mesh"] = mesh
    return out


def reciprocity_residual(table: dict, *, mesh: RadialMesh | None = None,
                         G: float = G_NEWTON) -> np.ndarray:
    """Relative violation of g h^phi_l = k^u_l, per degree.

    The identity follows from the symmetry of the bilinear form: with
    forcings b_u = -g a^2 e_U(a) and b_phi = -a^2 e_phi(a), equality of
    b_phi^T A^-1 b_u and b_u^T A^-1 b_phi gives a^2 k^u_l = g a^2
    h^phi_l.  It is eq. (64) of Al-Attar et al. (2024), recommended
    there as a check on numerical calculations, and should hold to
    solver precision; degrees whose potential dof is absent (l = 1)
    return zero.
    """
    if mesh is None:
        mesh = table["mesh"]
    ga = float(mesh.nodal_gravity(G=G)[-1, -1])
    ku, hphi = table["k_u"], table["h_phi"]
    scale = np.maximum(np.abs(ku), np.abs(ga * hphi))
    safe = np.where(scale > 0.0, scale, 1.0)
    return np.where(scale > 0.0, np.abs(ku - ga * hphi) / safe, 0.0)


def write_love_numbers(path, model=None, lmax: int | None = None, *,
                       table: dict | None = None, ngll: int = 5,
                       rmax: float | None = None, eps: float = 1e-8,
                       G: float = G_NEWTON, moduli=None) -> dict:
    """Write a Love number file in the format read by pyslfp.

    The file is plain text, loadable with numpy.loadtxt: one row per
    degree from 0 to lmax with columns

        l   h_u   k_u   h_phi   k_phi   h_t   k_t

    in the dimensional SI conventions of love_number_table() -- the
    pyslfp LoveNumbers reader non-dimensionalizes h_u, h_phi with
    load_scale/length_scale, k_u, k_phi with
    load_scale/gravitational_potential_scale, h_t with
    gravitational_potential_scale/length_scale, and reads k_t as
    already dimensionless.  Either pass a precomputed table or a model
    and lmax; the table used is returned.
    """
    if table is None:
        if model is None or lmax is None:
            raise ValueError("give either table= or both model and lmax")
        table = love_number_table(model, lmax, ngll=ngll, rmax=rmax,
                                  eps=eps, G=G, moduli=moduli)
    cols = np.column_stack([table["l"], table["h_u"], table["k_u"],
                            table["h_phi"], table["k_phi"],
                            table["h_t"], table["k_t"]])
    header = ("elastic Love numbers, Al-Attar convention (dimensional SI)\n"
              "l  h_u  k_u  h_phi  k_phi  h_t  k_t\n"
              "load columns per unit surface density; tidal columns per "
              "unit external potential (r/a)^l; degree 1 in the "
              "centre-of-mass frame")
    np.savetxt(path, cols, fmt=["%6d"] + ["%+.12e"] * 6, header=header)
    return table
