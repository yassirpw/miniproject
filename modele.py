"""
============================================================
MODELE.PY — Moteur physique du mini-projet
============================================================
Caractérisation et estimation de l'épaisseur d'une couche
métallique par analyse des coefficients de réflexion/transmission.

Configuration : Air (1) / Métal (2, épaisseur e) / Air (3)

Références cours :
  - Chapitre 1 : Lois de Descartes, coefficients de Fresnel
  - Chapitre 2 : Milieux conducteurs ohmiques, gamma = alpha + j*beta,
                 épaisseur de peau delta
  - Chapitre 3 : Superposition d'ondes, coefficient de réflexion Gamma
============================================================
"""

import numpy as np

# ============================================================
# 1. CONSTANTES PHYSIQUES FONDAMENTALES
# ============================================================

# Vitesse de la lumière dans le vide (m/s)
c = 3.0e8

# Perméabilité du vide (H/m)
mu0 = 4 * np.pi * 1e-7

# Permittivité du vide (F/m)
eps0 = 8.8541878128e-12

# Impédance d'onde du vide (Ohm) — Chapitre 1 : eta0 = sqrt(mu0/eps0)
eta0 = np.sqrt(mu0 / eps0)  # ≈ 376.730 Ohm

# ============================================================
# 2. CONDUCTIVITÉS DES MÉTAUX (données du sujet)
# ============================================================
# Le sujet impose ces valeurs (page 1 du PDF)
# On les suppose constantes sur 1-10 GHz

SIGMA_CUIVRE = 5.8e7  # S/m
SIGMA_ALUMINIUM = 3.5e7  # S/m
SIGMA_ACIER = 1.4e6  # S/m

# Dictionnaire pratique pour itérer sur les métaux
METAUX = {
    "Cuivre": SIGMA_CUIVRE,
    "Aluminium": SIGMA_ALUMINIUM,
    "Acier": SIGMA_ACIER,
}


# ============================================================
# 3. FONCTIONS PHYSIQUES — MÉTAL (Chapitre 2)
# ============================================================

def delta_peau(sigma, f):
    """
    Épaisseur de peau d'un bon conducteur ohmique.

    Chapitre 2, section C.4 :
        delta = sqrt(2 / (mu0 * sigma * omega))

    avec omega = 2*pi*f.

    Interprétation physique :
        - Profondeur à laquelle l'amplitude de l'onde est atténuée
          d'un facteur e (≈ 2.718)
        - Unité : mètre (m)

    Paramètres
    ----------
    sigma : float
        Conductivité électrique (S/m)
    f : float
        Fréquence (Hz)

    Retour
    ------
    float
        Épaisseur de peau (m)
    """
    omega = 2 * np.pi * f
    return np.sqrt(2.0 / (mu0 * sigma * omega))


def gamma_metal(sigma, f):
    """
    Constante de propagation complexe dans un bon conducteur.

    Chapitre 2, section C.4 :
        gamma = (1 + j) / delta

    Interprétation : gamma = alpha + j*beta avec
        - alpha = 1/delta (atténuation, Np/m)
        - beta  = 1/delta (phase, rad/m)

    Paramètres
    ----------
    sigma : float
        Conductivité (S/m)
    f : float
        Fréquence (Hz)

    Retour
    ------
    complex
        Constante de propagation (m^-1)
    """
    d = delta_peau(sigma, f)
    return (1.0 + 1.0j) / d


def eta_metal(sigma, f):
    """
    Impédance d'onde complexe d'un bon conducteur.

    Formule équivalente à Chapitre 2, section B.5 :
        eta_m = j*omega*mu0 / gamma = sqrt(j*omega*mu0 / sigma)

    Propriété importante :
        eta_m est PETITE devant eta0 (typiquement ~0.01 Ohm
        contre 377 Ohm pour l'air) → le métal est proche d'un
        court-circuit pour l'onde.

    Paramètres
    ----------
    sigma : float
        Conductivité (S/m)
    f : float
        Fréquence (Hz)

    Retour
    ------
    complex
        Impédance d'onde (Ohm)
    """
    omega = 2 * np.pi * f
    return np.sqrt(1.0j * omega * mu0 / sigma)


# ============================================================
# 4. COEFFICIENTS DE FRESNEL (Chapitre 1)
# ============================================================

def fresnel_normal(eta1, eta2):
    """
    Coefficients de Fresnel en incidence NORMALE (theta = 0).

    Chapitre 1, relations de structure (page 11) :
        r_12 = (eta2 - eta1) / (eta2 + eta1)
        t_12 = 2*eta2 / (eta2 + eta1)

    Attention : ces formules sont valables UNIQUEMENT en
    incidence normale. Pour l'incidence oblique (Étude 6),
    on utilisera d'autres formules (TE/TM).

    Paramètres
    ----------
    eta1, eta2 : complex
        Impédances des milieux 1 et 2 (Ohm)

    Retour
    ------
    (complex, complex)
        (r_12, t_12) : coefficients de réflexion et transmission
        en amplitude
    """
    r = (eta2 - eta1) / (eta2 + eta1)
    t = 2.0 * eta2 / (eta2 + eta1)
    return r, t


# ============================================================
# 5. COUCHE MINCE — RÉFLEXIONS MULTIPLES (Chapitre 3)
# ============================================================

def gamma_couche(eta1, eta2, eta3, gamma2, e):
    """
    Coefficients de réflexion et transmission d'une couche
    d'épaisseur finie, en tenant compte des réflexions
    multiples aux deux interfaces.

    Démonstration (somme géométrique) :
        L'onde subit :
          - une première réflexion r_12 à l'interface 1
          - une transmission t_12, puis t_21 à l'intérieur
          - des allers-retours amortis par exp(-2*gamma2*e)
        La somme de toutes ces contributions donne :

        Gamma = (r_12 + r_23 * exp(-2*gamma2*e))
                / (1 + r_12 * r_23 * exp(-2*gamma2*e))

        T     = (t_12 * t_23 * exp(-gamma2*e))
                / (1 + r_12 * r_23 * exp(-2*gamma2*e))

    Paramètres
    ----------
    eta1 : complex
        Impédance du milieu 1 (air, ≈ 377 Ohm)
    eta2 : complex
        Impédance du milieu 2 (métal)
    eta3 : complex
        Impédance du milieu 3 (air, ≈ 377 Ohm)
    gamma2 : complex
        Constante de propagation dans la couche (m^-1)
    e : float
        Épaisseur de la couche (m)

    Retour
    ------
    (complex, complex)
        (Gamma, T) : coefficients complexes de réflexion et
        transmission de la STRUCTURE COMPLETE
    """
    # Coefficients de Fresnel aux deux interfaces
    r12, t12 = fresnel_normal(eta1, eta2)  # interface 1 -> 2
    r23, t23 = fresnel_normal(eta2, eta3)  # interface 2 -> 3

    # Facteur de propagation aller-retour dans la couche
    # exp(-2*gamma2*e) = exp(-2*alpha*e) * exp(-j*2*beta*e)
    exp_terme = np.exp(-2.0 * gamma2 * e)

    # Dénominateur commun (somme géométrique des réflexions multiples)
    denominateur = 1.0 + r12 * r23 * exp_terme

    # Coefficient de réflexion total vu depuis le milieu 1
    Gamma = (r12 + r23 * exp_terme) / denominateur

    # Coefficient de transmission total vers le milieu 3
    T = (t12 * t23 * np.exp(-gamma2 * e)) / denominateur

    return Gamma, T


# ============================================================
# 6. COEFFICIENTS ÉNERGÉTIQUES
# ============================================================

def RTA(Gamma, T):
    """
    Coefficients énergétiques de réflexion, transmission et
    absorption.

    ATTENTION : la formule T_en = |T|^2 est valable ici car
    les milieux 1 et 3 sont IDENTIQUES (air/air). Si les milieux
    étaient différents, il faudrait un facteur correctif
    (eta1/eta3) ou (cos theta3 / cos theta1).

    Conservation de l'énergie :
        R + T + A = 1  (doit toujours être vérifié)

    Paramètres
    ----------
    Gamma : complex
        Coefficient de réflexion (amplitude)
    T : complex
        Coefficient de transmission (amplitude)

    Retour
    ------
    (float, float, float)
        (R, T_en, A) avec :
          - R = |Gamma|^2 (réflexion)
          - T_en = |T|^2 (transmission)
          - A = 1 - R - T_en (absorption)
    """
    R = np.abs(Gamma) ** 2
    T_en = np.abs(T) ** 2
    A = 1.0 - R - T_en
    return R, T_en, A


# ============================================================
# 7. FONCTION UTILITAIRE — CALCUL COMPLET
# ============================================================

def calcul_complet(sigma, f, e):
    """
    Fonction "tout-en-un" : calcule R, T, A pour un métal
    donné, à une fréquence et une épaisseur données.

    C'est la fonction qu'on appellera dans les études 2 à 5.

    Paramètres
    ----------
    sigma : float
        Conductivité du métal (S/m)
    f : float
        Fréquence (Hz)
    e : float
        Épaisseur de la couche (m)

    Retour
    ------
    dict
        Dictionnaire contenant :
          - 'delta' : épaisseur de peau (m)
          - 'eta_m' : impédance du métal (Ohm)
          - 'gamma' : constante de propagation (m^-1)
          - 'Gamma' : coefficient de réflexion complexe
          - 'T'     : coefficient de transmission complexe
          - 'R'     : coefficient de réflexion énergétique
          - 'T_en'  : coefficient de transmission énergétique
          - 'A'     : coefficient d'absorption
    """
    eta_m = eta_metal(sigma, f)
    gam = gamma_metal(sigma, f)
    Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e)
    R, T_en, A = RTA(Gamma, T)

    return {
        "delta": delta_peau(sigma, f),
        "eta_m": eta_m,
        "gamma": gam,
        "Gamma": Gamma,
        "T": T,
        "R": R,
        "T_en": T_en,
        "A": A,
    }


# ============================================================
# 8. TEST RAPIDE (si on exécute modele.py directement)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODELE.PY — Vérification rapide")
    print("=" * 60)
    print(f"eta0 = {eta0:.4f} Ohm")
    print(f"c    = {c:.3e} m/s")
    print()

    f_test = 5e9  # 5 GHz
    e_test = 1e-6  # 1 µm

    print(f"Test à f = {f_test / 1e9} GHz, e = {e_test * 1e6} µm :")
    print("-" * 60)

    for nom, sig in METAUX.items():
        res = calcul_complet(sig, f_test, e_test)
        print(f"{nom:10s} | delta = {res['delta'] * 1e6:6.3f} µm | "
              f"|eta_m| = {abs(res['eta_m']):.4f} Ohm")
        print(f"{'':10s} | R = {res['R']:.6f} | "
              f"T = {res['T_en']:.6f} | A = {res['A']:.6f}")
        print(f"{'':10s} | R+T+A = {res['R'] + res['T_en'] + res['A']:.10f}")
        print()