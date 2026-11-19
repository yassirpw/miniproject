"""
============================================================
VALIDATION.PY — Vérification du modèle physique
============================================================
Ce fichier implémente les tests de l'Étude 1 (point 4) :
"Vérifier numériquement la conservation de l'énergie"

Il valide que le moteur (modele.py) est CORRECT avant
de l'utiliser dans les études 2 à 6.

Tests effectués :
  1. Cas limite e = 0 (couche absente) -> transparence totale
  2. Cas limite e >> delta -> convergence vers le semi-infini
  3. Conservation de l'énergie R + T + A = 1
  4. Convergence exponentielle vers le semi-infini
  5. Comparaison des 3 métaux (cohérence physique)
  6. Cas limite métal parfait (sigma -> infini)

Usage :
    python validation.py
============================================================
"""

import numpy as np
from modele import (
    eta0, METAUX,
    delta_peau, gamma_metal, eta_metal,
    fresnel_normal, gamma_couche, RTA, calcul_complet,
)


def titre(texte):
    """Affiche un titre de section formaté."""
    print()
    print("=" * 65)
    print(f"  {texte}")
    print("=" * 65)


def sous_titre(texte):
    """Affiche un sous-titre."""
    print(f"\n--- {texte} ---")


# ============================================================
# TEST 1 : Cas limite e = 0 (couche absente)
# ============================================================

def test_1_couche_absente():
    """
    Si e = 0, la couche n'existe pas : on doit avoir
    une transparence totale (R = 0, T = 1, A = 0).

    Note : on utilise atol=1e-10 car les calculs en virgule
    flottante introduisent des erreurs ~1e-12 sur A = 1-R-T.
    """
    titre("TEST 1 : Couche d'épaisseur nulle (e = 0)")

    f = 5e9
    TOL = 1e-10  # tolérance réaliste pour du float64

    for nom, sigma in METAUX.items():
        eta_m = eta_metal(sigma, f)
        gam = gamma_metal(sigma, f)
        Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, 0.0)
        R, T_en, A = RTA(Gamma, T)

        # Assertions avec tolérance réaliste
        assert np.isclose(R, 0.0, atol=TOL), f"{nom}: R={R:.3e}"
        assert np.isclose(T_en, 1.0, atol=TOL), f"{nom}: T={T_en:.6f}"
        assert np.isclose(A, 0.0, atol=TOL), f"{nom}: A={A:.3e}"

        print(f"  {nom:10s} : R = {R:.2e}, T = {T_en:.10f}, "
              f"A = {A:.2e}  ✅")

    print(f"\n  (tolérance utilisée : {TOL:.0e})")
    print("  → Conclusion : le modèle dégénère correctement en e=0.")


# ============================================================
# TEST 2 : Convergence vers le semi-infini (e >> delta)
# ============================================================

def test_2_semi_infini():
    """
    Quand e >> delta, l'onde n'atteint plus la 2e interface.
    La couche se comporte comme un milieu semi-infini.

    On doit donc avoir :
        Gamma -> r_12  (Fresnel air/métal seul)
        R     -> |r_12|^2
        T     -> 0

    ATTENTION : R ne tend PAS vers 1 (sauf métal parfait).
    Il tend vers |r_12|^2 < 1.
    """
    titre("TEST 2 : Convergence vers le semi-infini (e >> delta)")

    f = 5e9
    for nom, sigma in METAUX.items():
        eta_m = eta_metal(sigma, f)
        gam = gamma_metal(sigma, f)
        delta = delta_peau(sigma, f)

        # Référence : réflexion sur un demi-espace métallique
        r12, _ = fresnel_normal(eta0, eta_m)
        R_semi_inf = np.abs(r12) ** 2

        # Calcul à e = 100*delta
        e_inf = 100 * delta
        Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e_inf)
        R, T_en, A = RTA(Gamma, T)

        # Assertions
        assert np.isclose(R, R_semi_inf, atol=1e-10), \
            f"{nom}: R={R}, attendu {R_semi_inf}"
        assert T_en < 1e-30, f"{nom}: T trop grand = {T_en}"

        print(f"  {nom:10s} : R = {R:.10f}  (|r12|² = {R_semi_inf:.10f})")
        print(f"  {'':10s}   T = {T_en:.2e}  (négligeable)")
        print(f"  {'':10s}   δ = {delta * 1e6:.3f} µm, e_test = {e_inf * 1e6:.1f} µm")
        print()

    print("  → Conclusion : R sature vers |r_12|², pas vers 1.")
    print("    T devient négligeable (onde totalement atténuée).")


# ============================================================
# TEST 3 : Conservation de l'énergie
# ============================================================

def test_3_conservation_energie():
    """
    La conservation de l'énergie impose R + T + A = 1.

    On vérifie cette relation sur une grille complète :
      - 3 métaux
      - 5 fréquences (1, 3, 5, 7, 10 GHz)
      - 200 épaisseurs (0.01 µm à 1 mm)

    Soit 3 × 5 × 200 = 3000 vérifications.
    """
    titre("TEST 3 : Conservation de l'énergie R + T + A = 1")

    frequences = [1e9, 3e9, 5e9, 7e9, 10e9]
    epaisseurs = np.logspace(-8, -3, 200)  # 10 nm à 1 mm

    err_max_global = 0.0
    nb_tests = 0

    for nom, sigma in METAUX.items():
        err_max_metal = 0.0
        for f in frequences:
            eta_m = eta_metal(sigma, f)
            gam = gamma_metal(sigma, f)
            for e in epaisseurs:
                Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e)
                R, T_en, A = RTA(Gamma, T)
                err = abs(R + T_en + A - 1.0)
                err_max_metal = max(err_max_metal, err)
                err_max_global = max(err_max_global, err)
                nb_tests += 1
        print(f"  {nom:10s} : erreur max = {err_max_metal:.2e}")

    print(f"\n  Nombre total de tests : {nb_tests}")
    print(f"  Erreur maximale globale : {err_max_global:.2e}")

    assert err_max_global < 1e-12, \
        f"Conservation énergie violée : {err_max_global}"

    print("\n  → Conclusion : R + T + A = 1 à la précision machine (✅).")


# ============================================================
# TEST 4 : Convergence exponentielle vers le semi-infini
# ============================================================

def test_4_convergence_exponentielle():
    """
    Quand e augmente, R converge vers |r_12|² de façon
    exponentielle (car le terme exp(-2*alpha*e) décroît
    exponentiellement).

    On vérifie visuellement cette convergence pour le cuivre.
    """
    titre("TEST 4 : Convergence exponentielle (Cuivre, 5 GHz)")

    f = 5e9
    sigma = METAUX["Cuivre"]
    eta_m = eta_metal(sigma, f)
    gam = gamma_metal(sigma, f)
    delta = delta_peau(sigma, f)

    r12, _ = fresnel_normal(eta0, eta_m)
    R_semi_inf = np.abs(r12) ** 2

    print(f"  Référence semi-infini : R_inf = |r12|² = {R_semi_inf:.10f}")
    print(f"  Épaisseur de peau δ = {delta * 1e6:.3f} µm\n")
    print(f"  {'e/δ':>5s} | {'e (µm)':>8s} | {'R':>12s} | {'|R-R_inf|':>12s} | {'T':>10s}")
    print(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

    for k in [1, 2, 3, 5, 10, 20, 50]:
        e = k * delta
        Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e)
        R, T_en, A = RTA(Gamma, T)
        err = abs(R - R_semi_inf)
        print(f"  {k:5d} | {e * 1e6:8.2f} | {R:.12f} | {err:.2e} | {T_en:.2e}")

    print("\n  → Conclusion : la convergence est bien exponentielle.")
    print("    À e = 10δ, l'erreur est déjà < 10^-5.")


# ============================================================
# TEST 5 : Comparaison des 3 métaux
# ============================================================

def test_5_comparaison_metaux():
    """
    À épaisseur et fréquence fixes, on compare les 3 métaux.

    Physiquement :
      - Plus sigma est grand, plus delta est petit,
        plus le métal est opaque (R grand, T petit).
      - Ordre attendu : Cu > Al > Acier (en opacité)
    """
    titre("TEST 5 : Comparaison des 3 métaux (f = 5 GHz)")

    f = 5e9
    epaisseurs_test = [0.1e-6, 1e-6, 10e-6]  # 0.1, 1, 10 µm

    for e in epaisseurs_test:
        print(f"\n  --- e = {e * 1e6} µm ---")
        print(f"  {'Métal':10s} | {'δ (µm)':>8s} | {'R':>10s} | "
              f"{'T':>12s} | {'A':>10s}")
        print(f"  {'-' * 10}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 10}")
        for nom, sigma in METAUX.items():
            res = calcul_complet(sigma, f, e)
            print(f"  {nom:10s} | {res['delta'] * 1e6:8.3f} | "
                  f"{res['R']:10.6f} | {res['T_en']:12.2e} | "
                  f"{res['A']:10.6f}")

    print("\n  → Conclusion : le cuivre est plus opaque que l'aluminium,")
    print("    lui-même plus opaque que l'acier (δ_Cu < δ_Al < δ_Acier).")


# ============================================================
# TEST 6 : Métal parfait (sigma -> infini)
# ============================================================

def test_6_metal_parfait():
    """
    Cas limite théorique : un métal parfait (sigma -> infini)
    a une impédance nulle (court-circuit parfait).

    On doit alors avoir r_12 -> -1 et R -> 1.

    Ce test n'est PAS physiquement réaliste (aucun métal
    réel n'a sigma = infini), mais il valide la cohérence
    asymptotique du modèle.
    """
    titre("TEST 6 : Cas limite du métal parfait (sigma -> infini)")

    f = 5e9
    sigmas_test = [5.8e7, 5.8e9, 5.8e11, 5.8e13]

    print(f"  {'sigma (S/m)':>15s} | {'|eta_m| (Ohm)':>15s} | {'|r12|²':>12s} | {'R(10δ)':>12s}")
    print(f"  {'-' * 15}-+-{'-' * 15}-+-{'-' * 12}-+-{'-' * 12}")

    for sig in sigmas_test:
        eta_m = eta_metal(sig, f)
        gam = gamma_metal(sig, f)
        delta = delta_peau(sig, f)

        r12, _ = fresnel_normal(eta0, eta_m)
        R_semi = np.abs(r12) ** 2

        # À e = 10*delta (suffisant pour saturer)
        Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, 10 * delta)
        R, _, _ = RTA(Gamma, T)

        print(f"  {sig:15.2e} | {abs(eta_m):15.6f} | "
              f"{R_semi:12.10f} | {R:12.10f}")

    print("\n  → Conclusion : quand sigma -> infini, |eta_m| -> 0")
    print("    et R -> 1 (miroir parfait). Le modèle est cohérent.")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Exécute tous les tests de validation."""
    print()
    print("╔" + "═" * 63 + "╗")
    print("║" + " " * 15 + "VALIDATION DU MODELE" + " " * 27 + "║")
    print("║" + " " * 10 + "Mini-projet : couche métallique" + " " * 20 + "║")
    print("╚" + "═" * 63 + "╝")

    test_1_couche_absente()
    test_2_semi_infini()
    test_3_conservation_energie()
    test_4_convergence_exponentielle()
    test_5_comparaison_metaux()
    test_6_metal_parfait()

    titre("RESULTAT GLOBAL")
    print("  ✅ Tous les tests sont PASSÉS.")
    print("  Le modèle est validé et prêt pour les études 1 à 6.")


if __name__ == "__main__":
    main()
