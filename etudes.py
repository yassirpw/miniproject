"""
============================================================
ETUDES.PY — Les 6 études du mini-projet
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import modele
from modele import (
    eta0, eps0, METAUX,
    delta_peau, gamma_metal, eta_metal,
    gamma_couche, RTA, calcul_complet,
)

# Style global
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})

# Couleurs des métaux
COULEURS = {
    "Cuivre": "#B87333",
    "Aluminium": "#A8A8A8",
    "Acier": "#4A4A4A",
}


# ============================================================
# ÉTUDE 1
# ============================================================
def etude_1_comparaison_metaux():
    print("=" * 65, flush=True)
    print("  ÉTUDE 1 : Comparaison des 3 métaux (f = 5 GHz)", flush=True)
    print("=" * 65, flush=True)

    f = 5e9
    e_list = np.logspace(-7, -3, 400)

    courbes = {}
    for nom, sigma in METAUX.items():
        eta_m = eta_metal(sigma, f)
        gam = gamma_metal(sigma, f)
        R_l, T_l, A_l = [], [], []
        for e in e_list:
            Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e)
            R, T_en, A = RTA(Gamma, T)
            R_l.append(R); T_l.append(T_en); A_l.append(A)
        courbes[nom] = {"R": R_l, "T": T_l, "A": A_l}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for nom, data in courbes.items():
        axes[0].semilogx(e_list, data["R"], label=nom, color=COULEURS[nom])
        axes[1].semilogx(e_list, data["T"], label=nom, color=COULEURS[nom])
        axes[2].semilogx(e_list, data["A"], label=nom, color=COULEURS[nom])

    for ax, titre, ylabel in zip(
        axes, ["Réflexion R(e)", "Transmission T(e)", "Absorption A(e)"], ["R", "T", "A"]
    ):
        ax.set_xlabel("Épaisseur e (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{titre} — f = 5 GHz")
        ax.legend()
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig("fig_etude1_comparaison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n  Tableau de valeurs :", flush=True)
    print(f"  {'e (µm)':>8s} | {'Métal':10s} | {'R':>10s} | {'T':>12s} | {'A':>10s}", flush=True)
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}", flush=True)
    for e_um in [0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000]:
        for nom, sigma in METAUX.items():
            res = calcul_complet(sigma, f, e_um * 1e-6)
            print(f"  {e_um:8.1f} | {nom:10s} | {res['R']:10.6f} | {res['T_en']:12.3e} | {res['A']:10.6f}", flush=True)
        print(flush=True)


# ============================================================
# ÉTUDE 2
# ============================================================
def etude_2_influence_epaisseur():
    print("=" * 65, flush=True)
    print("  ÉTUDE 2 : Influence de l'épaisseur (f = 5 GHz)", flush=True)
    print("=" * 65, flush=True)

    f = 5e9
    e_list = np.logspace(-7, -3, 500)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for nom, sigma in METAUX.items():
        eta_m = eta_metal(sigma, f)
        gam = gamma_metal(sigma, f)
        delta = delta_peau(sigma, f)
        R_l, T_l, A_l = [], [], []
        for e in e_list:
            Gamma, T = gamma_couche(eta0, eta_m, eta0, gam, e)
            R, T_en, A = RTA(Gamma, T)
            R_l.append(R); T_l.append(T_en); A_l.append(A)

        axes[0].semilogx(e_list, R_l, label=f"R ({nom})", color=COULEURS[nom])
        axes[0].loglog(e_list, T_l, "--", label=f"T ({nom})", color=COULEURS[nom])
        axes[0].semilogx(e_list, A_l, ":", label=f"A ({nom})", color=COULEURS[nom])
        axes[0].axvline(delta, color=COULEURS[nom], alpha=0.3, linestyle="-.")

        sens = np.abs(np.gradient(R_l, e_list))
        axes[1].loglog(e_list, sens, label=nom, color=COULEURS[nom])

    axes[0].set_xlabel("Épaisseur e (m)")
    axes[0].set_ylabel("Coefficient (R, A linéaire / T log)")
    axes[0].set_title("R, T, A vs épaisseur (f = 5 GHz)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_ylim(1e-15, 1.1)

    axes[1].set_xlabel("Épaisseur e (m)")
    axes[1].set_ylabel("|dR/de| (m⁻¹)")
    axes[1].set_title("Sensibilité de R à l'épaisseur")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fig_etude2_epaisseur.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n  Zones d'épaisseur :", flush=True)
    for nom, sigma in METAUX.items():
        delta = delta_peau(sigma, f)
        print(f"    {nom:10s} : δ = {delta*1e6:6.2f} µm → zone ≈ [{0.1*delta*1e6:.2f}, {3*delta*1e6:.1f}] µm", flush=True)


# ============================================================
# ÉTUDE 3
# ============================================================
def etude_3_influence_frequence():
    print("=" * 65, flush=True)
    print("  ÉTUDE 3 : Influence de la fréquence (1-10 GHz)", flush=True)
    print("=" * 65, flush=True)

    sigma = METAUX["Cuivre"]
    f_list = np.linspace(1e9, 10e9, 100)
    epaisseurs = [0.5e-6, 1e-6, 2e-6, 5e-6, 10e-6, 50e-6]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for e in epaisseurs:
        R_l, T_l, A_l = [], [], []
        for f in f_list:
            res = calcul_complet(sigma, f, e)
            R_l.append(res["R"]); T_l.append(res["T_en"]); A_l.append(res["A"])
        label = f"e = {e*1e6:.1f} µm"
        axes[0].plot(f_list/1e9, R_l, label=label)
        axes[1].semilogy(f_list/1e9, T_l, label=label)
        axes[2].plot(f_list/1e9, A_l, label=label)

    axes[0].set_title("Réflexion R(f) — Cuivre")
    axes[1].set_title("Transmission T(f) — Cuivre")
    axes[2].set_title("Absorption A(f) — Cuivre")
    for ax in axes:
        ax.set_xlabel("Fréquence f (GHz)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("R"); axes[1].set_ylabel("T (log)"); axes[2].set_ylabel("A")

    plt.tight_layout()
    plt.savefig("fig_etude3_frequence.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("  Calcul des cartes 2D...", flush=True)
    f_grid = np.linspace(1e9, 10e9, 150)
    e_grid = np.logspace(-7, -3, 150)
    F, E = np.meshgrid(f_grid, e_grid)
    R_map = np.zeros_like(F); T_map = np.zeros_like(F)
    for i, f in enumerate(f_grid):
        for j, e in enumerate(e_grid):
            res = calcul_complet(sigma, f, e)
            R_map[j, i] = res["R"]; T_map[j, i] = res["T_en"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im0 = axes[0].pcolormesh(F/1e9, E*1e6, R_map, cmap="viridis", shading="auto")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fréquence f (GHz)"); axes[0].set_ylabel("Épaisseur e (µm)")
    axes[0].set_title("Carte R(f, e) — Cuivre")
    plt.colorbar(im0, ax=axes[0], label="R")

    im1 = axes[1].pcolormesh(F/1e9, E*1e6, np.log10(T_map + 1e-30), cmap="plasma", shading="auto")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Fréquence f (GHz)"); axes[1].set_ylabel("Épaisseur e (µm)")
    axes[1].set_title("Carte log₁₀ T(f, e) — Cuivre")
    plt.colorbar(im1, ax=axes[1], label="log₁₀(T)")

    plt.tight_layout()
    plt.savefig("fig_etude3_cartes2D.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# ÉTUDE 4
# ============================================================
def etude_4_profondeur_peau():
    print("=" * 65, flush=True)
    print("  ÉTUDE 4 : Profondeur de peau (1-10 GHz)", flush=True)
    print("=" * 65, flush=True)

    f_list = np.linspace(1e9, 10e9, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    for nom, sigma in METAUX.items():
        deltas = [delta_peau(sigma, f) for f in f_list]
        ax.loglog(f_list/1e9, np.array(deltas)*1e6, label=nom, color=COULEURS[nom])

    for e_um in [0.1, 1, 10, 100, 1000]:
        ax.axhline(e_um, color="gray", alpha=0.3, linestyle="--")
        ax.text(10.2, e_um, f"{e_um} µm", fontsize=8, color="gray", va="center")

    ax.set_xlabel("Fréquence f (GHz)")
    ax.set_ylabel("Profondeur de peau δ (µm)")
    ax.set_title("Profondeur de peau vs fréquence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_etude4_profondeur_peau.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n  {'f (GHz)':>8s} | " + " | ".join(f"{nom:>10s}" for nom in METAUX) + "  (δ en µm)", flush=True)
    for f in [1e9, 2e9, 5e9, 8e9, 10e9]:
        vals = [delta_peau(sigma, f) * 1e6 for sigma in METAUX.values()]
        print(f"  {f/1e9:8.1f} | " + " | ".join(f"{v:10.3f}" for v in vals), flush=True)


# ============================================================
# ÉTUDE 5
# ============================================================
def etude_5_estimation():
    print("=" * 65, flush=True)
    print("  ÉTUDE 5 : Estimation numérique de l'épaisseur", flush=True)
    print("=" * 65, flush=True)

    np.random.seed(42)
    f = 5e9
    sigma = METAUX["Cuivre"]
    e_vraies = [0.5e-6, 1e-6, 1.5e-6, 2e-6]
    bruits = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

    print(f"\n  {'e_vrai (µm)':>12s} | {'bruit':>7s} | {'e_estimé (µm)':>15s} | {'erreur %':>10s}", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for e_vraie in e_vraies:
        res = calcul_complet(sigma, f, e_vraie)
        R_vrai = res["R"]; T_vrai = res["T_en"]
        erreurs = []
        for bruit in bruits:
            if bruit > 0:
                R_mes = R_vrai * (1 + bruit * np.random.randn())
                T_mes = T_vrai * (1 + bruit * np.random.randn())
            else:
                R_mes = R_vrai; T_mes = T_vrai

            def cout(e_est):
                if e_est <= 0: return 1e10
                res_mod = calcul_complet(sigma, f, e_est)
                dR = (R_mes - res_mod["R"]) ** 2
                T_mod = max(res_mod["T_en"], 1e-30)
                T_obs = max(T_mes, 1e-30)
                dT = (np.log10(T_obs) - np.log10(T_mod)) ** 2
                return dR + 1e-2 * dT

            e_grid = np.logspace(-8, -3, 1000)
            couts = np.array([cout(e) for e in e_grid])
            e_approx = e_grid[np.argmin(couts)]

            result = minimize_scalar(
                cout, bounds=(max(1e-9, e_approx/2), min(1e-3, e_approx*2)),
                method="bounded", options={"xatol": 1e-14},
            )
            e_est = result.x
            err = abs(e_est - e_vraie) / e_vraie * 100
            erreurs.append(err)
            print(f"  {e_vraie*1e6:12.2f} | {bruit*100:6.0f}% | {e_est*1e6:15.4f} | {err:10.3f}", flush=True)

        ax.semilogy(np.array(bruits)*100, erreurs, "o-", label=f"e = {e_vraie*1e6:.1f} µm")

    ax.set_xlabel("Niveau de bruit relatif (%)")
    ax.set_ylabel("Erreur relative d'estimation (%)")
    ax.set_title("Précision de l'estimation vs bruit (Cuivre, 5 GHz)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_etude5_estimation.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# ÉTUDE 6 — Incidence oblique
# ============================================================
def cos_theta_transmis(n_i, n_j, theta_i):
    sin_theta_j = n_i / n_j * np.sin(theta_i)
    return np.sqrt(1.0 - sin_theta_j**2 + 0j)

def fresnel_TE(eta_i, eta_j, theta_i, theta_j):
    num = eta_j / np.cos(theta_j) - eta_i / np.cos(theta_i)
    den = eta_j / np.cos(theta_j) + eta_i / np.cos(theta_i)
    return num / den

def fresnel_TM(eta_i, eta_j, theta_i, theta_j):
    num = eta_j * np.cos(theta_j) - eta_i * np.cos(theta_i)
    den = eta_j * np.cos(theta_j) + eta_i * np.cos(theta_i)
    return num / den

def gamma_couche_oblique(eta1, eta2, eta3, n1, n2, n3, gamma2, e, theta_i, polarisation="TE"):
    theta_2 = np.arccos(cos_theta_transmis(n1, n2, theta_i))
    theta_3 = np.arccos(cos_theta_transmis(n1, n3, theta_i))
    if polarisation == "TE":
        r12 = fresnel_TE(eta1, eta2, theta_i, theta_2)
        r23 = fresnel_TE(eta2, eta3, theta_2, theta_3)
    else:
        r12 = fresnel_TM(eta1, eta2, theta_i, theta_2)
        r23 = fresnel_TM(eta2, eta3, theta_2, theta_3)
    t12 = 1 + r12; t23 = 1 + r23
    e_eff = e / np.cos(theta_2) if abs(np.cos(theta_2)) > 1e-12 else e
    exp_term = np.exp(-2 * gamma2 * e_eff)
    denom = 1 + r12 * r23 * exp_term
    Gamma = (r12 + r23 * exp_term) / denom
    T = (t12 * t23 * np.exp(-gamma2 * e_eff)) / denom
    return Gamma, T

def RTA_oblique(Gamma, T, eta1, eta3, theta_i, theta_3):
    R = np.abs(Gamma)**2
    facteur = np.real(eta1 / eta3 * np.cos(theta_3) / np.cos(theta_i))
    T_en = np.abs(T)**2 * facteur
    A = 1 - R - T_en
    return R, T_en, A

def etude_6_oblique():
    print("=" * 65, flush=True)
    print("  ÉTUDE 6 : Angle et polarisation (TE/TM)", flush=True)
    print("=" * 65, flush=True)

    f = 5e9
    sigma = METAUX["Cuivre"]
    eta_m = eta_metal(sigma, f)
    gam_m = gamma_metal(sigma, f)
    omega = 2 * np.pi * f
    n_m = np.sqrt(1j * sigma / (omega * eps0))
    n_air = 1.0
    eta_air = eta0

    theta_deg = np.linspace(0, 70, 100)
    theta_rad = np.deg2rad(theta_deg)
    e_list = [0.5e-6, 1e-6, 2e-6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for e in e_list:
        R_TE, R_TM = [], []
        for th in theta_rad:
            G_TE, Tt_TE = gamma_couche_oblique(eta_air, eta_m, eta_air, n_air, n_m, n_air, gam_m, e, th, "TE")
            th3 = np.arccos(cos_theta_transmis(n_air, n_air, th))
            R_TE_val, _, _ = RTA_oblique(G_TE, Tt_TE, eta_air, eta_air, th, th3)
            R_TE.append(R_TE_val)

            G_TM, Tt_TM = gamma_couche_oblique(eta_air, eta_m, eta_air, n_air, n_m, n_air, gam_m, e, th, "TM")
            R_TM_val, _, _ = RTA_oblique(G_TM, Tt_TM, eta_air, eta_air, th, th3)
            R_TM.append(R_TM_val)

        label = f"e = {e*1e6:.1f} µm"
        axes[0, 0].plot(theta_deg, R_TE, label=label)
        axes[0, 1].plot(theta_deg, R_TM, label=label)

    e = 1e-6
    R_TE_1, R_TM_1 = [], []
    for th in theta_rad:
        G_TE, Tt_TE = gamma_couche_oblique(eta_air, eta_m, eta_air, n_air, n_m, n_air, gam_m, e, th, "TE")
        th3 = np.arccos(cos_theta_transmis(n_air, n_air, th))
        R_TE_val, _, _ = RTA_oblique(G_TE, Tt_TE, eta_air, eta_air, th, th3)
        R_TE_1.append(R_TE_val)

        G_TM, Tt_TM = gamma_couche_oblique(eta_air, eta_m, eta_air, n_air, n_m, n_air, gam_m, e, th, "TM")
        R_TM_val, _, _ = RTA_oblique(G_TM, Tt_TM, eta_air, eta_air, th, th3)
        R_TM_1.append(R_TM_val)

    axes[1, 0].plot(theta_deg, R_TE_1, "b-", label="TE", linewidth=2)
    axes[1, 0].plot(theta_deg, R_TM_1, "r-", label="TM", linewidth=2)

    diff = np.abs(np.array(R_TE_1) - np.array(R_TM_1))
    axes[1, 1].plot(theta_deg, diff, "k-", linewidth=2, label="|R_TE - R_TM|")

    axes[0, 0].set_title("Réflexion TE — Cuivre")
    axes[0, 1].set_title("Réflexion TM — Cuivre")
    axes[1, 0].set_title("Comparaison TE vs TM (e = 1 µm)")
    axes[1, 1].set_title("|R_TE - R_TM| (e = 1 µm)")

    for ax in axes.flat:
        ax.set_xlabel("Angle d'incidence (°)")
        ax.set_ylabel("R")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_etude6_oblique.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # etude_1_comparaison_metaux()
    # etude_2_influence_epaisseur()
    # etude_3_influence_frequence()
    # etude_4_profondeur_peau()
    # etude_5_estimation()
    # etude_6_oblique()
    etude_1_comparaison_metaux()
    etude_2_influence_epaisseur()
    etude_3_influence_frequence()
    etude_4_profondeur_peau()
    etude_5_estimation()
    etude_6_oblique()
