import matplotlib.pyplot as plt


def dessiner_schema():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')  # on enlève les axes

    # ============================================================
    # 1. LES TROIS MILIEUX
    # ============================================================
    # Milieu 1 (Air) : x < 3
    ax.axvspan(-1, 3, color='#E8F4FD', alpha=0.6, label='Milieu 1 (Air)')
    # Milieu 2 (Métal) : 3 < x < 6
    ax.axvspan(3, 6, color='#B0B0B0', alpha=0.8, label='Milieu 2 (Métal)')
    # Milieu 3 (Air) : x > 6
    ax.axvspan(6, 11, color='#E8F4FD', alpha=0.6, label='Milieu 3 (Air)')

    # Interfaces
    ax.axvline(3, color='black', linewidth=2, linestyle='--')
    ax.axvline(6, color='black', linewidth=2, linestyle='--')

    # ============================================================
    # 2. L'ONDE INCIDENTE (flèche vers la droite, dans milieu 1)
    # ============================================================
    ax.annotate('', xy=(2.8, 1.5), xytext=(0.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(1.3, 1.9, r'$\vec{E}_i$', color='red', fontsize=14, fontweight='bold')
    ax.text(1.3, 1.5, r'$\vec{k}_i$', color='red', fontsize=12)

    # ============================================================
    # 3. L'ONDE RÉFLÉCHIE (flèche vers la gauche)
    # ============================================================
    ax.annotate('', xy=(0.5, -1.0), xytext=(2.8, -0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(1.3, -1.2, r'$\vec{E}_r$', color='blue', fontsize=14, fontweight='bold')
    ax.text(1.3, -0.8, r'$\vec{k}_r$', color='blue', fontsize=12)

    # ============================================================
    # 4. L'ONDE TRANSMISE (flèche vers la droite, dans milieu 3)
    # ============================================================
    ax.annotate('', xy=(10.5, -1.5), xytext=(6.2, -0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(8.3, -1.5, r'$\vec{E}_t$', color='green', fontsize=14, fontweight='bold')
    ax.text(8.3, -1.1, r'$\vec{k}_t$', color='green', fontsize=12)

    # ============================================================
    # 5. L'ÉPAISSEUR e (double flèche dans le métal)
    # ============================================================
    ax.annotate('', xy=(3, 2.2), xytext=(6, 2.2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(4.5, 2.35, r'$e$', fontsize=14, fontweight='bold')

    # ============================================================
    # 6. LES IMPÉDANCES
    # ============================================================
    ax.text(1.0, 0.0, r'$\eta_0$', fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(4.5, 0.0, r'$\eta_m, \sigma$', fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(8.5, 0.0, r'$\eta_0$', fontsize=14, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # ============================================================
    # 7. TITRES ET LÉGENDES
    # ============================================================
    ax.text(1.0, -2.2, 'Milieu 1\n(Air)', ha='center', fontsize=11)
    ax.text(4.5, -2.2, 'Milieu 2\n(Métal)', ha='center', fontsize=11)
    ax.text(8.5, -2.2, 'Milieu 3\n(Air)', ha='center', fontsize=11)

    ax.set_title('Structure Air / Métal / Air — Réflexion et transmission',
                 fontsize=14, fontweight='bold')

    # ============================================================
    # 8. AXE Z (normale à l'interface)
    # ============================================================
    ax.annotate('', xy=(4.5, -0.2), xytext=(4.5, -2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.text(4.7, -2.3, r'$z$', fontsize=12)

    plt.tight_layout()
    plt.savefig('schema_situation.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    dessiner_schema()