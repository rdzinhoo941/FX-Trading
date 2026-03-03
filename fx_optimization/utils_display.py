"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  FONCTIONS D'AFFICHAGE CONSOLE                                                 ║
║                                                                                ║
║  Utilisées pour formater proprement l'affichage dans le terminal :            ║
║  • hdr()  : affiche un header avec des ====                                   ║
║  • sub()  : affiche un sous-titre avec des ──                                 ║
║  • tbl()  : affiche un tableau ASCII joliment formaté                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

def hdr(title: str):
    """Affiche un header formaté dans la console."""
    print(f"\n{'='*80}\n  {title}\n{'='*80}")

def sub(title: str):
    """Affiche un sous-titre formaté."""
    print(f"\n  ── {title} ──")

def tbl(headers: list, rows: list):
    """
    Affiche un tableau ASCII avec des bordures.
    
    Exemple:
        tbl(["Nom", "Age"], [["Alice", 25], ["Bob", 30]])
    """
    # Calculer la largeur de chaque colonne
    cw = []
    for i, h in enumerate(headers):
        max_width = len(str(h))
        for r in rows:
            max_width = max(max_width, len(str(r[i])))
        cw.append(max_width + 2)  # +2 pour l'espacement
    
    # Créer la ligne de séparation
    sep = "+" + "+".join("-" * w for w in cw) + "+"
    
    # Fonction pour formater une ligne
    def format_row(v):
        return "|" + "|".join(str(x).center(w) for x, w in zip(v, cw)) + "|"
    
    # Afficher le tableau
    print(sep)
    print(format_row(headers))
    print(sep)
    for r in rows:
        print(format_row(r))
    print(sep)


def print_progress(current: int, total: int, bar_length: int = 50):
    """Affiche une barre de progression."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f'\r  Progression: |{bar}| {percent:.1%}', end='\r')
    if current == total:
        print()