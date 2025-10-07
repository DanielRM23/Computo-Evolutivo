
import numpy as np


# =================== Operadores de selección adicionales ===================

def seleccion_sus(poblacion, probas, k):
    """
        SUS: n=k puntos equidistantes en [0,1), con "offset" aleatorio en [0,1/k).
        Menor varianza que Ruleta, misma proporcionalidad.
            -Observación: Offset es un desplazamiento inicial aleatorio que determina la posición del primer punto de muestreo.
            Así, los puntos de selección se distribuyen uniformemente pero empiezan desde una posición aleatoria dentro del primer intervalo.
    """
    probas = np.asarray(probas, dtype=float)
    cum = np.cumsum(probas)
    step = 1.0 / k
    start = np.random.random() * step
    pointers = start + step * np.arange(k)
    idxs = np.searchsorted(cum, pointers, side="right")
    seleccion = [poblacion[i] for i in idxs]
    return seleccion, idxs


def seleccion_torneo(poblacion,
                     k_seleccion,
                     *,
                     k=3,
                     with_replacement=True,
                     ttype="deterministic",   # "deterministic" | "probabilistic"
                     p=0.75,                  # prob. de elegir al mejor en probab.
                     tie_breaker="random"     # "random" | "first"
                     ):
    """
    Torneo:
      - deterministic: gana el de mayor aptitud (awas! aquí minimizamos fitness)
      - probabilistic: posiciones ordenadas por fitness; con prob p el mejor, (1-p) bajar
    Empates: 'random' elige al azar entre empatados; 'first' toma el primero.
    """
    N = len(poblacion)
    out = []

    # Aptitud para torneo con problema de minimización:
    # menor fitness ⇒ mejor. Usamos -fitness como 'aptitud' ordinal.
    fits = np.array([ind["fitness"] for ind in poblacion], dtype=float)

    for _ in range(k_seleccion):
        if with_replacement:
            cand_idx = np.random.randint(0, N, size=k)
        else:
            cand_idx = np.random.choice(N, size=k, replace=False)

        f_c = fits[cand_idx]
        # Orden ascendente por fitness (minimización); el más pequeño es el mejor
        order = np.argsort(f_c)
        cand_sorted = cand_idx[order]
        f_sorted = f_c[order]

        if ttype == "deterministic":
            best_f = f_sorted[0]
            eq = cand_sorted[f_sorted == best_f]
            winner_idx = np.random.choice(eq) if tie_breaker == "random" else eq[0]
        else:
            # probabilístico: distribución geométrica truncada sobre posiciones 0..k-1
            probs = np.array([p * (1 - p)**i for i in range(k)], dtype=float)
            probs /= probs.sum()
            pos = np.random.choice(k, p=probs)
            level_f = f_sorted[pos]
            eq = cand_sorted[f_sorted == level_f]
            winner_idx = np.random.choice(eq) if tie_breaker == "random" else eq[0]

        out.append(poblacion[winner_idx])

    return out, None


def seleccion_aleatoria(poblacion, k):
    """Baseline: selección uniforme sin presión de selección."""
    idxs = np.random.randint(0, len(poblacion), size=k)
    seleccion = [poblacion[i] for i in idxs]
    return seleccion, idxs
