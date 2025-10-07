import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def meta_funcion(f):
    """Devuelve (nombre, color, x_opt, y_opt, f_opt) según la función objetivo."""
    if f.__name__ == "esfera":
        return "Función de Esfera", "orange", 0.0, 0.0, 0.0
    if f.__name__ == "bukin":
        return "Función de Bukin", "red",   -10.0, 1.0, 0.0
    if f.__name__ == "himmelblau":
        return "Función de Himmelblau", "blue", 3.584428, -1.848126, 0.0
    if f.__name__ == "eggholder":
        return "Función de Eggholder",  "green", 512.0, 404.2319, -959.6407
    if f.__name__ == "easom":
        return "Función de Easom",   "purple", np.pi, np.pi, -1.0
    return f.__name__, "gray", None, None, None



def graficar_individual(hist_best, hist_mean, f, operador, elitismo_modo, elitismo_percent=0.0, outdir="Imagenes"):
    outdir = Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
    nombre, color, x_opt, y_opt, f_opt = meta_funcion(f)

    titulo = f"AG - {nombre} | Sel: {operador.upper()} | Elitismo: {elitismo_modo}"
    
    if elitismo_modo == "top_percent":
        titulo += f"({int(100*elitismo_percent)}%)"

    plt.figure(figsize=(14,7))
    plt.plot(hist_best,  linestyle="--", marker=None, alpha=0.9, label="Mejor")
    plt.plot(hist_mean,  linestyle="-",  marker=None, alpha=0.9, label="Media")
    if f_opt is not None:
        plt.axhline(f_opt, color="black", linestyle=":", label=f"Óptimo real f={f_opt:.4f}")
    plt.xlabel("Generación"); plt.ylabel("Fitness")
    plt.title(titulo)
    plt.legend(); plt.grid(True); plt.tight_layout()

    fname = titulo.replace(" ", "_").replace("|", "-") + ".png"
    path = outdir / fname
    plt.savefig(path, dpi=160); plt.close()
    print(f"[OK] Gráfica guardada → {path}")
    return path


def graficar_comparativa(traces_dict, outdir="Imagenes", titulo_extra=""):
    """
    traces_dict: { "RUEDA | elit:top(5%)": (hist_best, hist_mean, f), ... }
    """
    outdir = Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(14,7))

    # asegurar misma longitud
    L = min(len(v[0]) for v in traces_dict.values())
    for label, (hb, hm, f) in traces_dict.items():
        hb = hb[:L]; hm = hm[:L]
        plt.plot(hb, linestyle="--", alpha=0.9, label=f"{label} - Mejor")
        plt.plot(hm, linestyle="-",  alpha=0.9, label=f"{label} - Media")

    plt.xlabel("Generación"); plt.ylabel("Fitness")
    plt.title(f"Comparativa de Operadores + Elitismo {titulo_extra}".strip())
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True); plt.tight_layout()

    fname = "Comparativa_Operadores_Elitismo"
    if titulo_extra: fname += "_" + titulo_extra.replace(" ","_")
    path = outdir / f"{fname}.png"
    plt.savefig(path, dpi=160); plt.close()
    print(f"[OK] Gráfica comparativa guardada → {path}")
    return path
