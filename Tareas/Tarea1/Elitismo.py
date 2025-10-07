import math
import random


# =================== Elitismo: selección e inserción ===================

def seleccionar_elites(poblacion, modo="best_one", percent=0.05):
    """
    Devuelve lista de élites (copias) ordenadas por fitness ascendente (minimización).
    modo: "best_one" | "top_percent" | "total"
    """
    orden = sorted(poblacion, key=lambda d: d["fitness"])  # mejor primero
    if modo == "best_one":
        elites = orden[:1]
    elif modo == "top_percent":
        n = max(1, int(math.ceil(percent * len(poblacion))))
        elites = orden[:n]
    elif modo == "total":
        elites = orden[:]  # todos
    else:
        raise ValueError("Modo de elitismo inválido")
    # Recordemos que los individuos son de esta forma: 
        # item = {
        #         "cromosoma": individuo, 
        #         "x":x,
        #         "y":y,
        #         "fitness": None (porque no se ha hecho una evaluación hasta el momento)
        #     }
    elites = [dict(cromosoma=e["cromosoma"][:], x=e["x"], y=e["y"], fitness=e["fitness"]) for e in elites]
    return elites


def insertar_elites(next_poblacion, elites, *, mecanismo="replace_worst"):
    """
    Inserta élites en la siguiente generación. OJITO: **saltando cruza/mutación**.
    mecanismo:
      - "replace_worst": reemplaza peores por élites
      - "antepon": antepone élites y recorta
    """
    if not elites:
        return next_poblacion

    if mecanismo == "replace_worst":
        # ordenar 'next' por fitness descendente (peor al frente)
        next_sorted = sorted(next_poblacion, key=lambda d: d["fitness"], reverse=True)
        k = min(len(elites), len(next_sorted))
        # reemplazar peores k por élites
        for i in range(k):
            next_sorted[i] = elites[i]
        # remezclar para no inducir errores
        random.shuffle(next_sorted)
        return next_sorted

    elif mecanismo == "antepon":
        cat = elites + next_poblacion
        return cat[:len(next_poblacion)]

    else:
        raise ValueError("Mecanismo de inserción inválido")
