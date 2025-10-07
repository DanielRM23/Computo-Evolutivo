import random

from Tarea1 import*
from Tarea1 import EJECUCIONES

from Elitismo import seleccionar_elites, insertar_elites
from Operadores_Seleccion import seleccion_sus, seleccion_torneo, seleccion_aleatoria

from Utilidades import*


def AG_Simple_2(variables_rango,
              precision,
              M,
              G,
              p_cruce,
              f,
              # --- Selección ---
              operador_seleccion="ruleta",   # "ruleta"|"sus"|"torneo"|"aleatoria"
              torneo_params=None,            # dict: {"k":3,"with_replacement":True,"ttype":"deterministic","p":0.75,"tie_breaker":"random"}
              # --- Elitismo ---
              elitismo_modo="best_one",      # "best_one"|"top_percent"|"total"
              elitismo_percent=0.05,         # si top_percent
              elitismo_mecanismo="replace_worst"  # "replace_worst"|"prepend_then_trim"
              ):
    # === Preparación inicial (igual que tu versión) ===
    a1, b1 = variables_rango[0]
    a2, b2 = variables_rango[1]

    longitudes = longitud(variables_rango)
    valores = valores_a_representar(precision, longitudes)
    bits = bits_necesarios(valores)

    p_mut = 1.0 / suma_bits(bits)
    n = int(bits[0]); m = int(bits[-1])

    individuos = individuos_decodificados(M, a1, b1, a2, b2, bits)
    individuos = evaluar_poblacion(individuos, f)

    # === Históricos para gráficas comparables ===
    mejores = []   # mejor fitness por generación
    medias  = []   # media de fitness por generación

    # Parámetros por defecto de torneo
    if torneo_params is None:
        torneo_params = dict(k=3, with_replacement=True, ttype="deterministic", p=0.75, tie_breaker="random")

    # === Bucle evolutivo ===
    for gen in range(G):
        # --- Estadísticas de la generación ---
        mejor_gen = min(individuos, key=lambda d: d["fitness"])     # minimización
        media_gen = sum(ind["fitness"] for ind in individuos) / len(individuos)
        mejores.append(mejor_gen["fitness"])
        medias.append(media_gen)

        # --- Elitismo: seleccionar élites de la generación actual ---
        elites = seleccionar_elites(individuos, modo=elitismo_modo, percent=elitismo_percent)

        # --- Selección de padres (k = tamaño población) ---
        if operador_seleccion == "ruleta":
            probas = calculo_probas(individuos)  # maneja minimización y fitness 0/negativo
            seleccionados, _ = seleccion_ruleta(individuos, probas, k=len(individuos))
        elif operador_seleccion == "sus":
            probas = calculo_probas(individuos)
            seleccionados, _ = seleccion_sus(individuos, probas, k=len(individuos))
        elif operador_seleccion == "torneo":
            seleccionados, _ = seleccion_torneo(individuos, k_seleccion=len(individuos), **torneo_params)
        elif operador_seleccion == "aleatoria":
            seleccionados, _ = seleccion_aleatoria(individuos, k=len(individuos))
        else:
            raise ValueError("Operador de selección inválido")

        # --- Emparejar, cruzar, mutar ---
        parejas = formar_parejas(seleccionados)
        hijos = []
        for p in parejas:
            c1, c2 = p[0]["cromosoma"], p[1]["cromosoma"]
            if random.random() < p_cruce:
                h1, h2 = cruce_1punto(c1, c2)
            else:
                h1, h2 = c1[:], c2[:]
            h1 = mutacion(h1, p_mut)
            h2 = mutacion(h2, p_mut)
            hijos.append(construir_individuo_desde_cromosoma(h1, a1, b1, a2, b2, n, m))
            hijos.append(construir_individuo_desde_cromosoma(h2, a1, b1, a2, b2, n, m))

        hijos = evaluar_poblacion(hijos, f)

        # --- mantener población constante ---
        if len(hijos) > len(individuos):
            random.shuffle(hijos)
            next_poblacion = hijos[:len(individuos)]
        elif len(hijos) < len(individuos):
            faltan = len(individuos) - len(hijos)
            next_poblacion = hijos + random.sample(hijos, k=faltan)
        else:
            next_poblacion = hijos

        # --- Inserción de élites (saltan cruza y mutación) ---
        individuos = insertar_elites(next_poblacion, elites, mecanismo=elitismo_mecanismo)

    # === Resultado final ===
    mejor = min(individuos, key=lambda d: d["fitness"])
    return mejor['fitness'], mejor['x'], mejor['y'], mejores, medias




nombres_funciones = ["ESFERA 2-D", 
                     "BUKIN", 
                     "HIMMELBLAU", 
                     "EGGHOLDER", 
                     "EASOM"
                     ]

for nombre in nombres_funciones:
    cfg = EJECUCIONES[nombre] #EJECUCIONES está definida en Tarea1.py
    traces = {}

    # Aleatoria
    b1, x1, y1, hb1, hm1 = AG_Simple_2(**cfg,
                                        operador_seleccion="aleatoria",
                                        elitismo_modo="best_one"
                                        )
    traces["ALEATORIA | elit:best_one"] = (hb1, hm1, cfg["f"])

    # Ruleta
    b2, x2, y2, hb2, hm2 = AG_Simple_2(**cfg,
                                        operador_seleccion="ruleta", 
                                        elitismo_modo="top_percent", 
                                        elitismo_percent=0.05
                                        )
    traces["RULETA | elit:top(5%)"] = (hb2, hm2, cfg["f"])

    # SUS
    b3, x3, y3, hb3, hm3 = AG_Simple_2(**cfg,
                                        operador_seleccion="sus",
                                        elitismo_modo="top_percent", 
                                        elitismo_percent=0.05
                                        )
    traces["SUS | elit:top(5%)"] = (hb3, hm3, cfg["f"])

    # Torneo determinista k=3
    b4, x4, y4, hb4, hm4 = AG_Simple_2(**cfg, 
                                    operador_seleccion="torneo",
                                    torneo_params=dict(k=3, with_replacement=True, ttype="deterministic", p=0.75, tie_breaker="random"),
                                    elitismo_modo="top_percent", 
                                    elitismo_percent=0.05
                                    )
    traces["TORNEO k=3 det | elit:top(5%)"] = (hb4, hm4, cfg["f"])

    graficar_comparativa(traces,
                        titulo_extra=cfg["f"].__name__.upper()
                        )
