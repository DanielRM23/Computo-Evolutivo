import numpy as np 
import matplotlib.pyplot as plt
import random
import math




def longitud(variables_rango):
    """
    variables_rango: [[a1,b1], [a2,b2], ...]
    Regresa las longitudes |b - a| de cada variable.
    """
    return [abs(r[1] - r[0]) for r in variables_rango]
        

def valores_a_representar(precision, longitudes):
    """
    Calcula cuántos valores discretos puede tomar cada variable
    según su rango y la precisión en decimales.

    precision : número de decimales deseados (ej. 3 → milésimas).
    longitudes: lista con longitudes de cada rango (b-a).

    Retorna una lista con el número de pasos aproximados por variable.
    """
    diez_elevado_a = 10**precision      
    return [longitud * diez_elevado_a for longitud in longitudes]


def bits_necesarios(valores):
    """
    valores = [L * 10^precision] por variable.
    Con decodificación a pasos de (b-a)/((2**m)-1), necesitamos:
    2**m >= (pasos + 1)  (incluye ambos extremos)
    """
    bits = []
    for v in valores:
        pasos = int(math.ceil(v))
        puntos = pasos + 1
        m = int(math.ceil(math.log2(puntos)))
        bits.append(m)
    return bits


def suma_bits(bits_necesarios):
    """
    Suma el número de bits requeridos por todas las variables.
    Sirve para saber la longitud total del cromosoma.
    """
    return np.sum(bits_necesarios)


def generar_individuo(N):
    """
    Genera un cromosoma binario aleatorio de longitud N.
    Cada gen es 0 o 1 con probabilidad uniforme.
    """
    N = int(N)
    return [random.randint(0,1) for _ in range(N)]


def binario_a_decimal(bits):
    """
    Convierte un arreglo de 0s y 1s (representando un número binario)
    a su valor decimal.
    
    Ejemplo:
    [1, 0, 1, 1] -> 11
    """
    decimal = 0
    n = len(bits)
    
    for i in range(n):
        # multiplicamos el bit por la potencia de 2 correspondiente
        decimal += bits[n - 1 - i] * (2 ** i)
    
    return decimal


def decodificacion(a, b, individuo_binario, m):
    """
    Mapea un cromosoma binario a un real en [a,b] con resolución de m bits.
    Maneja casos borde cuando m<=0 o (2**m - 1)==0.
    """
    if m <= 0:
        return a  # sin bits → extremo inferior
    denominador = (2**m) - 1
    if denominador == 0:
        return a
    paso = (b - a) / denominador
    return a + binario_a_decimal(individuo_binario) * paso


def individuos_decodificados(M, a1, b1, a2, b2, bits_necesaios):
    """
    Genera M individuos:
    - cromosoma binario de longitud N = n+m
    - decodifica primeras n bits a x \in [a1,b1] y siguientes m bits a y \in [a2,b2]
    - inicializa fitness=None (esta después se actualiza)
    """
    n = int(bits_necesaios[0])    # bits para x
    m = int(bits_necesaios[-1])   # bits para y
    N = int(suma_bits(bits_necesaios))

    poblacion = []
    for _ in range(M):
        indiv = generar_individuo(N)
        bits_x = indiv[:n]
        bits_y = indiv[n:n+m]
        x = decodificacion(a1, b1, bits_x, n)
        y = decodificacion(a2, b2, bits_y, m)
        poblacion.append({
            "cromosoma": indiv,
            "x": x,
            "y": y,
            "fitness": None
        })
    return poblacion



# ======================================= Evaluación de individuos =======================================

def evaluar_poblacion(individuos, f):
    """
    individuos: es un una lista donde cada elemento es de la forma:
        item = {
            "cromosoma": individuo, 
            "x":x,
            "y":y,
            "fitness": None (porque no se ha hecho una evaluación hasta el momento)
        }
    f: funcion a evaluar, función de dos variables
    """
    M = len(individuos)
    for i in range(M):
        x = individuos[i]["x"]
        y = individuos[i]["y"]
        individuos[i]["fitness"] = f(x,y)
    return individuos


# ======================================= Selección de individuos =======================================


def calculo_probas(individuos_evaluados):
    """
    Calcula las probabilidades de selección (ruleta) a partir de los fitness.
    Se tiene el problema de minimización → menor fitness = mayor probabilidad.
    """
    # Extrae fitness de la población
    f = np.array([ind["fitness"] for ind in individuos_evaluados], dtype=float)

    # Desplaza valores para que el mínimo quede en 0 (g >= 0)
    g = f - f.min()

    # Inversión: menor fitness → mayor aptitud
    f_inverted = 1.0 / (1.0 + g)

    # Normaliza para que las probabilidades sumen 1
    suma_total = f_inverted.sum()
    if suma_total > 0:
        probas = f_inverted / suma_total
    else:  # caso raro: todos iguales
        probas = np.ones_like(f_inverted) / len(f_inverted)

    return probas


def seleccion_ruleta(poblacion, probas, k):
    # poblacion: lista de individuos (p. ej., dicts con "cromosoma", "x", "y", "fitness")
    # probas: arreglo 1D de probabilidades (mismo tamaño que 'poblacion'), suma ≈ 1
    # k: número de individuos a seleccionar 

    # Elegimos 'k' índices de 0..len(poblacion)-1 según la distribución 'probas'.
    # replace=False ⇒ sin reemplazo: un mismo individuo no puede ser seleccionado dos veces.
    # OJO: con replace=False debe cumplirse k <= len(poblacion).
    idxs = np.random.choice(len(poblacion), size=k, p=probas, replace=True)

    # Construimos la lista de individuos seleccionados usando los índices obtenidos.
    seleccion = [poblacion[i] for i in idxs]

    # Devolvemos la selección y los índices elegidos (útiles para depurar o registrar).
    return seleccion, idxs



# ======================================= Cruza y Muta =======================================


def formar_parejas(individuos_seleccionados):
    # Copia y mezcla para evitar sesgos por orden
    coso = list(individuos_seleccionados)
    random.shuffle(coso)

    k = len(coso)
    parejas = []

    # Pares disjuntos: [0,1], [2,3], ...
    for i in range(0, k - 1, 2):
        parejas.append([coso[i], coso[i + 1]])  # <-- recuerda que es lista

    # Si queda uno sobrante, emparejarlo con uno aleatorio del resto
    if k % 2 == 1:
        sobrante = coso[-1]
        if k > 1:
            pareja_aleatoria = random.choice(coso[:-1])
            parejas.append([sobrante, pareja_aleatoria])  # <-- recuerda que es lista, no tupla
        else:
            # Caso extremo: solo hay 1 individuo → se empareja consigo mismo
            parejas.append([sobrante, sobrante])          # <-- listaaaaaaaa, no tupla

    return parejas


def cruce_1punto(cromosoma_a, cromosoma_b):
    """Cruce de 1 punto sobre dos listas de bits del mismo largo."""
    N = len(cromosoma_a)
    if N < 2:
        return cromosoma_a[:], cromosoma_b[:]
    c = random.randint(1, N-1)  # este es el punto de corte
    h1 = cromosoma_a[:c] + cromosoma_b[c:]
    h2 = cromosoma_b[:c] + cromosoma_a[c:]
    return h1, h2


def mutacion(crom, p_mut):
    """
    Mutación por bit independiente con prob p_mut.
    - Recorre cada bit b de 'crom'
    - Con probabilidad p_mut, lo voltea (0↔1); si no, lo mantiene.
    """
    return [
        (1 - b) if random.random() < p_mut else b
        for b in crom
    ]


def construir_individuo_desde_cromosoma(cromosoma, a1,b1,a2,b2, n, m):
    """Parte el cromosoma en (n|m), decodifica x,y y arma el dict item."""
    bits_izq = cromosoma[:n]
    bits_der = cromosoma[n:n+m]
    x = decodificacion(a1,b1,bits_izq,n)
    y = decodificacion(a2,b2,bits_der,m)
    return {"cromosoma": cromosoma, "x": x, "y": y, "fitness": None}


# ======================================= Ejecución del Programa =======================================
# SE COMENTA ESTO PARA LA TAREA 2


def AG_Simple(variables_rango,
              precision,
              M,
              G,
              p_cruce,
              f
              ):
    
    a1, b1 = variables_rango[0]
    a2, b2 = variables_rango[1]

    longitudes = longitud(variables_rango)
    valores = valores_a_representar(precision, longitudes)
    bits = bits_necesarios(valores)
    p_mut = 1.0 / suma_bits(bits) if suma_bits(bits) > 0 else 0.01
    n = int(bits[0]); m = int(bits[-1])

    individuos = individuos_decodificados(M, a1, b1, a2, b2, bits)
    individuos = evaluar_poblacion(individuos, f)

    mejores = []  # mejor fitness por generación

    for gen in range(G):
        probas = calculo_probas(individuos)
        seleccionados, _ = seleccion_ruleta(individuos, probas, k=len(individuos))  
        parejas = formar_parejas(seleccionados)

        hijos = []
        for p in parejas:
            c1, c2 = p[0]["cromosoma"], p[1]["cromosoma"]
            h1, h2 = cruce_1punto(c1, c2) if random.random() < p_cruce else (c1[:], c2[:])
            h1 = mutacion(h1, p_mut)
            h2 = mutacion(h2, p_mut)
            hijos.append(construir_individuo_desde_cromosoma(h1, a1, b1, a2, b2, n, m))
            if len(hijos) < len(individuos):
                 hijos.append(construir_individuo_desde_cromosoma(h2, a1, b1, a2, b2, n, m))

        hijos = evaluar_poblacion(hijos, f)
        individuos = hijos

        mejor_gen = min(individuos, key=lambda d: d["fitness"])
        mejores.append(mejor_gen["fitness"])

    # Al final: se obtiene el mejor de toda la ejecución
    mejor = min(individuos, key=lambda d: d["fitness"])
    
    # Se devuelven los resultados que el script principal necesita
    return mejor['x'], mejor['y'], mejor['fitness'], mejores


