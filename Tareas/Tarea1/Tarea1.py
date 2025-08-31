import numpy as np 
import matplotlib.pyplot as plt
import random
import math


random.seed(0)
np.random.seed(0)



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



# --- Operadores genéticos (añadir) ---

def cruce_1punto(crom_a, crom_b):
    """Cruce de 1 punto sobre dos listas de bits del mismo largo."""
    N = len(crom_a)
    if N < 2:
        return crom_a[:], crom_b[:]
    c = random.randint(1, N-1)  # este es el punto de corte
    h1 = crom_a[:c] + crom_b[c:]
    h2 = crom_b[:c] + crom_a[c:]
    return h1, h2


def mutacion_bitflip(crom, p_mut):
    """Mutación por bit-flip independiente con prob p_mut."""
    return [1-b if random.random() < p_mut else b for b in crom]


def construir_individuo_desde_crom(crom, a1,b1,a2,b2, n, m):
    """Parte el cromosoma en (n|m), decodifica x,y y arma el dict item."""
    bits_izq = crom[:n]
    bits_der = crom[n:n+m]
    x = decodificacion(a1,b1,bits_izq,n)
    y = decodificacion(a2,b2,bits_der,m)
    return {"cromosoma": crom, "x": x, "y": y, "fitness": None}


# ======================================= Ejecución del Programa =======================================


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

    valores = valores_a_representar(precision, longitudes)     # pasos por variable (aprox)
    bits = bits_necesarios(valores)                            # bits por variable

    p_mut   = 1.0 / suma_bits(bits)
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
            h1 = mutacion_bitflip(h1, p_mut)
            h2 = mutacion_bitflip(h2, p_mut)
            hijos.append(construir_individuo_desde_crom(h1, a1, b1, a2, b2, n, m))
            hijos.append(construir_individuo_desde_crom(h2, a1, b1, a2, b2, n, m))

        hijos = evaluar_poblacion(hijos, f)

        # Reemplazo generacional
        if len(hijos) > len(individuos):
            random.shuffle(hijos)
            individuos = hijos[:len(individuos)]
        elif len(hijos) < len(individuos):
            faltan = len(individuos) - len(hijos)
            individuos = hijos + random.sample(hijos, k=faltan)
        else:
            individuos = hijos

        # --- Registrar mejor de la generación ---
        mejor_gen = min(individuos, key=lambda d: d["fitness"])
        mejores.append(mejor_gen["fitness"])

    # Al final: mejor de toda la ejecución
    mejor = min(individuos, key=lambda d: d["fitness"])
    print("Mejor fitness:", mejor["fitness"], "x,y=", mejor["x"], mejor["y"])

    # Esta variable se usa para darle nombre a la image que se va a guardar 
    nombre = "Función de "

    if f == esfera:
        nombre += "Esfera"
        color = "orange"
        # Valores a los que se quiere llegar 
        x_real = 0
        y_real = 0
        minimo = 0

    elif f == bukin:
        nombre += "Bukin"
        color = "red"
        # Valores a los que se quiere llegar 
        x_real = -10
        y_real = 1
        minimo = 0

    elif f == himmelblau:
        nombre += "Himmelblau"
        color = "blue"
        # Valores a los que se quiere llegar 
        x_real = 3.584428
        y_real = -1.848126
        minimo = 0

    elif f == eggholder:
        nombre += "Eggholder"
        color = "green"
        # Valores a los que se quiere llegar 
        x_real = 512
        y_real = 404.2319
        minimo = -959.6407

    elif f == easom:
        nombre += "Easom"
        color = "purple"
        # Valores a los que se quiere llegar 
        x_real = np.pi
        y_real = np.pi
        minimo = -1


    # --- Gráficas ---
    plt.figure(figsize=(15,8))
    plt.plot(mejores, marker="o", linestyle='--', alpha=0.7, color=f"{color}")
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness")
    plt.title(f"Algoritmo Genético Simple con función: {nombre}")
    plt.plot(mejores, marker="o", linestyle="--", alpha=0.7, color=color,
         label=f"Mejor GA → f={mejor['fitness']:.4f}, x={mejor['x']:.4f}, y={mejor['y']:.4f}")
    plt.axhline(minimo, color="black", linestyle=":",
                label=f"Óptimo real → f={minimo:.4f}, x={x_real:.4f}, y={y_real:.4f}")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{nombre}.png")
    plt.show()

    return mejor['fitness'], mejor['x'], mejor['y']


# ======================================= FUNCIONES A PROBAR =======================================


def esfera(x,y):
    return x*x + y*y   # x es un vector (np.array), válido para n=2 o n=5


def bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))


# ============================================ EJECUCIÓN DEL PROGRAMA ============================================

EJECUCIONES = {
    "ESFERA 2-D": dict(f=esfera, variables_rango=[[-10,10],[-10,10]], precision=3, M=30, G=100, p_cruce=0.9),
    "BUKIN":      dict(f=bukin,  variables_rango=[[-15,-5],[-3,3]],    precision=3, M=20, G=100, p_cruce=0.3),
    "HIMMELBLAU": dict(f=himmelblau, variables_rango=[[-5,5],[-5,5]], precision=3, M=40, G=200, p_cruce=0.9),
    "EGGHOLDER":  dict(f=eggholder,  variables_rango=[[-512,512],[-512,512]], precision=3, M=250, G= 1000, p_cruce=0.9),
    "EASOM":      dict(f=easom,      variables_rango=[[-10,10],[-10,10]], precision=3, M=250, G=500, p_cruce=0.9),
}

def run_AG_Simple(name):
    ejecucion = EJECUCIONES[name]
    print(f"\n=== {name} ===")
    return AG_Simple(ejecucion["variables_rango"], ejecucion["precision"], ejecucion["M"], ejecucion["G"], ejecucion["p_cruce"], ejecucion["f"])


# Ejecución de todas las funciones
# comentar las que no se quieren visualizar 
run_AG_Simple("ESFERA 2-D")
run_AG_Simple("BUKIN")
run_AG_Simple("HIMMELBLAU")
run_AG_Simple("EGGHOLDER")
run_AG_Simple("EASOM")