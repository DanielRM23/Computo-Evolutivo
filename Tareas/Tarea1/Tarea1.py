import numpy as np 
import matplotlib.pyplot as plt
import random




def longitud(variables_rango):
    """
    variables_rango: [[a1,b1], [a2,b2], ...]
    Regresa las longitudes |b - a| de cada variable.
    """
    return [abs(r[1] - r[0]) for r in variables_rango]
        

def valores_a_representar(precision, longitudes):
    diez_elevado_a = 10**precision
    return [longitud*diez_elevado_a for longitud in longitudes]


def bits_necesarios(valores):
    return [np.floor(np.log2(i)+1) for i in valores]


def suma_bits(bits_necesarios):
    return np.sum(bits_necesarios)


def generar_individuo(N):
    N = int(N)
    return [random.randint(0,1) for _ in range(0,N)]


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


def decodificacion(a,b, individuo_binario, m):
    x = a + binario_a_decimal(individuo_binario)*( (b-a)/((2**m) - 1) )
    return x


def individuos_decodificados(M,a1,b1,a2,b2,bits_necesaios):
    n = int(bits_necesaios[0])
    m = int(bits_necesaios[-1])

    N = suma_bits(bits_necesaios)

    parejas_individuos = []

    for _ in range(M):
        individuo = generar_individuo(N)
        
        bits_izquierdo = individuo[:n] 
        bits_derecho = individuo[n:n+m]

        x = decodificacion(a1,b1,bits_izquierdo,n)
        y = decodificacion(a2,b2,bits_derecho,m)

        item = {
            "cromosoma": individuo, 
            "x":x,
            "y":y,
            "fitness": None
        }

        parejas_individuos.append( item )

    return parejas_individuos


# ======================================= FUNCIONES A PROBAR =======================================


def esfera(x):
    return np.sum(np.square(x))   # x es un vector (np.array), válido para n=2 o n=5


def bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) \
           - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))


# ======================================= Evaluación de individuos =======================================

def f(x,y):
    return x + y


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
    f = np.array([ind["fitness"] for ind in individuos_evaluados], dtype=float)
    g = f - f.min()                  # g ≥ 0
    f_inverted = 1.0 / (1.0 + g)            # aptitudes positivas
    suma_total  = f_inverted.sum()
    probas = f_inverted / suma_total if suma_total > 0 else np.ones_like(f_inverted)/len(f_inverted)
    return probas


def seleccion_ruleta(poblacion, probas, k):
    # poblacion: lista de individuos (p. ej., dicts con "cromosoma", "x", "y", "fitness")
    # probas: arreglo 1D de probabilidades (mismo tamaño que 'poblacion'), suma ≈ 1
    # k: número de individuos a seleccionar (tamaño del mating pool)

    # Elegimos 'k' índices de 0..len(poblacion)-1 según la distribución 'probas'.
    # replace=False ⇒ sin reemplazo: un mismo individuo no puede ser seleccionado dos veces.
    # OJO: con replace=False debe cumplirse k <= len(poblacion).
    idxs = np.random.choice(len(poblacion), size=k, p=probas, replace=False)

    # Construimos la lista de individuos seleccionados usando los índices obtenidos.
    seleccion = [poblacion[i] for i in idxs]

    # Devolvemos la selección (mating pool) y los índices elegidos (útiles para depurar o registrar).
    return seleccion, idxs



# ======================================= Ejecución del Programa =======================================

variables_rango = [[-3.0, 12.1],
                   [ 4.1,  5.8]]

a1, b1 = variables_rango[0]
a2, b2 = variables_rango[1]

longitudes = longitud(variables_rango)
precision = 4

valores = valores_a_representar(precision, longitudes)     # pasos por variable (aprox)
bits = bits_necesarios(valores)                            # bits por variable

M = 10
individuos = individuos_decodificados(M, a1, b1, a2, b2, bits)


individuos_evaluados = evaluar_poblacion(individuos, f)

k = len(individuos_evaluados)

probas = calculo_probas(individuos_evaluados)


aaaaa = seleccion_ruleta(individuos_evaluados, probas, k=k)
print(aaaaa)
