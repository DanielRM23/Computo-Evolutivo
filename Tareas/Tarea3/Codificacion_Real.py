import numpy as np 
import random
import matplotlib.pyplot as plt




# ======================================= FUNCIONES A PROBAR =======================================


def esfera(*x):
    return np.sum(np.array(x)**2)   # x es un vector (np.array), válido para n=2 o n=5


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

INSTANCIAS = {
    "ESFERA 5-D": dict(f=esfera, variables_rango=[[-10,10],[-10,10],[-10,10],[-10,10],[-10,10]], precision=3, pob_size=30, num_gen=100, p_cruce=0.9),
    "BUKIN":      dict(f=bukin,  variables_rango=[[-15,-5],[-3,3]], precision=3, pob_size=20, num_gen=100, p_cruce=0.3),
    "HIMMELBLAU": dict(f=himmelblau, variables_rango=[[-5,5],[-5,5]], precision=3, pob_size=40, num_gen=200, p_cruce=0.9),
    "EGGHOLDER":  dict(f=eggholder,  variables_rango=[[-512,512],[-512,512]], precision=3, pob_size=250, num_gen= 1000, p_cruce=0.9),
    "EASOM":      dict(f=easom,      variables_rango=[[-100,100],[-100,100]], precision=3, pob_size=250, num_gen=500, p_cruce=0.9),
}

# ----------- Generar población -------------------
# Se genera la población de manera aleatoria sobre un rango de valores 


def generar_poblacion(instancia):
    pob_size = instancia["pob_size"]
    decimales = instancia["precision"]

    rangos = np.array(instancia["variables_rango"], dtype=float)  # shape (n, 2)
    lows  = rangos[:, 0]  # (n,) #Toma toda la primera columna (valores mínimos)
    highs = rangos[:, 1]  # (n,) Toma toda la segunda columna (valores máximos)
    
    X = np.random.uniform(lows, highs, size=(pob_size, len(rangos)))  # (pob_size, n)
    return np.round(X, decimals=decimales)


# ----------- Evaluar población -------------------
# Se evalúa la población con la función objetivo

def evaluar_fitness_2D(instancia, poblacion): # esta es para funciones en 2D
    f = instancia["f"]
    fitness = [f(ind[0], ind[1]) for ind in poblacion]
    return np.array(fitness)


def evaluar_fitness_5D(instancia, poblacion): #esta es para la esfera en 5D
    f = instancia["f"]
    fitness = [f(*ind) for ind in poblacion]  # desempaqueta los 5 valores
    return np.array(fitness)


# ------------ Torneo -------------------
def torneo(poblacion, poblacion_evaluada, k):
    # poblacion_evaluada: vector con los fitness de cada individuo
    indices = np.random.choice(len(poblacion_evaluada), k, replace=False)
    # índice con el fitness mínimo
    minimo = np.argmin(poblacion_evaluada[indices])
    ganador = indices[minimo]
    return poblacion[ganador]


# ------------ Cruza ------------

def cruza(instancia, padre, madre):
    rangos = np.array(instancia["variables_rango"])
    lows  = rangos[:,0]
    highs = rangos[:,1]

    hijo1 = []
    hijo2 = []
    for (gen_padre, gen_madre) in zip(padre, madre):
        alpha = np.random.uniform(-0.25, 1.25)  # α distinto por gen
        beta = np.random.uniform(-0.25, 1.25) 

        gen_hijo_1 = gen_padre + alpha*(gen_madre-gen_padre)
        gen_hijo_2 = gen_madre + beta*(gen_padre-gen_madre)
        
        hijo1.append(gen_hijo_1)
        hijo2.append(gen_hijo_2)

    hijo1 = np.array(hijo1)
    hijo1 = np.clip(hijo1, lows, highs)  # recortar al rango permitido

    hijo2 = np.array(hijo2)
    hijo2 = np.clip(hijo2, lows, highs)  # recortar al rango permitido

    return hijo1, hijo2


# ------------ Mutación ------------

def mutacion(instancia, hijo, m): 
    
    numero_genes = len(hijo)
    p = 1/numero_genes # proba por cada gen 
    
    rangos = np.array(instancia["variables_rango"])
    hijo_mutado = []

    for i, gen in enumerate(hijo):
        if np.random.rand() < p: #aquí muta
            
            low  = rangos[i,0]
            high = rangos[i,1]
            rango = 0.5* (high - low)
            
            delta = 0            
            for j in range(0, m):
                if np.random.rand() < 1/m: 
                    delta += 2**(-j)
                
            s = np.random.choice([-1,1])
            nuevo_gen = gen + s*rango*delta

            nuevo_gen = np.clip(nuevo_gen, low, high)
            hijo_mutado.append(nuevo_gen) 
        else:
            hijo_mutado.append(gen)
    
    return hijo_mutado




def ciclo_algoritmo(instancia, e, m):
    """
    Ejecuta el GA con elitismo de tamaño e y mutación tipo 'non-uniform' por bits (m).
    Devuelve: (best_x, best_f, historial_best)
    """
    decimales = instancia["precision"]
    M = instancia["pob_size"]
    G = instancia["num_gen"]
    p_cruce = instancia["p_cruce"]
    f = instancia["f"]

    # 1) Población inicial
    poblacion = generar_poblacion(instancia)  # (M, n)

    historial_best = []

    for _ in range(G):
        # 2) Evaluación
        n = len(instancia["variables_rango"])
        if n == 2:
            fitness = evaluar_fitness_2D(instancia, poblacion)
        else:
            fitness = evaluar_fitness_5D(instancia, poblacion)

        # 3) Elitismo
        idx_orden = np.argsort(fitness)           # menor = mejor
        idx_elite = idx_orden[:e]
        elite      = poblacion[idx_elite].copy()
        elite_fit  = fitness[idx_elite].copy()

        # registrar mejor
        historial_best.append(elite_fit[0])

        # 4) Reproducción hasta completar M - e hijos
        hijos = []
        while len(hijos) < M - e:
            padre = torneo(poblacion, fitness, k=5)
            madre = torneo(poblacion, fitness, k=5)

            if np.random.rand() < p_cruce:
                h1, h2 = cruza(instancia, padre, madre)
            else:
                h1, h2 = padre.copy(), madre.copy()

            h1 = mutacion(instancia, h1, m)
            h2 = mutacion(instancia, h2, m)

            # redondeo a la precisión deseada
            h1 = np.round(h1, decimals=decimales)
            h2 = np.round(h2, decimals=decimales)

            hijos.append(h1)
            if len(hijos) < M - e:
                hijos.append(h2)

        hijos = np.array(hijos, dtype=float)

        # 5) Nueva generación = élite + hijos
        poblacion = np.vstack([elite, hijos])

    # evaluación final
    n = len(instancia["variables_rango"])
    if n == 2:
        fitness = evaluar_fitness_2D(instancia, poblacion)
    else:
        fitness = evaluar_fitness_5D(instancia, poblacion)

    best_idx = np.argmin(fitness)
    best_x   = poblacion[best_idx]
    best_f   = fitness[best_idx]
    return best_x, best_f, np.array(historial_best)





# eggholder = INSTANCIAS["EGGHOLDER"]
# pob_eggholder = generar_poblacion(eggholder)
# poblacion_evaluada_eggholder = evaluar_fitness_5D(eggholder, pob_eggholder)
# print(poblacion_evaluada_eggholder)


# esfera = INSTANCIAS["ESFERA 5-D"]
# poblacion_esfera = generar_poblacion(esfera)
# poblacion_evaluada_esfera = evaluar_fitness_5D(esfera, poblacion_esfera)
# #print(poblacion_evaluada_esfera)


# padre = torneo(poblacion_esfera, poblacion_evaluada_esfera, k=5)
# madre = torneo(poblacion_esfera, poblacion_evaluada_esfera, k=5)


# hijo = cruza(esfera, padre, madre)

# hijo_mutado = mutacion(esfera, hijo, m=4)
# print(hijo)

# print(hijo_mutado)

inst = INSTANCIAS["ESFERA 5-D"]
best_x, best_f, hist = ciclo_algoritmo(instancia=inst, e=4, m=5)
print("Mejor x:", best_x, "\nMejor f:", best_f)
plt.plot(hist); plt.xlabel("Generación"); plt.ylabel("Mejor fitness"); plt.show()

