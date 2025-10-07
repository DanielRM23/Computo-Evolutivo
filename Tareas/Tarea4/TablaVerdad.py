import copy
import random



class Nodo: 
    def __init__(self, valor, hijos): 
        """Constructor de la clase Nodo.
        Args: 
            valor: El valor del nodo; AND, OR, NOT, A, B, C.
            hijos: Una lista de nodos hijos.
        """
        self.valor = valor 
        self.hijos = hijos if hijos is not None else []

    def agregar_hijo(self, hijo): 
        """Agrega un hijo al nodo.
        Args: 
            hijo: El nodo hijo a agregar.
        """
        self.hijos.append(hijo)
    

class Individuo:
    def __init__(self, nodo_raiz):
        """Constructor de la clase Individuo.
        
        Args:
            nodo_raiz: El nodo raíz del árbol que representa este individuo.
        """
        self.nodo_raiz = nodo_raiz
        self.fitness = 0

    def evaluar(self, valores_entrada):
        """Evalúa el árbol del individuo completo.
        
        Args:
            valores_entrada: Un diccionario con los valores de las variables,
                             ej: {'A': 1, 'B': 0, 'C': 1}.
                             
        Returns:
            El resultado booleano (0 o 1) de la evaluación.
        """
        return self._evaluar_recursivo(self.nodo_raiz, valores_entrada)


    def _evaluar_recursivo(self, nodo_actual, valores_entrada):
        """Función auxiliar para evaluar el árbol de forma recursiva."""
        
        # Caso Base: Si el nodo es un terminal (A, B, o C).
        if nodo_actual.valor in ['A', 'B', 'C']:
            return valores_entrada[nodo_actual.valor]

        # Caso Recursivo: Si el nodo es una función.
        # 1. Evalúa los hijos primero para obtener sus resultados.
        resultados_hijos = []
        for hijo in nodo_actual.hijos:
            #print(f"Evaluando hijo: {hijo.valor}")
            resultado_hijo = self._evaluar_recursivo(hijo, valores_entrada)
            resultados_hijos.append(resultado_hijo)

        # 2. Aplica la operación del nodo actual a los resultados de los hijos.
        if nodo_actual.valor == 'AND':
            return 1 if resultados_hijos[0] == 1 and resultados_hijos[1] == 1 else 0
        
        elif nodo_actual.valor == 'OR':
            return 1 if resultados_hijos[0] == 1 or resultados_hijos[1] == 1 else 0
            
        elif nodo_actual.valor == 'XOR':
            return 1 if resultados_hijos[0] != resultados_hijos[1] else 0
            
        elif nodo_actual.valor == 'NOT':
            return 1 if resultados_hijos[0] == 0 else 0
        

    def __str__(self):
        """
        Devuelve una representación en string del árbol (notación LISP).
        Cuando llames a print(tu_individuo), Python ejecutará este método.
        """
        return self._imprimir_recursivo(self.nodo_raiz)

    def _imprimir_recursivo(self, nodo_actual):
        """Función auxiliar que recorre el árbol para construir el string."""

        # Caso Base: Si el nodo es un terminal, solo devuelve su valor.
        if not nodo_actual.hijos:
            return nodo_actual.valor
        
        # Caso Recursivo: Si es una función, construye la expresión.
        # Primero, obtiene la representación de todos sus hijos.
        hijos_str = " ".join([self._imprimir_recursivo(hijo) for hijo in nodo_actual.hijos])
        
        # Luego, los une en el formato (FUNCION HIJO1 HIJO2 ...)
        return f"({nodo_actual.valor} {hijos_str})"


def crear_arbol_aleatorio(profundidad_maxima, profundidad_actual, conjunto_funciones):
    
    # Si alcanzamos la profundidad máxima, o si decidimos al azar, creamos un terminal.
    if profundidad_actual == profundidad_maxima or random.random() < 0.4:
        valor_aleatorio = random.choice(TERMINALES) # Se elige un valor de los terminales
        return Nodo(valor=valor_aleatorio, hijos=[]) # Aquí se crea el nodo, con el terminal elegido en la línea anterior. 
                # El decir que los hijos de este nodo son [], se da a entender que no tiene hijos, por lo que es una hoja

    # Si no, creamos una función
    else:
        # Elegimos una función al azar del conjunto que nos pasen   
        nombre_funcion = random.choice(list(conjunto_funciones.keys())) #Tomamos una función al azar
        aridad = conjunto_funciones[nombre_funcion]
        
        # Creamos el nodo para esta función, de momento no tiene hijos
        nodo_funcion = Nodo(valor=nombre_funcion, hijos=[])
        
        # Aquí se crean recursivamente sus hijos
        for _ in range(aridad):
            # Aumentamos en 1 la profundidad, hasta llegar a la proundidad máxima
            hijo = crear_arbol_aleatorio(profundidad_maxima, profundidad_actual + 1, conjunto_funciones)
            nodo_funcion.agregar_hijo(hijo) # Agregamos al hijo
            
        return nodo_funcion
    


def calcular_fitness(individuo):
    """Calcula el fitness de un individuo contando los aciertos en la tabla de verdad."""
    aciertos = 0
    for fila in TABLA_VERDAD: # Tabla de verdad es una constante, y es la tabla que nos dan
        # Extraemos los valores de entrada y el resultado esperado
        valores_entrada = {'A': fila['A'], 'B': fila['B'], 'C': fila['C']}
        salida_esperada = fila['S']
        
        try:
            # Evaluamos el individuo con las entradas actuales
            resultado_evaluado = individuo.evaluar(valores_entrada)
            
            # Si el resultado es el esperado, contamos un acierto
            if resultado_evaluado == salida_esperada:
                aciertos += 1
        except (IndexError, TypeError):
            # Si el árbol está mal formado (raro, pero posible), su fitness es 0
            # Casos particulares, pues, por si acaso lo dejo en este bloque except
            aciertos = 0
            break
            
    individuo.fitness = aciertos # El fitness ahora es el número de aciertos


def seleccion_por_torneo(poblacion, tamaño_torneo=3):
    """
    Selecciona un individuo de la población mediante el método de torneo.
    """
    # Se elige un grupo de competidores al azar de la población
    competidores = random.sample(poblacion, tamaño_torneo)
    
    # Se encuentra al competidor con el mejor fitness
    ganador = max(competidores, key=lambda individuo: individuo.fitness)
    
    return ganador

def _obtener_nodos(nodo_raiz):
    """Función auxiliar para obtener una lista de todos los nodos de un árbol."""
    nodos = []
    _obtener_nodos_recursivo(nodo_raiz, nodos)
    return nodos


def _obtener_nodos_recursivo(nodo_actual, lista_nodos):
    """Recorre el árbol y agrega cada nodo a la lista."""
    if nodo_actual is not None:
        lista_nodos.append(nodo_actual)
        for hijo in nodo_actual.hijos:
            _obtener_nodos_recursivo(hijo, lista_nodos)


def cruzamiento(padre1, padre2):
    """
    Crea dos hijos combinando los árboles de dos padres.
    """
    # Se recomienda trabajar con copias para no alterar a los padres originales
    # Entonces, así  lo hago 
    # Recordemos que para crear un individuo necesitamos una raiz
    hijo1 = Individuo(copy.deepcopy(padre1.nodo_raiz))
    hijo2 = Individuo(copy.deepcopy(padre2.nodo_raiz))

    # Obtenemos una lista de todos los nodos de cada hijo
    nodos_hijo1 = _obtener_nodos(hijo1.nodo_raiz)
    nodos_hijo2 = _obtener_nodos(hijo2.nodo_raiz)

    # Elegimos un nodo al azar de cada árbol para el intercambio
    nodo_a_intercambiar1 = random.choice(nodos_hijo1)
    nodo_a_intercambiar2 = random.choice(nodos_hijo2)

    # Intercambiamos los valores y los hijos de los nodos seleccionados
    nodo_a_intercambiar1.valor, nodo_a_intercambiar2.valor = \
        nodo_a_intercambiar2.valor, nodo_a_intercambiar1.valor
    
    nodo_a_intercambiar1.hijos, nodo_a_intercambiar2.hijos = \
        nodo_a_intercambiar2.hijos, nodo_a_intercambiar1.hijos

    return hijo1, hijo2


def mutacion(individuo, profundidad_maxima, conjunto_funciones):
    """
    Muta un individuo reemplazando un nodo aleatorio con un nuevo sub-árbol.
    """
    individuo_mutado = Individuo(copy.deepcopy(individuo.nodo_raiz))
    
    nodos = _obtener_nodos(individuo_mutado.nodo_raiz)
    nodo_a_mutar = random.choice(nodos)
    
    # Creamos un nuevo sub-árbol aleatorio
    profundidad_actual = 0 # Simplificación, idealmente se calcularía la profundidad real
    nuevo_sub_arbol = crear_arbol_aleatorio(profundidad_maxima, profundidad_actual, conjunto_funciones)
    
    # Reemplazamos el nodo seleccionado
    nodo_a_mutar.valor = nuevo_sub_arbol.valor
    nodo_a_mutar.hijos = nuevo_sub_arbol.hijos
    
    return individuo_mutado


# =======================================================
# EL BUCLE PRINCIPAL
# =======================================================
if __name__ == "__main__":
    
    # --- PASO 1: Configuración del Algoritmo ---
    TAMAÑO_POBLACION = 150
    NUM_GENERACIONES = 100
    PROB_CRUZAMIENTO = 0.9
    PROB_MUTACION = 0.1
    TAMAÑO_TORNEO = 3
    PROFUNDIDAD_MAXIMA = 4
    ELITISMO = True # Guardar al mejor individuo de cada generación
    TERMINALES = ['A', 'B', 'C']

    # Esta es la tabla de verdad que se quiere resolver
    TABLA_VERDAD = [
        {'A': 0, 'B': 0, 'C': 0, 'S': 1},
        {'A': 0, 'B': 0, 'C': 1, 'S': 0},
        {'A': 0, 'B': 1, 'C': 0, 'S': 0},
        {'A': 0, 'B': 1, 'C': 1, 'S': 1},
        {'A': 1, 'B': 0, 'C': 0, 'S': 0},
        {'A': 1, 'B': 0, 'C': 1, 'S': 1},
        {'A': 1, 'B': 1, 'C': 0, 'S': 1},
        {'A': 1, 'B': 1, 'C': 1, 'S': 0},
    ]

    # --- PASO 2: Definición de los Experimentos ---
    experimentos = [
        {
            "nombre": "Experimento 1: F1 = {AND, OR, NOT}",
            "funciones": {'AND': 2, 'OR': 2, 'NOT': 1}
        },
        {
            "nombre": "Experimento 2: F2 = {AND, OR, NOT, XOR}",
            "funciones": {'AND': 2, 'OR': 2, 'NOT': 1, 'XOR': 2}
        }
    ]

    # --- PASO 3: Bucle Principal de Experimentos ---
    for exp in experimentos:
        print(f"\n=================================================")
        print(f"INICIANDO {exp['nombre']}")
        print(f"=================================================")

        # --- PASO 4: Inicialización de la Población ---
        poblacion = []
        for _ in range(TAMAÑO_POBLACION):
            nodo_raiz = crear_arbol_aleatorio(PROFUNDIDAD_MAXIMA, 0, exp['funciones'])
            poblacion.append(Individuo(nodo_raiz))

        mejor_solucion_global = None # de momento no tengo una mejor solución 

        # --- PASO 5: Bucle de Generaciones ---
        for gen in range(NUM_GENERACIONES):
            # a. Evaluación
            for individuo in poblacion:
                calcular_fitness(individuo)

            # b. Reporte y Verificación
            mejor_de_la_generacion = max(poblacion, key=lambda ind: ind.fitness)
            if mejor_solucion_global is None or mejor_de_la_generacion.fitness > mejor_solucion_global.fitness:
                # Trabajo con copias, pa no equivocarse
                mejor_solucion_global = copy.deepcopy(mejor_de_la_generacion) 

            print(f"Generación {gen+1}/{NUM_GENERACIONES} | Mejor Fitness: {mejor_de_la_generacion.fitness}/8")

            if mejor_de_la_generacion.fitness == 8:
                print(f"\n Bibaaaaaaa ¡SOLUCIÓN ENCONTRADA EN LA GENERACIÓN {gen+1}!")
                break

            # c. Creación de la Nueva Generación
            nueva_poblacion = []

            # Elitismo: el mejor individuo pasa directamente a la siguiente generación
            if ELITISMO:
                nueva_poblacion.append(copy.deepcopy(mejor_de_la_generacion))
            
            while len(nueva_poblacion) < TAMAÑO_POBLACION:
                # Selección
                padre1 = seleccion_por_torneo(poblacion, TAMAÑO_TORNEO)
                padre2 = seleccion_por_torneo(poblacion, TAMAÑO_TORNEO)
                
                # Cruzamiento
                if random.random() < PROB_CRUZAMIENTO:
                    hijo1, hijo2 = cruzamiento(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1, padre2
                
                # Mutación
                nueva_poblacion.append(mutacion(hijo1, PROFUNDIDAD_MAXIMA, exp['funciones']))
                if len(nueva_poblacion) < TAMAÑO_POBLACION:
                    nueva_poblacion.append(mutacion(hijo2, PROFUNDIDAD_MAXIMA, exp['funciones']))

            # d. Reemplazo
            poblacion = nueva_poblacion
        
        # --- PASO 6: Resultados Finales del Experimento ---
        print("\n--- RESULTADO FINAL DEL EXPERIMENTO ---")
        if mejor_solucion_global.fitness == 8:
            print(f"Solución perfecta encontrada:")
            print(f"Fórmula: {mejor_solucion_global}")
            print(f"Fitness: {mejor_solucion_global.fitness}/8")
        else:
            print(f"No se encontró una solución perfecta en {NUM_GENERACIONES} generaciones.")
            print(f"La mejor fórmula encontrada fue:")
            print(f"Fórmula: {mejor_solucion_global}")
            print(f"Fitness: {mejor_solucion_global.fitness}/8")