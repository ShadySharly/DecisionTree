# ////////////////////////////////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////// # LIBRERIAS # ////////////////////////////////////////// #
# ////////////////////////////////////////////////////////////////////////////////////////////////////// #

library("C50")
library("caret")
library(tidyverse)
library(waffle)

# ////////////////////////////////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////// # ATRIBUTOS # ////////////////////////////////////////// #
# ////////////////////////////////////////////////////////////////////////////////////////////////////// #

# - code: Numero del codigo de la muestra
# - clumpThickness: Grosor del grupo (1 - 10)
# - unifCellSize: Tamano de celula uniforme (1 - 10)
# - unifCellShape: Forma de celula uniforme (1 - 10)
# - marginalAdhesion: Adhesion marginal (1 - 10)
# - epithCellSize: Tamano de celula epitelial (1 - 10)
# - bareNuclei: Nucleos desnudos (1 - 10)
# - blandChromatin: Cromatina suave (1 - 10)
# - normalNucleoli: Nucleolos normales (1 - 10) 
# - mitoses: Mitosis (1 - 10)
# - class: Clase (2 para BENIGNO, 4 para MALIGNO)

# Se crean los nombres que representaran a cada columna, relativos a los parametros que son de relevancia
# en cada observacion.
columns = c("code",
            "clumpThickness",
            "unifCellSize",
            "unifCellShape",
            "marginalAdhesion",
            "epithCellSize",
            "bareNuclei",
            "blandChromatin",
            "normalNucleoli",
            "mitoses",
            "class"
)

# Se procede a almacenar los datos desde el repositorio web "Breast Cancer Wisconsin" (Original), esto en
# un data frame llamado "df"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#url = "breast-cancer-wisconsin.data"
df = read.csv(url, header = F, sep=",", col.names = columns)


# ////////////////////////////////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////// # PRE-PROCESAMIENTO # ////////////////////////////////////// #
# ////////////////////////////////////////////////////////////////////////////////////////////////////// #

# Se sabe que el conjunto de datos cuenta con 16 observaciones que presentan missing values para 
# la variable "bareNuclei", denotados por un caracter "?", sin embargo el lenguaje R normalmente
# asocia este tipo de valores con el simbolo "NA" al igual que todos los paquetes relativos a los
# missing values, por lo que para trabajar de mejor manera se procede a cambiar los "?" por "NA".
df.n = nrow(df)
df.m = ncol(df)

for (row in 1:df.n) {
  for (col in 1:df.m) {
    if (df[row, col] == "?") {
      df[row, col] <- NA
    }
  }
}

# Debido a que la variable bareNuclei contiene valores "?" la variable esta clasificada como de tipo
# "character". por lo que es necesario modificarla para que sea del tipo "integer".
df$bareNuclei <- as.integer(df$bareNuclei)

# Una de las formas de manejar los valores omitidos, consiste en simplemente eliminar cada
# observacion que en sus variables presente uno o mas missing values, metodo conocido como 
# "Listwise Deletion", ahora bien la simplicidad de este metodo viene perjudicado por el hecho de que el
# modelo pierde fuerza, ya que se pierde informacion relevante, ahora bien dependiendo de la razon entre
# el numero de observaciones que presentan missing values y el total de observaciones, puede afectar en
# menor o mayor medida la precision del modelo para predecir una variable de estudio. En este caso, razon
# de observaciones que se perderian al aplicar este metodo corresponde a un 2.3% aproximadamente,
# un numero bastante bajo para considerar este metodo.
df <- na.omit(df)

# Almacenamos el conjunto de datos original, en caso de realizar modificaciones posteriores y no 
# afectar los datos del conjunto inicial, y necesitar estos datos originales.
df.original <- df

df <- subset(df, select = -c(code))

# Ahora dado que el dominio para cada una de las variables es discreto, y que estos van desde el 1 al 10, 
# es posible convertir cada una de estas variables enteras, a variables de tipo "factor", a excepcion
# de la variable "class", la cual tiene un comportamiento distinto tomando solo dos valores 2 y 4.

df[, 1:9] = lapply(df[, 1:9], factor)

df$class = factor(
  df$class, 
  levels = c(2, 4), 
  labels = c("Benigno", "Maligno"))

# ////////////////////////////////////////////////////////////////////////////////////////////////////// #
# /////////////////////////////////////// # ARBOLES DE DECISION # ////////////////////////////////////// #
# ////////////////////////////////////////////////////////////////////////////////////////////////////// #

# Un arbol de decision es corresponde a un tipo particular de Algoritmo de Aprendizaje Supervisado que puede
# ser utilizado ya sea tanto para problemas de regresion, como de clasificacion. Ademas, este funciona
# para tanto entradas categoricas, como continuas al igual que las salidas.

# Para comprender de mejor manera los fundamentos de este algirtmo, es necesario antes definir 
# una serie de conceptos bajo los cuales el algotirmo se desarrolla, y dentro de los cuales se consideran
# los siguientes:

#________________________________________________________________________________________________________#
# I. CONCEPTOS BASICOS

# 1. Nodo Raiz: Representa a muestra o poblacion completa, en este caso al conjunto de datos total. Al 
#    aplicar, el metodo se divide en dos o mas conjuntos homogeneos.

# 2. "Splitting": Corresponde al proceso de dividir un nodo en dos o mas sub-nodos. Dependiendo de si
#    un sub-nodo se puede dividir en dos o mas sub-nodos, o si por el contrario ya no puede dividirse mas,
#    este se clasifica en dos grupos.
#    
#    a. Nodo de Decision: Se denomina asi a un sub-nodo el cual se divide en otros sub-nodos.
#    b. Nodo Terminal: Aquellos sub-nodos que no se dividen en mas sub-nodos, y tambien son denominados
#       como "HOJAS".

# 3. "Prunning": Proceso contrario al splitting, donde se REMUEVEN sub-nodos de un nodo de decision.

# 4. Rama: Corresponde a una sub-seccion de un arbol entero.

# *********************************************************************************************************
# * Se definen como NODOS PADRES a aquellos nodos de los cuales derivan dos o mas sub-nodos, de manera    *
# * analoga se definen NODOS HIJOS, a los sub-nodos que derivan del primero.                              *                                    *
# *********************************************************************************************************

##################################
# a. Arboles de Clasificacion    #
##################################

# Un arbol de CLasificacion es un tipo particular de arbol de decision, al igual que los arboles de 
# regresion, con la diferencia de que este caso particular de arbol es utilizado para predecir RESPUESTAS
# CUALITATIVAS, a diferencia de los arboles de regresion destinados a una respuestas cuantitativa.

# Para estos arboles, la idea es predecir para cada una de las observaciones una relacion de pertenencia
# a la clase, de aquellas observaciones de entrenamiento que ocurren con mas frecuencia en la region a la 
# que pertenece.

# Para esta configuracion en particular, es necesario determinar un metodo a traves del cual se generan
# las distintas bifuraciones del arbol, es decir las configuraciones (PADRE - HIJOS), ante lo cual existen
# distintos metodos.

#________________________________________________________________________________________________________#
# II. CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# 
# Para la construccion del arbol, se generan un conjunto de entrenamiento y un conjuno de prueba

# Antes verificaremos la relacion en la cual esta distribuida la clase a lo largo del conjunto completo.
set.seed(20)
table(df$class)

training.index = createDataPartition(df$class, p = 0.7)$Resample1
df.training = df[training.index, ]
df.test = df[-training.index, ]

# Ahora verificaremos si la praticion se ha realizado correctamente, tanto para el conjunto de prueba, 
# como para el conjunto de entrenamiento
prop.table(table(df.training$class))
prop.table(table(df.test$class))




#________________________________________________________________________________________________________#
# III. AJUSTE DEL MODELO

# Para este caso se utiliza el algoritmo C5.0 para la generacion del arbol, el cual es aplicado para cada
# una de las variables, excepto para la variable explicativa que en este caso corresponde a la variable
# o columna "class" (indice 10 en el data frame)

tree = C5.0(class ~ ., df.training)
tree.rules = C5.0(x = df.training[, -10], y = df.training$class, rules = T)
tree.pred.class = predict(tree, df.test, type = "class")
tree.pred.prob = predict(tree, df.test, type = "prob")


tree.pred.class

#________________________________________________________________________________________________________#
# III. EVALUACION DEL MODELO

conf.matrix.tree = confusionMatrix(table(df.test$class, tree.pred.class))
print(conf.matrix.tree)

head(tree.pred.prob)

#________________________________________________________________________________________________________#
# III. EVALUACION DEL MODELO

plot(tree)
summary(tree)

#________________________________________________________________________________________________________#
# III. BOOSTING

tree_b = C5.0(class ~ ., df.training, trials = 5)
tree_b.rules = C5.0(x = df.training[, -10], y = df.training$class, rules = T, trials = 5)
tree_b.pred.class = predict(tree_b, df.test, type = "class")
tree_b.pred.prob = predict(tree_b, df.test, type = "prob")

conf.matrix.tree.b = confusionMatrix(table(df.test$class, tree_b.pred.class))
print(conf.matrix.tree.b)


