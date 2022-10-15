# ------------------------------------------------------------------------------
# DESCRIPCION
# ------------------------------------------------------------------------------
# Se realiza el proceso de optimización bayesiana con las siguientes caracteristicas:
# 1- Se entrena con POS = { BAJA+1, BAJA+2 }
# 2- Optimizacion Bayesiana de hiperparametros de  lightgbm realizando undersampling con un porcentaje del dataset de entrenamiento
# 3- Ajuste de hiperparametros min_data_in_leaf y num_leaves
# 3- 5-fold cross validation con vector de 5 semillas
# 4- la probabilidad de corte es un hiperparametro

# ------------------------------------------------------------------------------
# RECURSOS NECESARIOS
# ------------------------------------------------------------------------------
# Este script esta pensado para correr en Google Cloud
#   16 vCPU
# 128 GB memoria RAM
# 256 GB espacio en disco

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")
require("lightgbm")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})

# -----------------------------------------------------------------------------------------------------------
# VARIABLES 
# -----------------------------------------------------------------------------------------------------------

kdirectorio_trabajo <- "~/buckets/b1/" #Directorio de trabajo

#Directorio de dataset y archivo datase
kdirectorio_dataset<-"./dataset/dataset_FE_historico_LAG3.csv.gz"

kdirectortio_exp<-"./exp/" #Directorio donde queda el experimiento
kexperimento   <- "undersampling_0.25_LAG3" #Nombre del experimiento

ksemilla_azar  <- c(102191, 251789, 414899, 885107, 892237) #Vector semilla

ksemilla_dataset = ksemilla_azar[1]
# ksemilla_dataset = ksemilla_azar[3] # para generar un segundo modelo con un undersampling con distintos datos
# ksemilla_dataset = ksemilla_azar[5] # para generar un tercer modelo con un undersampling con distintos datos

#periodos en donde entreno
ktraining      <- c( 201903, 202012, 202101 )
# ATENCION  si NO se quiere utilizar  undersampling  se debe  usar  kundersampling <- 1.0
lista_kundersampling  <- c(0.25)   # un undersampling de 0.1 toma solo el 10% de los CONTINUA

kPOS_ganancia  <- 78000
kNEG_ganancia  <- -2000
kBO_iter  <- 100   #cantidad de iteraciones de la Optimizacion Bayesiana

#................... Inicio Definición de rangos para Hiperparametros....................
#learning rate
learning_rate_low       <- 0.01
learning_rate_high      <- 0.3  
#feature_fraction
feature_fraction_low    <- 0.2
feature_fraction_high   <- 0.9
#uso la relación que queire del dataset para el "min_data_in_leaf"
min_data_size_low       <- 0.005  #0.5%
min_data_size_high      <- 0.05   #5%
#Coverage factor, que uso para achicar el num_leaves en base a los min_data_in_leaf seleccionados en funcion del tamaño del dataset.
#Previene el overfiting
coverage_porcentage     <-   0.8
#................... Fin Definición de rangos para Hiperparametros....................

# *************  WARNING!!!! ******************
# Esto es para hacer pruebas rápidas. 
# Full dataset boolreduce== FALSE
boolreduce <- FALSE
end_dataset <-4000

# -----------------------------------------------------------------------------------------------------------
# FUNCIONES 
# -----------------------------------------------------------------------------------------------------------

#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos
loguear  <- function( reg, arch=NA, folder="./exp/", ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )
  
  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=archivo )
  }
  
  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )
  
  cat( linea, file=archivo, append=TRUE )  #grabo al archivo
  
  if( verbose )  cat( linea )   #imprimo por pantalla
}

#------------------------------------------------------------------------------

#esta funcion calcula internamente la ganancia de la prediccion probs
fganancia_logistic_lightgbm   <- function( probs, datos) 
{
  vlabels  <- get_field(datos, "label")
  vpesos   <- get_field(datos, "weight")
  
  gan  <- sum( (probs > PROB_CORTE  ) *
                 ifelse( vpesos == 1.0000002, kPOS_ganancia, 
                         ifelse( vpesos == 1.0000001, kNEG_ganancia, kNEG_ganancia / kundersampling ) ) )
  
  
  return( list( "name"= "ganancia", 
                "value"=  gan,
                "higher_better"= TRUE ) )
}

#------------------------------------------------------------------------------

#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...
EstimarGanancia_lightgbm  <- function( x )
{
  gc()  #libero memoria
  
  #llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  PROB_CORTE <<- x$prob_corte   #asigno la variable global
  
  kfolds  <- 5   # cantidad de folds para cross validation
  
  ganancia_total <- 0
  cantidad_semillas_usadas <- 0
  
  for (semilla in ksemilla_azar){
    
    
    param_basicos  <- list( objective= "binary",
                            metric= "custom",
                            first_metric_only= TRUE,
                            boost_from_average= TRUE,
                            feature_pre_filter= FALSE,
                            verbosity= -100,
                            max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                            min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                            lambda_l1= 0.0,         #por ahora, lo dejo fijo
                            lambda_l2= 0.0,         #por ahora, lo dejo fijo
                            max_bin= 31,            #por ahora, lo dejo fijo
                            num_iterations= 9999,   #un numero muy grande, lo limita early_stopping_rounds
                            force_row_wise= TRUE,   #para que los alumnos no se atemoricen con tantos warning
                            seed= semilla
    )
    
    #el parametro discolo, que depende de otro
    param_variable  <- list(  early_stopping_rounds= as.integer(50 + 5/x$learning_rate) )
    
    param_completo  <- c( param_basicos, param_variable, x )
    
    set.seed( semilla )
    modelocv  <- lgb.cv( data= dtrain,
                         eval= fganancia_logistic_lightgbm,
                         stratified= TRUE, #sobre el cross validation
                         nfold= kfolds,    #folds del cross validation
                         param= param_completo,
                         verbose= -100
    )
    
    #obtengo la ganancia
    ganancia_semilla <-  unlist(modelocv$record_evals$valid$ganancia$eval)[ modelocv$best_iter ]
    ganancia_total  <- ganancia_total + ganancia_semilla
    
    cantidad_semillas_usadas <- cantidad_semillas_usadas + 1
    
    if (ganancia_semilla < 4400000)
    {
      break
    }
  }
  
  ganancia_normalizada  <-  (ganancia_total / cantidad_semillas_usadas) * kfolds   #normailizo la ganancia
  #el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  attr(ganancia_normalizada ,"extras" )  <- list("num_iterations"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra
  
  param_completo$num_iterations <- modelocv$best_iter  #asigno el mejor num_iterations
  param_completo["early_stopping_rounds"]  <- NULL     #elimino de la lista el componente  "early_stopping_rounds"
  
  #logueo 
  xx  <- param_completo
  xx$ganancia  <- ganancia_normalizada   #le agrego la ganancia
  xx$cantidad_semillas_usadas <- cantidad_semillas_usadas
  xx$iteracion <- GLOBAL_iteracion
  loguear( xx, arch= klog )
  
  return( ganancia_normalizada )
}

# -----------------------------------------------------------------------------------------------------------
# Aqui empieza el programa
# -----------------------------------------------------------------------------------------------------------

#<<<<<<<<<<<<<<<<<<<<<<<<<< Lectura dataset >>>>>>>>>>>>>>>>>>>>>>>>
# Aqui se debe poner la carpeta de la computadora local
setwd(kdirectorio_trabajo)
#creo la carpeta donde va el experimento
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0( "./exp/", kexperimento, "/"), showWarnings = FALSE )

#cargo el dataset donde voy a entrenar el modelo
dataset  <- fread(kdirectorio_dataset)

#This is only for test. If boolreduce==1, reduce dataset based on "end_dataset"
if (boolreduce){ 
  print("WARNING!! -- Se selecciónió reducción del dataset --")
  dataset <- dataset[1:end_dataset,]
}

#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes %in% ktraining, clase01 := ifelse( clase_ternaria=="CONTINUA", 0L, 1L) ]
#paso la clase a binaria que tome valores {-1,1} para el mes 202102, solo vamos a quedarnos con los 1 (BAJA+1)
#dataset[ foto_mes == "202102", clase01 := ifelse( clase_ternaria=="BAJA+1", 1L, -1L) ] # comentado porque no generó mejores resultados

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01", "azar", "training" ) )

set.seed( ksemilla_dataset );dataset[  , azar := runif( nrow( dataset ) ) ]

#<<<<<<<<<<<<<<<<<<<<<<<<<< Experimentos con undersampling >>>>>>>>>>>>>>>>>>>>>>>>

# Realizamos la optimización bayesiana con distintos porcentajes de undersampling
for( kundersampling  in  lista_kundersampling  )
{ 
  setwd(kdirectorio_trabajo) #Establezco el Working Directory
  #creo la carpeta donde va el experimento de undersampling
  dir.create( paste0( "./exp/", kexperimento, "/undersampling_", kundersampling, "/"), showWarnings = FALSE )
  setwd( paste0( "./exp/", kexperimento, "/undersampling_", kundersampling, "/") )   #Establezco el Working Directory DEL EXPERIMENTO
  
  #en estos archivos quedan los resultados
  kbayesiana  <- paste0( kexperimento, "_", kundersampling, ".RDATA" )
  klog        <- paste0( kexperimento, "_", kundersampling, ".txt" )
  
  GLOBAL_iteracion  <- 0   #inicializo la variable global
  
  #si ya existe el archivo log, traigo hasta donde llegue
  if( file.exists(klog) )
  {
    tabla_log  <- fread( klog )
    GLOBAL_iteracion  <- nrow( tabla_log )
  }
  
  dataset[  , training := 0L ]
  dataset[ foto_mes %in% ktraining & ( azar <= kundersampling | clase_ternaria %in% c( "BAJA+1", "BAJA+2" ) ), training := 1L ]
  # agrego los "BAJA+1" de febrero al dataset de entrenamiento
  # dataset[ foto_mes == "202102" & clase01 == 1L, training := 1L ] # comentado porque no generó mejores resultados
  
  dataset_size<-nrow(dataset[training == 1]) # Tamaño filas dataset
  
  #dejo los datos de entrenamiento en el formato que necesita LightGBM
  dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ training == 1L, campos_buenos, with=FALSE]),
                          label= dataset[ training == 1L, clase01 ],
                          weight=  dataset[ training == 1L, ifelse( clase_ternaria=="BAJA+2", 1.0000002, ifelse( clase_ternaria=="BAJA+1",  1.0000001, 1.0) )],
                          free_raw_data= FALSE  )
  
  # Aqui se cargan los rangos de hiperparametros para la optimización bayesiana
  # Establezco las probabilidades para cada porcentaje de undersampling
  prob_min  <- 0.5/( 1 + kundersampling*39)
  prob_max  <- pmin( 1.0, 4/( 1 + kundersampling*39) )
  
  #Calculo para poder pasarlo al parametro de mbo
  lower_min_data = as.integer(dataset_size*min_data_size_low)
  upper_min_data = as.integer(dataset_size*min_data_size_high)
  
  #calcúlo con el coverage los limites low & high con respecto a la relación del 
  #tamaño del dataset y el min_data_in_leaf.
  lower_num_leaves = as.integer((dataset_size/upper_min_data)*coverage_porcentage)
  upper_num_leaves = as.integer((dataset_size/lower_min_data )*coverage_porcentage)
  
  #Aqui se cargan los hiperparametros
  hs <- makeParamSet( 
    makeNumericParam("learning_rate",    lower=  learning_rate_low   , upper=    learning_rate_high),
    makeNumericParam("feature_fraction", lower=  feature_fraction_low    , upper=    feature_fraction_high),
    makeIntegerParam("min_data_in_leaf", lower = lower_min_data, upper = upper_min_data),
    makeIntegerParam("num_leaves", lower = lower_num_leaves, upper =  upper_num_leaves),
    #forbideen limita los puntos de prueba según la función de relación. En este caso todo num_leaves >dataset_size/min_data_in_leaf) se elimina.
    forbidden = expression(num_leaves > as.integer((dataset_size/min_data_in_leaf)*coverage_porcentage)),
    makeNumericParam("prob_corte",  lower= prob_min, upper= prob_max  )  
  )
  
  #Aqui comienza la configuracion de la Bayesian Optimization
  funcion_optimizar  <- EstimarGanancia_lightgbm   #la funcion que voy a maximizar
  
  configureMlr( show.learner.output= FALSE)
  
  #configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
  #por favor, no desesperarse por lo complejo
  obj.fun  <- makeSingleObjectiveFunction(
    fn=       funcion_optimizar, #la funcion que voy a maximizar
    minimize= FALSE,   #estoy Maximizando la ganancia
    noisy=    TRUE,
    par.set=  hs,     #definido al comienzo del programa
    has.simple.signature = FALSE   #paso los parametros en una lista
  )
  
  ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
  ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )   #cantidad de iteraciones
  ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )
  
  #establezco la funcion que busca el maximo
  surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))
  
  #inicio la optimizacion bayesiana
  if( !file.exists( kbayesiana ) ) {
    run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
  } else {
    run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
  }
}
# ---------------------------------------------------------------------------------

quit( save="no" )

