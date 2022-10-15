# ------------------------------------------------------------------------------
# DESCRIPCION
# ------------------------------------------------------------------------------
# Genera modelos predictivos para un unico conjunto de hiperparametros con distintas semillas
# Se aplica el modelo para predicir distintos meses de la historia con distintos cantidades de envio de incentivos

# ------------------------------------------------------------------------------
# RECURSOS NECESARIOS
# ------------------------------------------------------------------------------
# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM
# 256 GB espacio en disco

# son varios archivos, subirlos INTELIGENTEMENTE a Kaggle

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")

kdataset       <- "./datasets/competencia1_historia_2022_ajustadoIPC.csv.gz"

ksemillas  <- c(892237) #reemplazar por las propias semillas

ktraining      <- c( 202101 )   #periodos en donde entreno

#periodos donde aplico el modelo final
kmeses_prediccion <- c(201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 
                       202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012)

kexperimento   <- paste0("EXP_HISTORIA_", format(Sys.time(), "%Y%m%d%H%M%S"))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa
setwd( "~/buckets/b1" )

#--------------------------------------
#cargo el dataset
dataset  <- fread(kdataset, stringsAsFactors= TRUE)

#creo las carpetas donde van los resultados
#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0("./exp/", kexperimento, "/" ), showWarnings = FALSE )
setwd( paste0("./exp/", kexperimento, "/" ) )   #Establezco el Working Directory DEL EXPERIMENTO

#paso la clase a binaria que tome valores {0,1}  enteros
#set trabaja con la clase  POS = { BAJA+1, BAJA+2 } 
#esta estrategia es MUY importante
dataset[ , clase01 := ifelse( clase_ternaria %in%  c("BAJA+2","BAJA+1"), 1L, 0L) ]

#--------------------------------------

#los campos que se van a utilizar 
# son todos los campos menos la "clase_ternaria" que se descarta y se reemplaza por "clase01" que se convierte en una clase binaria POSITIVO o NEGATIVO
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )

#--------------------------------------

#establezco donde entreno (¿no sería más rapido hacerlo al reves?)
dataset[ , train  := 0L ] # Marcamos todos los registros (100%) como que no son de entrenamiento a traves del campo train = 0
dataset[ foto_mes %in% ktraining, train  := 1L ] # Marcamos los registros que van a ser de entrenamiento a traves del campo train = 1


#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ train==1L, campos_buenos, with=FALSE]),
                        label= dataset[ train==1L, clase01] )

#escribo los parametros del experimento
cat( file="parametros.txt",
     sep= "\t",
     "kparametros",
     "kmax_bin",
     "klearning_rate", 
     "knum_iterations",
     "knum_leaves",
     "kmin_data_in_leaf",
     "kfeature_fraction",
     "\n")

kparametros        <- 0.25
kmax_bin           <- 31
klearning_rate     <- 0.0163000010894652
kfeature_fraction  <- 0.512275326810889
knum_iterations    <- 534
kmin_data_in_leaf  <- 208
knum_leaves        <- 34

cat(  file="parametros.txt",
      append= TRUE,
      sep= "\t",
      kparametros,
      kmax_bin, 
      klearning_rate,
      knum_iterations, 
      knum_leaves,
      kmin_data_in_leaf,
      kfeature_fraction, 
      "\n"  )     

#--------------------------------------
cat( file="analisis_predicciones.txt",
     append= TRUE,
     sep= "\t",
     "kparametros",
     "semilla",
     "mes_prediccion",
     "envios", 
     "envios_aciertos", 
     "envios_no_aciertos", 
     "no_envios_aciertos", 
     "no_envios_no_aciertos", 
     "ganancia",
     "\n")

#genero el modelo
#estos hiperparametros  salieron de una laaarga Optmizacion Bayesiana
for( semilla  in  ksemillas  )
{
  modelo  <- lgb.train( data= dtrain,
                        param= list( objective=          "binary",
                                     max_bin=            kmax_bin,
                                     learning_rate=      klearning_rate,
                                     num_iterations=     knum_iterations,
                                     num_leaves=         knum_leaves,
                                     min_data_in_leaf=   kmin_data_in_leaf,
                                     feature_fraction=   kfeature_fraction,
                                     seed=               semilla
                        )
  )
  
  #--------------------------------------
  #ahora imprimo la importancia de variables
  tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
  archivo_importancia  <- paste0("importantes_", kparametros, "_", semilla , ".txt")
  
  fwrite( tb_importancia, 
          file= archivo_importancia, 
          sep= "\t" )
  
  #--------------------------------------
  #aplico el modelo a los distintos meses a predecir    
  for( mes  in  kmeses_prediccion  )
  {
    dapply  <- dataset[ foto_mes== mes ]
    
    #aplico el modelo a los datos nuevos
    prediccion  <- predict( modelo, 
                            data.matrix( dapply[, campos_buenos, with=FALSE ]))
    
    #genero la tabla de entrega
    tb_entrega  <-  dapply[ , list( numero_de_cliente, foto_mes, clase01 ) ]
    tb_entrega[  , prob := prediccion ]
    
    #ordeno por probabilidad descendente
    setorder( tb_entrega, -prob )
    
    #grabo las probabilidad del modelo
    #fwrite( tb_entrega,
    #        file= paste0("prediccion_", kparametros, "_", semilla, "_", mes , ".txt"),
    #        sep= "\t" )
    
    #genero archivos con los  "envios" mejores
    #deben subirse "inteligentemente" a Kaggle para no malgastar submits
    cortes <- seq( 8000, 12000, by=500 )
    for( envios  in  cortes )
    {        
      clientes_envios <- tb_entrega[1:envios]
      clientes_no_envios <- tb_entrega[(envios+1):nrow(tb_entrega)]
      envios_aciertos <- clientes_envios[clase01==1, .N]
      envios_no_aciertos <- clientes_envios[clase01==0, .N]
      no_envios_aciertos <- clientes_no_envios[clase01==0, .N]
      no_envios_no_aciertos <- clientes_no_envios[clase01==1, .N]
      ganancia <- (envios_aciertos * 78000) - (envios_no_aciertos * 2000)
      
      cat( file="analisis_predicciones.txt",
           append= TRUE,
           sep= "\t",
           kparametros,
           semilla,
           mes,
           envios, 
           envios_aciertos, 
           envios_no_aciertos, 
           no_envios_aciertos, 
           no_envios_no_aciertos, 
           ganancia,
           "\n")  
    }
  }
}

#--------------------------------------

quit( save= "no" )


