# ------------------------------------------------------------------------------
# DESCRIPCION
# ------------------------------------------------------------------------------
# Genera un nuevo dataset a partir del dataset original competencia1_historia_2022.csv.gz 
# realizando un ajuste por inflación de todas las variables monetarias

# ------------------------------------------------------------------------------
# RECURSOS NECESARIOS
# ------------------------------------------------------------------------------
# Necesita para correr en Google Cloud:
#  32 GB de memoria RAM
# 256 GB de espacio en el disco local
#   8 vCPU

rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

# ------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------

kdirectorio_trabajo <- "~/buckets/b1/"
kdataset       <- "./datasets/competencia1_historia_2022.csv.gz" # Dataset original con historia

MESES_AJUSTADOS = c(
  201901,201902,201903,201904,201905,201906,201907,201908,201909,201910,201911,201912,
  202001,202002,202003,202004,202005,202006,202007,202009,202010,202008,202011,202012,
  202101,202102)
cantidad_meses = seq( 1, length(MESES_AJUSTADOS), by=1 )
#IPC_MESES de 201901 a 202103 (27 meses)
IPC_MESES = c(
  189.6, 196.8, 206.0, 213.1, 219.6, 225.5, 230.5, 239.6, 253.7, 262.1, 273.2, 283.4,
  289.8, 295.7, 305.6, 310.1, 314.9, 322.0, 328.2, 337.1, 346.6, 359.7, 371.0, 385.9,
  401.5,415.9,435.9)
INDICE_ULTIMO_MES = length(IPC_MESES)
#variables a ajustar por IPC_MESES
variablesMonetarias = c(
  "mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen", "mcuenta_corriente_adicional", "mcuenta_corriente",
  "mcaja_ahorro", "mcaja_ahorro_adicional", "mcaja_ahorro_dolares", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo", "mtarjeta_master_consumo",
  "mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios", "mplazo_fijo_dolares", "mplazo_fijo_pesos", "minversion1_pesos",
  "minversion1_dolares", "minversion2", "mpayroll", "mpayroll2", "mcuenta_debitos_automaticos", "mttarjeta_visa_debitos_automaticos", "mttarjeta_master_debitos_automaticos",
  "mpagodeservicios", "mpagomiscuentas", "mcajeros_propios_descuentos", "mtarjeta_visa_descuentos", "mtarjeta_master_descuentos", "mcomisiones_mantenimiento", 
  "mcomisiones_otras", "mforex_buy", "mforex_sell", "mtransferencias_recibidas", "mtransferencias_emitidas", "mextraccion_autoservicio", "mcheques_depositados",
  "mcheques_emitidos", "mcheques_depositados_rechazados", "mcheques_emitidos_rechazados", "matm", "matm_other", "Master_mfinanciacion_limite", "Master_msaldototal",
  "Master_msaldopesos", "Master_msaldodolares", "Master_mconsumospesos", "Master_mconsumosdolares", "Master_mlimitecompra", "Master_madelantopesos", "Master_madelantodolares",
  "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares", "Master_mconsumototal", "Master_mpagominimo", "Visa_mfinanciacion_limite", "Visa_msaldototal",
  "Visa_msaldopesos", "Visa_msaldodolares", "Visa_mconsumospesos", "Visa_mconsumosdolares", "Visa_mlimitecompra", "Visa_madelantopesos", "Visa_madelantodolares",
  "Visa_mpagado", "Visa_mpagospesos", "Visa_mpagosdolares", "Visa_mconsumototal", "Visa_mpagominimo")

# ------------------------------------------------------------------------------
# Aqui empieza el programa
# ------------------------------------------------------------------------------

setwd( kdirectorio_trabajo )

#cargo el dataset
dataset  <- fread( kdataset )

#ajusto los valores monetarios al indice de inflación del último mes
for (var in variablesMonetarias) { 
  for (mes in cantidad_meses) {
    dataset[foto_mes == MESES_AJUSTADOS[mes], (var) := lapply(.SD, '*', IPC_MESES[INDICE_ULTIMO_MES] / IPC_MESES[mes]), .SDcols = var]
  } 
}

#grabo el dataset ajustado
setwd( "./datasets" )
fwrite( dataset,
        "historia_2022_ajustadoIPC_MESES.csv.gz",
        logical01= TRUE,
        sep= "," )