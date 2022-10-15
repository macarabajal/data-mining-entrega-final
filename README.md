# data-mining-entrega-final

- Reproducción de experimento elegido en kaggle

1. Ejecutar "01_ajustar inflacion.r"
2. Ejecutar "02_feature_engineering.r"
3. Ejecutar "04_lgbm_kaggle_semillerio.r" 3 veces, una por cada conjunto de los hiperparametros detallados en el script:
	- hiperparametros de modelo 1
	- hiperparametros de modelo 2
	- hiperparametros de modelo 3

- Los 3 conjuntos de hiperparámetros elegidos salieron de ejecutar "03_BO_lgbm_undersampling_cv5semillas.r" 3 veces, cada vez con una ksemilla_dataset diferente para que en cada ejecución haga un undersampling de datos distintos.

- El script "lgbm_prediciendo_historia.r" fue utilizado para predecir los meses del dataset historico entrenando con el mes 202101 y comparar las ganancias
