



# Proyecto de LDA 

### Josué Jacobs 


En ese proyecto se realizo un scrapping de reviews the trip advisor.



El proyecto cuenta con un api.

Un endpoint para entrenar el modelo.
- /train

Un endpoint para predecir si un review es bueno o malo 
- /predict?reivews="review to predict"

Para correr el proyecto se hace con el siguiente comando.
``` bash
uvicorn api.main:app --reload
```