import plotly.express as px
import pandas as pd

def scatter_plot(df, x_col, y_col, color_col, size_col, hover_cols):
    """
    Crea un gráfico de dispersión interactivo con Plotly Express.

    Parámetros:
    - df: DataFrame de pandas con los datos
    - x_col: nombre de la columna para el eje x
    - y_col: nombre de la columna para el eje y
    - color_col: nombre de la columna para diferenciar por color
    - size_col: nombre de la columna para definir el tamaño de los puntos
    - hover_cols: lista de columnas que aparecerán al pasar el cursor

    Retorna:
    - Un objeto Figure que puede mostrarse con .show()
    """
    fig = px.scatter(df, 
                     x=x_col, 
                     y=y_col, 
                     color=color_col,
                     size=size_col,
                     hover_data=hover_cols)
    fig.show()
    return fig

#Uso
#df = px.data.iris()
#scatter_plot(df, 
#             x_col="sepal_width", 
#             y_col="sepal_length", 
#             color_col="species", 
#             size_col="petal_length", 
#             hover_cols=["petal_width"])
