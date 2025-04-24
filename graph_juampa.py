import pandas as pd
import plotly.express as px

def iris_errorbars(df):
    df = px.data.iris()
    df["e"] = df["sepal_width"]/100
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    error_x="e", error_y="e")
    fig.show()