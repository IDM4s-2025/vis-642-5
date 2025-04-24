import pandas as pd 
import plotly.express as px

def graph(): 
    data = pd.read_csv("iris.csv")
    color_discrete_map = {'Iris-setosa': 'rgb(240,72,128)', 'Iris-versicolor': 'rgb(126,60,250)', 'Iris-virginica': 'rgb(250,217,40)'}
    fig = px.scatter(data, x="SepalLengthCm", y="PetalLengthCm", color="Species",
                    size='PetalWidthCm', hover_data=['SepalWidthCm'], color_discrete_map=color_discrete_map)
    fig.show()