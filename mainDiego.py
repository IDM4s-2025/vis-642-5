import pandas as pd 
import plotly.express as px

def graph(): 
    data = pd.read_csv("iris.csv")
    fig = px.scatter(data, x="SepalLengthCm", y="PetalLengthCm", color="Species",
                    size='PetalWidthCm', hover_data=['SepalWidthCm'])
    fig.show()