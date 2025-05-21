import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def normalizeData(list_indicators: list, column_name: str, data: pd.DataFrame, countries: list, data_life: pd.DataFrame):
    """
    Function that normalizes a parameter according to some indicators in a dataframe

    Args:
        list_indicators (list): A list containing all the values to normalize
        column_name (str): Name of the new column of the indicator
        data (pd.DataFrame): Dataframe to modify
        countries (list): A list of the countries
    """
    for country in countries:
        ed_sum = 0
        #print(country)
        for indicator in list_indicators:
            maxi = data_life.loc[data_life["INDICATOR"] == indicator]["OBS_VALUE"].max()
            mini = data_life.loc[data_life["INDICATOR"] == indicator]["OBS_VALUE"].min()
            val = data_life.loc[(data_life["INDICATOR"] == indicator) & (data_life["LOCATION"] == country)]["OBS_VALUE"]
            
            #If a value is not found it is replaced by the mean
            if(val.shape[0] == 0):
                val = data_life.loc[(data_life["INDICATOR"] == indicator)]["OBS_VALUE"].mean()
            else:
                val = val.iloc[0]
            
            #Doing the normalization
            norm = (val - mini) / (maxi - mini)

            ed_sum += norm
        
        ed_sum /= len(list_indicators)
        
        data.loc[data["Code"] == country, column_name] = ed_sum
        #print(data)

    #Deleting the countries that does not appear in both datasets
    data.dropna(inplace=True)
    #print(data)

def Diegopreprocess(ruta1, ruta2) -> pd.DataFrame:
    # 1. Carga y procesamiento de datos del PIB (Banco Mundial)
    data_gpd = pd.read_csv(
        ruta1,
        skiprows=4,
        usecols=["Country Name", "Country Code", "2023"],
        encoding='utf-8'
    ).rename(columns={
        "Country Name": "Entity",
        "Country Code": "Code",
        "2023": "GDP"
    }).dropna()

    # 2. Carga y filtrado de datos de calidad de vida (OCDE)
    data_life = pd.read_csv(ruta2)
    data_life = data_life[data_life["INEQUALITY"] == "TOT"]

    # 3. Creación del DataFrame final
    final_data = data_gpd[["Code", "GDP"]].copy()
    
    # Normalizar indicadores
    countries = data_life["LOCATION"].unique()
    
    # Educación
    normalizeData(
        list_indicators=["ES_EDUA", "ES_STCS", "ES_EDUEX"],
        column_name="EDUCATION",
        data=final_data,
        countries=countries,
        data_life=data_life
    )
    
    # Satisfacción de vida
    normalizeData(
        list_indicators=["SW_LIFS"],
        column_name="LIFE_SATISFACTION",
        data=final_data,
        countries=countries,
        data_life=data_life
    )
    
    # Añadir nombres de países
    final_data["Country"] = final_data["Code"].map(data_gpd.set_index("Code")["Entity"])
    
    return final_data.dropna()


def Diegoshow_graphs(final_data: pd.DataFrame):
    fig = px.imshow(final_data.drop(columns=["Code", "Country"]).corr(), text_auto=True)
    fig.update_layout(
        title_text = "Correlation Matrix",
        title_x=0.5
    )
    fig.show()

    fig = px.scatter(final_data, x="GDP", y="LIFE_SATISFACTION",  hover_data=["GDP", "LIFE_SATISFACTION", "Country"])
    fig.update_layout(title="LIFE SATISFACTION AND GDP", title_x=0.5)
    fig.show()

    fig = px.scatter(final_data, x="GDP", y="EDUCATION", color_discrete_sequence=['red'], hover_data=["GDP", "LIFE_SATISFACTION", "Country"])
    fig.update_layout(title="EDUCATION AND GDP", title_x=0.5)
    fig.show()

    fig = px.scatter(final_data, x="LIFE_SATISFACTION", y="EDUCATION", color_discrete_sequence=['green'], hover_data=["GDP", "LIFE_SATISFACTION", "Country"])
    fig.update_layout(title="LIFE SATISFACTION AND EDUCATION", title_x=0.5)
    fig.show()

def juampaproces(ruta2):
    df = pd.read_csv(ruta2)
    # Tomamos las columnas que nos importan
    df = df.drop(columns = [columns for columns in df.columns if columns not in ["LOCATION", "INDICATOR", "OBS_VALUE", "UNIT_MEASURE", "INEQUALITY"]])
    df = df[df["INEQUALITY"] == "TOT"]
    return df


def getCorr(dataSet):
    """
    Mostrar estadísticas y matriz de correlación para indicadores.
    """

    print(dataSet.head())
    print("\nDescripción estadística:\n")
    print(dataSet.describe())
    print("")

    # Matriz de correlación
    plt.figure(figsize=(14, 10))
    sns.heatmap(dataSet.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('CORRELATION MATRIX', fontweight='bold')
    plt.tight_layout()
    plt.show()



def juampaproces(ruta2):
    df = pd.read_csv(ruta2)
    df = df.drop(columns=[col for col in df.columns if col not in ["LOCATION", "INDICATOR", "OBS_VALUE", "UNIT_MEASURE", "INEQUALITY"]])
    df = df[df["INEQUALITY"] == "TOT"]

    df_corr = df.pivot(index='LOCATION', columns='INDICATOR', values='OBS_VALUE')

    homicide_rate = df[df["INDICATOR"] == "PS_REPH"].sort_values(by='LOCATION').reset_index(drop=True)
    students_skills = df[df["INDICATOR"] == "ES_STCS"].sort_values(by='LOCATION').reset_index(drop=True)

    common = set(homicide_rate["LOCATION"]) & set(students_skills["LOCATION"])
    homicide_rate = homicide_rate[homicide_rate["LOCATION"].isin(common)].reset_index(drop=True)
    students_skills = students_skills[students_skills["LOCATION"].isin(common)].reset_index(drop=True)

    homicide_rate_n = (homicide_rate['OBS_VALUE'] - homicide_rate['OBS_VALUE'].min()) / (homicide_rate['OBS_VALUE'].max() - homicide_rate['OBS_VALUE'].min())
    students_skills_n = (students_skills['OBS_VALUE'] - students_skills['OBS_VALUE'].min()) / (students_skills['OBS_VALUE'].max() - students_skills['OBS_VALUE'].min())

    return {
        'df_corr': df_corr,
        'homicide_rate': homicide_rate,
        'students_skills': students_skills,
        'homicide_rate_n': homicide_rate_n,
        'students_skills_n': students_skills_n
    }


def juampashow(datos):
    df_corr = datos['df_corr']
    homicide_rate = datos['homicide_rate']
    students_skills = datos['students_skills']
    homicide_rate_n = datos['homicide_rate_n']
    students_skills_n = datos['students_skills_n']

    print("Tabla de indicadores pivotada (primeras filas):")
    print(df_corr.head())

    print("\nDescripción estadística:")
    print(df_corr.describe())

    plt.figure(figsize=(14, 10))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('CORRELATION MATRIX', fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("Homicide Rate UNIT_MEASURE:", homicide_rate["UNIT_MEASURE"].unique())
    print("Students Skills UNIT_MEASURE:", students_skills["UNIT_MEASURE"].unique())

    print(f"homicide rate normalized: max -> {homicide_rate_n.max()}, min -> {homicide_rate_n.min()}, avg -> {homicide_rate_n.mean()}")
    print(f"students skills normalized: max -> {students_skills_n.max()}, min -> {students_skills_n.min()}, avg -> {students_skills_n.mean()}")

    df_plot = pd.DataFrame({
        'Country': homicide_rate['LOCATION'],
        'Homicide Rate': homicide_rate_n.values,
        'Students Skills': students_skills_n.values
    })

    df_long = df_plot.melt(id_vars='Country', var_name='Index', value_name='Value')

    fig1 = px.line(
        df_long,
        x='Country',
        y='Value',
        color='Index',
        markers=True,
        title='Normalized Index Comparison'
    )

    fig1.update_layout(
        yaxis_title='Value',
        xaxis_title='',
        template='plotly_white',
        legend_title=None,
        margin=dict(t=60, l=40, r=40, b=40)
    )
    fig1.show()

    df_scatter = pd.DataFrame({
        'Homicide Rate (normalized)': homicide_rate_n.values,
        'Students Skills (normalized)': students_skills_n.values
    })

    fig2 = px.scatter(
        df_scatter,
        x='Homicide Rate (normalized)',
        y='Students Skills (normalized)',
        opacity=0.7,
        title='Scatter for ratio between indicators'
    )

    fig2.update_layout(
        xaxis_title='Homicide Rate (normalized)',
        yaxis_title='Students Skills (normalized)',
        title_font=dict(size=20, family='Arial', color='black'),
        template='plotly_white',
        margin=dict(t=60, l=40, r=40, b=40)
    )
    fig2.show()

def Daniproces(ruta):
    df = pd.read_csv(ruta)

    expected_cols = {"LOCATION", "INDICATOR", "OBS_VALUE", "INEQUALITY", "UNIT_MEASURE"}
    df = df[df["INEQUALITY"] == "TOT"]
    df = df[["LOCATION", "INDICATOR", "OBS_VALUE"]]

    edu = df[df["INDICATOR"] == "ES_EDUEX"].dropna(subset=["OBS_VALUE"])
    wealth = df[df["INDICATOR"] == "IW_HNFW"].dropna(subset=["OBS_VALUE"])

    common = set(edu["LOCATION"]) & set(wealth["LOCATION"])

    edu = edu[edu["LOCATION"].isin(common)].sort_values(by="LOCATION").reset_index(drop=True)
    wealth = wealth[wealth["LOCATION"].isin(common)].sort_values(by="LOCATION").reset_index(drop=True)

    edu_n = (edu['OBS_VALUE'] - edu['OBS_VALUE'].min()) / (edu['OBS_VALUE'].max() - edu['OBS_VALUE'].min())
    wealth_n = (wealth['OBS_VALUE'] - wealth['OBS_VALUE'].min()) / (wealth['OBS_VALUE'].max() - wealth['OBS_VALUE'].min())

    return {
        'edu': edu,
        'wealth': wealth,
        'edu_n': edu_n,
        'wealth_n': wealth_n
    }

def DaniShow(datos):
    edu = datos['edu']
    wealth = datos['wealth']
    edu_n = datos['edu_n']
    wealth_n = datos['wealth_n']

    df_plot = pd.DataFrame({
        'Country': edu['LOCATION'],
        'Years in Education (normalized)': edu_n,
        'Household Net Wealth (normalized)': wealth_n
    })

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=["Country", "Years in Education", "Household Net Wealth"],
                    fill_color='paleturquoise', align='center'),
        cells=dict(values=[df_plot['Country'], edu['OBS_VALUE'], wealth['OBS_VALUE']],
                   fill_color='lavender', align='center'))
    ])
    fig_table.update_layout(title='Valores Originales de los Indicadores')
    fig_table.show()

    df_long = df_plot.melt(id_vars='Country', var_name='Indicator', value_name='Value')

    fig_line = px.line(df_long, x='Country', y='Value', color='Indicator', markers=True,
                       title='Comparación: Years in Education vs Household Net Wealth')
    fig_line.update_layout(xaxis_tickangle=-45)
    fig_line.show()

    fig_scatter = px.scatter(
        df_plot,
        x='Household Net Wealth (normalized)',
        y='Years in Education (normalized)',
        text='Country',
        title='Dispersión: Years in Education vs Household Net Wealth',
        opacity=0.8
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.show()