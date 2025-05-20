import pandas as pd 
import plotly.express as px


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

def preprocess()->pd.DataFrame:
    #Data of the GPD of the countries
    data_gpd = pd.read_csv("gdp-per-capita-worldbank.csv")
    data_gpd = data_gpd[data_gpd["Year"]>=2023]
    data_gpd = data_gpd.rename(columns={"GDP per capita, PPP (constant 2021 international $)": "GDP"})

    #There are some nan
    print("Null data: ")
    print(data_gpd.isnull().sum())

    #Deleting the nan
    data_gpd = data_gpd.dropna()

    #Data of life quality
    data_life = pd.read_csv("OECD,DF_BLI,+all.csv")
    data_life = data_life[data_life["INEQUALITY"] == "TOT"]
    #print("Data-Life")

    final_data = pd.DataFrame()

    cols = ["Code", "GDP"]
    final_data= data_gpd[cols]

    #For education
    interst_indicators_edu = ["ES_EDUA", "ES_STCS", "ES_EDUEX"]
    countries = data_life["LOCATION"].unique()
    normalizeData(list_indicators=interst_indicators_edu, column_name="EDUCATION", data=final_data, countries=countries, data_life=data_life)

    #For Life Satisfaction
    interst_indicators_life_sat = ["SW_LIFS"]
    normalizeData(list_indicators=interst_indicators_life_sat, column_name="LIFE_SATISFACTION", data=final_data, countries=countries, data_life=data_life)

    for row in final_data.iterrows():
        final_data.loc[final_data["Code"] == row[1]["Code"], "Country"] = data_gpd[data_gpd["Code"] == row[1]["Code"]]["Entity"].values[0]
    
    return final_data


def show_graphs(final_data: pd.DataFrame):
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