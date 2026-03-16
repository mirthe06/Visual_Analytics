import pandas as pd
df = pd.read_csv('city_pollutant_health_merged_v2.csv')
if len(df) > 5000:
    df = df.tail(5000).reset_index(drop=True)

df_health = df.dropna(subset=['NO2', 'TotaalNieuwvormingen_8'])
if len(df_health) > 0:
    print('Corr:', df_health[['NO2', 'TotaalNieuwvormingen_8']].corr().iloc[0,1])
else:
    print("NO DATA")

