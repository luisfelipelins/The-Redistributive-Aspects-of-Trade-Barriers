from fredapi import Fred
from config import *
import pandas as pd
import requests
import zipfile
import json
import io

### --- Parameters --- ###
σ                   = 2    # (Carroll & Hur, 2023)
δ                   = 0.95  # (Dix-Carneiro, Pessoa, Reyes-Heroles & Traiberman, 2023)
ϱ                   = 0.91  # (Carroll & Hur, 2023)
σ_ϵ                 = 0.23  # (Carroll & Hur, 2023)
α                   = 0.81  # Capital weight in HK (Carroll & Hur, 2023)
ψ                   = 0.40  # Outer CES curvature (Carroll & Hur, 2023)
χ                   = -0.49  # Inner CES curvature (Carroll & Hur, 2023)
γ                   = 0.55  # (Carroll & Hur, 2023)
M                   = 1
τ           : float = 0.024
ξ           : float = 0.0
rebate_share: float = 1.0

# Getting non-college persistence
url = 'https://www.federalreserve.gov/consumerscommunities/files/SHED_public_use_data_2019_(CSV).zip'

response = requests.get(url)

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    print(z.namelist())

    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f,low_memory=False,encoding='latin-1')

df = df.loc[df['ppage'].between(22,59)]

parents_educ_int_cases = ['Less than High School degree','High school degree or GED','Some college but no degree','Certificate or technical degree','Associate degree']

df                 = df.loc[(df['CH2'].isin(parents_educ_int_cases)) & (df['CH3'].isin(parents_educ_int_cases))]
full_sample        = df['weight_pop'].sum()
own_educ_int_cases = ['Less than high school','High school','Some college']
persistance        = df.loc[df['ppeducat'].isin(own_educ_int_cases)]['weight_pop'].sum()

π_LL = (persistance / full_sample)**(1 / 30)

cps_link = 'https://www2.census.gov/programs-surveys/demo/tables/educational-attainment/2019/cps-detailed-tables/table-2-1.xlsx'
data     = pd.read_excel(cps_link)

data.columns = data.iloc[4,:]
data         = data.iloc[[5]]

bachelor_or_more = data.iloc[:,7:].sum().sum()
total            = data.iloc[:,1].sum()

ls_emp_share = 1 - (bachelor_or_more / total)

π_HH = 1 - (ls_emp_share / (1 - ls_emp_share)) * (1 - π_LL)

del(cps_link,data,bachelor_or_more,total)

### --- Moments --- ###

# Skill Premium (Carroll & Hur, 2023)

skill_premium = 1.85

# Capital to Output Ratio (FRED-UC Davis)
fred_api = open(CONFIG / 'fred_key.txt','r').read()
fred     = Fred(api_key=fred_api)
k_stock  = fred.get_series('RKNANPUSA666NRUG').loc['2019-01-01']
gdp      = fred.get_series('RGDPNAUSA666NRUG').loc['2019-01-01']
k_to_Y   = k_stock / gdp

# High-skill Share (Own Calculation)
H_to_L   = skill_premium * ((1 - ls_emp_share) / ls_emp_share)
capital_income_share = 0.36  # Data benchmark, not a CES weight
HS_share = (1 - capital_income_share) * (H_to_L / (1 + H_to_L))

# Low-skilled Tasks Offshoring Share (TiVA-OECD)
tiva_link = ('https://sdmx.oecd.org/sti-public/rest/data/'
             'OECD.STI.PIE,DSD_TIVA_FDVA@DF_FDVA,1.1/'
             '.W+USA._T.USA.A+B+C+D_E+F..A'
             '?startPeriod=2019&endPeriod=2019'
             '&dimensionAtObservation=AllDimensions'
             '&format=csvfilewithlabels')

r    = requests.get(tiva_link,headers={'User-Agent': 'Mozilla/5.0'},timeout=60)
data = pd.read_csv(io.StringIO(r.text))

us_va = data.loc[data['VALUE_ADDED_SOURCE_AREA'] == 'USA']['OBS_VALUE'].sum()
wr_va = data.loc[data['VALUE_ADDED_SOURCE_AREA'] == 'W']['OBS_VALUE'].sum()

I = 1 - (us_va / wr_va)

del(tiva_link,r,data,us_va,wr_va)

# W to W^* (Conference Board ILC and TiVA-OECD)

ilc_data         = pd.read_excel(DATA_RAW / 'ilccompensationtimeseries_2016.xlsx',sheet_name=2)
ilc_data         = ilc_data.iloc[1:38,8:10]
ilc_data.columns = ['Country','Wage']

tiva_link = (
    'https://sdmx.oecd.org/sti-public/rest/data/'
    'OECD.STI.PIE,DSD_TIVA_FDVA@DF_FDVA,1.1/'
    '.TWN+NZL+PHL+SGP+AUS+AUT+BEL+CAN+CZE+DNK+EST+FIN+FRA+DEU'
    '+GRC+HUN+IRL+ISR+ITA+JPN+KOR+MEX+NLD+NOR+POL+PRT+SVK'
    '+ESP+SWE+CHE+TUR+GBR+USA+ARG+BRA+CHN+IND._T.USA._T..A'
    '?startPeriod=2013&endPeriod=2013'
    '&dimensionAtObservation=AllDimensions'
    '&format=csvfilewithlabels'
)
r                 = requests.get(tiva_link,headers={'User-Agent': 'Mozilla/5.0'},timeout=60)
tiva_data         = pd.read_csv(io.StringIO(r.text))
tiva_data         = tiva_data[['Value added origin area','OBS_VALUE']]
tiva_data.columns = ['Country','Value Added']

renamer = {'Chinese Taipei': 'Taiwan',
           'Türkiye': 'Turkey',
           'Czechia': 'Czech Republic',
           'Korea': 'South Korea',
           'China (People’s Republic of)': 'China',
           'Slovak Republic': 'Slovakia'}

tiva_data['Country'] = tiva_data['Country'].replace(renamer)
data                 = pd.merge(ilc_data,tiva_data,on='Country',how='outer').set_index('Country')
us                   = data.loc['United States']['Wage']
data.drop('United States',inplace=True)
row = (data['Wage'] * (data['Value Added'] / data['Value Added'].sum())).sum()

w_to_wstar = us / row

### --- Saving to pre-GMM parameters --- ###
parameters = {'σ': σ,'δ': δ,'ϱ': ϱ,'σ_ϵ': σ_ϵ,'α': α,'γ': γ,'ψ': ψ,'χ': χ,'M': M,'π_LL': π_LL,'π_HH': π_HH,
              'τ': τ,'ξ': ξ,'rebate_share': rebate_share}
moments = {'skill_premium': skill_premium,'K/Y': k_to_Y,'HS_share': HS_share,'capital_income_share': capital_income_share,'I': I,'w_to_wstar': w_to_wstar}

save_dict = {'production': 'nested_ces','parameters': parameters,'moments': moments}

with open(DATA_PARAMS / 'pre_gmm_params_ces.json','w') as f:
    json.dump(save_dict,f,indent=4)
