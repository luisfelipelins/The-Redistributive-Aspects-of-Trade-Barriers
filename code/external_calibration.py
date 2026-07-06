from fredapi import Fred
from config import *
import pandas as pd
import requests
import zipfile
import json
import io

### --- Parameters --- ###
σ    = 2    # (Carroll & Hur, 2023) 
δ    = 0.95 # (Dix-Carneiro, Pessoa, Reyes-Heroles & Traiberman, 2023) 
ρ    = 0.91 # (Carroll & Hur, 2023) 
σ_ϵ  = 0.23 # (Carroll & Hur, 2023) 
γ    = 0.36 # (Carroll & Hur, 2023) 
M    = 1 

# Getting non-college persistence
url = "https://www.federalreserve.gov/consumerscommunities/files/SHED_public_use_data_2019_(CSV).zip"

response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    print(z.namelist()) 
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, low_memory=False, encoding='latin-1')

df = df.loc[df['ppage'].between(22,59)]

parents_educ_int_cases = ['Less than High School degree','High school degree or GED','Some college but no degree','Certificate or technical degree','Associate degree']

df                 = df.loc[(df['CH2'].isin(parents_educ_int_cases)) & (df['CH3'].isin(parents_educ_int_cases))]
full_sample        = df['weight_pop'].sum()
own_educ_int_cases = ['Less than high school','High school', 'Some college']
persistance        = df.loc[df['ppeducat'].isin(own_educ_int_cases)]['weight_pop'].sum()

π_LL = persistance/full_sample

cps_link = 'https://www2.census.gov/programs-surveys/demo/tables/educational-attainment/2019/cps-detailed-tables/table-2-1.xlsx' 
data = pd.read_excel(cps_link)

data.columns = data.iloc[4,:]
data = data.iloc[[5]]

bachelor_or_more = data.iloc[:,7:].sum().sum()
total            = data.iloc[:,1].sum()

ls_emp_share = 1-(bachelor_or_more / total)

π_HH = 1-(ls_emp_share/(1-ls_emp_share))*(1-π_LL)

del(cps_link,data,bachelor_or_more,total)

### --- Moments --- ###

# Skill Premium (Carroll & Hur, 2023)
skill_premium = 1.85

# Capital to Output Ratio (FRED-UC Davis)
fred_api = open(CONFIG / 'fred_key.txt','r').read()
fred     = Fred(api_key=fred_api)
k_stock  = fred.get_series('RKNANPUSA666NRUG').loc['2019-01-01']
gdp      = fred.get_series('RGDPNAUSA666NRUG').loc['2019-01-01']
k_to_Y   = k_stock/gdp

# High-skill Share (Own Calculation)
H_to_L   = skill_premium * ((1-ls_emp_share)/ls_emp_share)
HS_share = (1-γ)*(H_to_L/(1+H_to_L))

# Low-skilled Tasks Offshoring Share (TiVA-OECD)
tiva_link = ("https://sdmx.oecd.org/sti-public/rest/data/"
             "OECD.STI.PIE,DSD_TIVA_FDVA@DF_FDVA,1.1/"
             ".W+USA._T.USA.A+B+C+D_E+F..A"
             "?startPeriod=2019&endPeriod=2019"
             "&dimensionAtObservation=AllDimensions"
             "&format=csvfilewithlabels")

r    = requests.get(tiva_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
data = pd.read_csv(io.StringIO(r.text))

us_va = data.loc[data['VALUE_ADDED_SOURCE_AREA']=='USA']['OBS_VALUE'].sum()
wr_va = data.loc[data['VALUE_ADDED_SOURCE_AREA']=='W']['OBS_VALUE'].sum()

I = 1-(us_va/wr_va)

del(tiva_link, r, data, us_va,wr_va)

# W to W^* (Conference Board ILC and TiVA-OECD)
ilc_data = pd.read_excel(DATA_RAW / 'ilccompensationtimeseries_2016.xlsx', sheet_name=2)
ilc_data = ilc_data.iloc[1:38,8:10]
ilc_data.columns = ['Country','Wage']

tiva_link = (
    "https://sdmx.oecd.org/sti-public/rest/data/"
    "OECD.STI.PIE,DSD_TIVA_FDVA@DF_FDVA,1.1/"
    ".TWN+NZL+PHL+SGP+AUS+AUT+BEL+CAN+CZE+DNK+EST+FIN+FRA+DEU"
    "+GRC+HUN+IRL+ISR+ITA+JPN+KOR+MEX+NLD+NOR+POL+PRT+SVK"
    "+ESP+SWE+CHE+TUR+GBR+USA+ARG+BRA+CHN+IND._T.USA._T..A"
    "?startPeriod=2013&endPeriod=2013"
    "&dimensionAtObservation=AllDimensions"
    "&format=csvfilewithlabels"
)
r         = requests.get(tiva_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
tiva_data = pd.read_csv(io.StringIO(r.text))
tiva_data = tiva_data[['Value added origin area','OBS_VALUE']]
tiva_data.columns = ['Country','Value Added']

renamer = {'Chinese Taipei':'Taiwan',
           'Türkiye'       :'Turkey',
           'Czechia'       :'Czech Republic',
           'Korea'         :'South Korea',
           'China (People’s Republic of)':'China',
           'Slovak Republic':'Slovakia'}

tiva_data['Country'] = tiva_data['Country'].replace(renamer)
data                 = pd.merge(ilc_data,tiva_data,on='Country',how='outer').set_index('Country')
us                   = data.loc['United States']['Wage']
data.drop('United States', inplace=True)
row                  = (data['Wage'] * (data['Value Added']/data['Value Added'].sum())).sum()

#####################################
COMTRADE_KEY = "49018b796e864c25a92af3e66f5c44b4"

code_to_country = {
    490: 'Taiwan',        554: 'New Zealand',   608: 'Philippines',  702: 'Singapore',
     36: 'Australia',      40: 'Austria',         56: 'Belgium',      124: 'Canada',
    203: 'Czech Republic', 208: 'Denmark',        233: 'Estonia',     246: 'Finland',
    250: 'France',        276: 'Germany',         300: 'Greece',      348: 'Hungary',
    372: 'Ireland',       376: 'Israel',          380: 'Italy',       392: 'Japan',
    410: 'South Korea',   484: 'Mexico',          528: 'Netherlands', 578: 'Norway',
    616: 'Poland',        620: 'Portugal',        703: 'Slovakia',    724: 'Spain',
    752: 'Sweden',        756: 'Switzerland',     792: 'Turkey',      826: 'United Kingdom',
     32: 'Argentina',      76: 'Brazil',          156: 'China',       356: 'India'
}

mfg_chapters  = ','.join(str(c) for c in range(28, 98))
partner_list  = list(code_to_country.keys())
batch_size    = 5
records       = []

for i in range(0, len(partner_list), batch_size):
    batch = ','.join(str(p) for p in partner_list[i:i+batch_size])
    link  = (
        "https://comtradeapi.un.org/data/v1/get/C/A/HS"
        f"?reporterCode=842&partnerCode={batch}&period=2013"
        f"&flowCode=M&cmdCode={mfg_chapters}"
    )
    r = requests.get(link, headers={"Ocp-Apim-Subscription-Key": COMTRADE_KEY}, timeout=60)
    records.extend(r.json()['data'])

imports_data             = pd.DataFrame(records)[['partnerCode', 'primaryValue']]
imports_data             = imports_data.groupby('partnerCode')['primaryValue'].sum().reset_index()
imports_data['Country']  = imports_data['partnerCode'].map(code_to_country)
imports_data             = imports_data[['Country', 'primaryValue']].rename(columns={'primaryValue': 'Imports'})

data  = pd.merge(ilc_data, imports_data, on='Country', how='outer').set_index('Country')
us    = data.loc['United States']['Wage']
data.drop('United States', inplace=True)
row   = (data['Wage'] * (data['Imports'] / data['Imports'].sum())).sum()

#####################################

w_to_wstar = us/row

### --- Saving to pre-GMM parameters --- ###
parameters = {'σ':σ,'δ':δ,'ρ':ρ,'σ_ϵ':σ_ϵ,'γ':γ,'M':M,'π_LL':π_LL,'π_HH':π_HH}
moments    = {'skill_premium':skill_premium,'K/Y':k_to_Y,'HS_share':HS_share,'I':I,'w_to_wstar':w_to_wstar}

save_dict = {'parameters':parameters,'moments':moments}

with open(DATA_PARAMS / 'pre_gmm_params.json', 'w') as f:
    json.dump(save_dict, f, indent=4)