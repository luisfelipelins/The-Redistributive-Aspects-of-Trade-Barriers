# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 18:49:29 2025

@author: lfval
"""

from fredapi import Fred
from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import requests
import io

fred_api = open(CONFIG / 'fred_key.txt','r').read()
fred = Fred(api_key=fred_api)

############################
### --- Slides Facts --- ###
############################


# Real Wage (FRED-BLS)
mfg_wage  = fred.get_series('CES3000000008').loc['1979-01-01':]
pbs_wage  = fred.get_series('CES6000000008').loc['1979-01-01':]
fin_wage  = fred.get_series('CES5500000008').loc['1979-01-01':]
inflation = fred.get_series('CPIAUCSL')

mfg_wage  = mfg_wage.div(mfg_wage.loc['1979-01-01'])
pbs_wage  = pbs_wage.div(pbs_wage.loc['1979-01-01'])
fin_wage  = fin_wage.div(fin_wage.loc['1979-01-01'])
inflation = inflation.div(inflation.loc['1979-01-01'])

mfg_wage = (mfg_wage/inflation)*100
pbs_wage = (pbs_wage/inflation)*100
fin_wage = (fin_wage/inflation)*100

fig,ax=plt.subplots(ncols=1,figsize=(9,5))
ax.axhline(100,color='black')
ax.plot(mfg_wage,color='darkgreen',label='Manufacturing')
ax.plot(pbs_wage,color='darkblue',label='Professional and Business Services')
ax.plot(fin_wage,color='darkred',label='Financial Services')
ax.grid(linestyle='--')
ax.set_title('Real Wage Indices (Source: BLS, FRED)',fontweight='bold',loc='left')
ax.set_ylabel('Index (Jan/1979 = 100)')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_real_wages.pdf')
plt.close()

# Income Inequality (BLS-Consumer Expenditure Surveys)
url2012 = 'https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/cu-income-quintiles-before-taxes-2012.xlsx'
url2024 = 'https://www.bls.gov/cex/tables/calendar-year/mean-item-share-average-standard-error/cu-income-quintiles-before-taxes-2024.xlsx'

_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def import_data(url):
    r = requests.get(url, headers=_headers)
    r.raise_for_status()
    data  = pd.read_excel(io.BytesIO(r.content))

    tot_i  = data.loc[(data.iloc[:,0]=='Money income before taxes')|(data.iloc[:,0]=='Income before taxes')].index
    wage_i = data.loc[data.iloc[:,0]=='Wages and salaries'].index
    ss_i   = data.loc[data.iloc[:,0]=='Social Security, private and government retirement'].index
    cap_i  = data.loc[(data.iloc[:,0]=='Interest, dividends, rental income, other property income')|(data.iloc[:,0]=='Interest, dividends, rental income, and other property income')].index
    
    index  = ['Name','All','p1','p2','p3','p4','p5']
    tot    = data.loc[tot_i+1].iloc[0].replace({'Mean':'Total'}).set_axis(index)
    wage   = data.loc[wage_i+1].iloc[0].replace({'Mean':'Wages'}).set_axis(index)
    ss     = data.loc[ss_i+1].iloc[0].replace({'Mean':'Social Security'}).set_axis(index)
    cap    = data.loc[cap_i+1].iloc[0].replace({'Mean':'Capital'}).set_axis(index)

    data         = pd.concat([tot,wage,ss,cap],axis=1)
    data.columns = data.iloc[0]
    data         = data[1:]

    data['Total ex-Social Security'] = data['Total'] - data['Social Security']
    data = data[['Total ex-Social Security','Wages','Capital']]

    return data

d2012     = import_data(url2012).loc[['p1','p3','p5']]
d2024_raw = import_data(url2024).loc[['p1','p3','p5']]

inflation    = fred.get_series('CPIAUCSL').resample('Y').mean()
inflation    = inflation.div(inflation.loc['2012-12-31'])
inflation    = inflation.loc['2024-12-31']
d2024        = d2024_raw/inflation
d2024_pct    = d2024.div(d2024['Total ex-Social Security'],axis=0)
growth_rates = d2024/d2012-1

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(10,6))

ax[0].barh([f"1st Quintile ({round(d2024_pct['Total ex-Social Security']['p1']*100,1)}%)",
            f"3rd Quintile ({round(d2024_pct['Total ex-Social Security']['p3']*100,1)}%)",
            f"5th Quintile ({round(d2024_pct['Total ex-Social Security']['p5']*100,1)}%)"], 
            growth_rates['Total ex-Social Security'],color='darkgreen')
ax[0].set_axisbelow(True)
ax[0].set_title('Real ex-Soc. Security Income Growth',fontweight='bold')
ax[0].grid(linestyle='--')
ax[0].xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax[1].barh([f"({round(d2024_pct['Wages']['p1']*100,1)}%)",
              f"({round(d2024_pct['Wages']['p3']*100,1)}%)",
              f"({round(d2024_pct['Wages']['p5']*100,1)}%)"], 
             growth_rates['Wages'],color='darkgreen')
ax[1].set_axisbelow(True)
ax[1].set_title('Wages',fontweight='bold')
ax[1].grid(linestyle='--')
ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax[2].barh([f"({round(d2024_pct['Capital']['p1']*100,1)}%)",
              f"({round(d2024_pct['Capital']['p3']*100,1)}%)",
              f"({round(d2024_pct['Capital']['p5']*100,1)}%)"], 
             growth_rates['Capital'],color='darkgreen')
ax[2].set_axisbelow(True)
ax[2].set_title('Capital Income',fontweight='bold')
ax[2].grid(linestyle='--')
ax[1].set_xlabel('2012 to 2024 Growth Rates in Income from Source (parenthesis denote share of income in 2024 from source)')
ax[2].xaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_income_inequality.pdf')
plt.close()

# Manufacturing Employment and Productivity Adjusted (FRED-BLS)
emp  = fred.get_series('MANEMP')
emp  = emp.resample('M').last().bfill()
prod = (1+fred.get_series('PRS30006091').rename('var')/300)
prod.index = prod.index + pd.offsets.MonthEnd(3)
prod = prod.resample('M').last().bfill()
out = (1+fred.get_series('PRS30006042').rename('var')/300)
out.index = out.index + pd.offsets.MonthEnd(3)
out = out.resample('M').last().bfill()

emp = pd.DataFrame(emp.rename('emp'))
emp['prod'] = (out/prod).shift(-12)
emp['emp_cf'] = np.nan

emp['emp_cf'][emp['prod'].dropna().index[0:12]] = emp['emp'][emp['prod'].dropna().index[0:12]]

for i in range(len(emp['prod'].dropna().index)):
    
    if len(emp['prod'].dropna().index)-i<=12: break
    
    tdy = emp['prod'].dropna().index[i]
    tmr = emp['prod'].dropna().index[i+12]
    
    emp['emp_cf'][tmr] = emp['emp_cf'][tdy]*emp['prod'][tdy]

emp = emp.dropna().resample('Q').mean()/1000

fig,ax=plt.subplots(ncols=1,figsize=(9,5))
ax.plot(emp['emp'].rolling(4).mean()   ,color='darkgreen',label='Actual')
ax.plot(emp['emp_cf'].rolling(4).mean(),color='darkblue',label='Counterfactual')
ax.grid(linestyle='--')
ax.set_title('Manufacturing Employment Levels (4Q Rolling Avg., Source: BLS/FRED)',fontweight='bold',loc='left')
ax.set_ylabel('Millions of Persons')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_employment_levels.pdf')
plt.close()

# Number of Establishments
d2002 = pd.read_csv(DATA_RAW / '2002_detailed_industry.dat', delimiter='|').iloc[1:][['NAICS2002_MEANING','ESTAB']]
d2022 = pd.read_csv(DATA_RAW / '2022.dat', delimiter='|')
d2022 = d2022.loc[(d2022['GEO_TTL']=='United States')&(d2022['NAICS2022'].str.len()==3)][['NAICS2022_TTL','ESTAB']]

d2002.columns = ['industry','estab']
d2022.columns = ['industry','estab']

d2002['industry'] = d2022['industry'].values

d2002.set_index('industry',inplace=True)
d2022.set_index('industry',inplace=True)

delta = (d2022-d2002)['estab']
delta['Total'] = delta.sum()
delta = delta.sort_values()/1000
delta.index = [i.replace('manufacturing','mfg.') for i in delta.index]

fig,ax=plt.subplots(ncols=1,figsize=(9,5))

ax.set_axisbelow(True)
ax.grid(linestyle='--')
colors = []
for ind, value in delta.items():
    if ind == 'Total':
        colors.append('darkblue')
    else:
        colors.append('darkgreen')

bars = ax.barh(delta.index, delta.values, color=colors)

ax.set_title(r'$\Delta$ in Mfg. Estab. in the US (Source: Census Bureau)',fontweight='bold')

ax.axvline(0, color='black', linewidth=0.8)
ax.tick_params(axis='y',labelsize=9)
ax.set_xlabel('Change in the Number of Establishments between 2002 and 2022, thousands')

for bar, ind in zip(bars, delta.index):
    width  = bar.get_width()
    
    y_center = bar.get_y() + bar.get_height() / 2
    
    ax.text(width, 
            y_center, 
            f'{round(delta[ind],1)}', 
            va='center', 
            ha='center', 
            fontsize=9,
            bbox=dict(facecolor='white',
                      edgecolor='gray',
                      boxstyle='round,pad=0.2',
                      alpha=0.9))
plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_establishments.pdf')
plt.close()

# fdi
data = pd.read_excel(DATA_RAW / 'bea_bop_direct_investment_position.xlsx',index_col=0).T
data.index = pd.to_datetime(data.index)+pd.offsets.YearEnd()
inflation = fred.get_series('CPIAUCSL').resample('Y').mean()
inflation = inflation.div(inflation.loc['1999-12-31'])
data = data.div(inflation,axis=0).dropna()/1000

fig,ax=plt.subplots(ncols=1,figsize=(9,5))
ax.set_axisbelow(True)
ax.grid(linestyle='--')
ax.stackplot(data.index, data.T, colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
ax.legend(['Europe','Africa','Middle East','Asia & Pacific','Rest of Americas'],loc=2)
ax.set_title('US Manufacturing Firms Real FDI Position (Source: BEA)',fontweight='bold')
ax.set_ylabel('Billions of 1999 US Dollars')
plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_real_fdi_position.pdf')
plt.close()

# profits abroad and domestically
data = pd.read_excel(DATA_RAW / 'bea_bop_direct_investment_income.xlsx',index_col=0).T
data.index = pd.to_datetime(data.index)+pd.offsets.YearEnd()
profits = fred.get_series('N3212C0A144NBEA')
profits.index = pd.to_datetime(profits.index)+pd.offsets.YearEnd()
inflation = fred.get_series('CPIAUCSL').resample('Y').mean()
inflation = inflation.div(inflation.loc['1999-12-31'])
data = data.div(inflation,axis=0).dropna()/1000
profits = profits.div(inflation,axis=0).dropna()/1000

fig,ax=plt.subplots(ncols=2,figsize=(9,5))
ax[0].set_axisbelow(True)
ax[0].grid(linestyle='--')
ax[0].stackplot(data.index, data.T, colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
ax[0].legend(['Europe','Africa','Middle East','Asia & Pacific','Rest of Americas'],loc=2)
ax[0].set_title('US Mfg. Firms Real FDI Income (Source: BEA)',fontweight='bold')
ax[0].set_ylabel('Billions of 1999 US Dollars')

ax[1].plot(profits,marker='o',color='darkgreen')
ax[1].grid(linestyle='--')
ax[1].set_title('Mfg. Firms Real Profits after Tax',fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_real_fdi_income.pdf')
plt.close()


########################
### --- WP Facts --- ###
########################

# Fact 1: employment and establishments
emp  = fred.get_series('MANEMP')
emp  = emp.resample('M').last().bfill()
prod = (1+fred.get_series('PRS30006091').rename('var')/300)
prod.index = prod.index + pd.offsets.MonthEnd(3)
prod = prod.resample('M').last().bfill()
out = (1+fred.get_series('PRS30006042').rename('var')/300)
out.index = out.index + pd.offsets.MonthEnd(3)
out = out.resample('M').last().bfill()

emp = pd.DataFrame(emp.rename('emp'))
emp['prod'] = (out/prod).shift(-12)
emp['emp_cf'] = np.nan

emp['emp_cf'][emp['prod'].dropna().index[0:12]] = emp['emp'][emp['prod'].dropna().index[0:12]]

for i in range(len(emp['prod'].dropna().index)):
    
    if len(emp['prod'].dropna().index)-i<=12: break
    
    tdy = emp['prod'].dropna().index[i]
    tmr = emp['prod'].dropna().index[i+12]
    
    emp['emp_cf'][tmr] = emp['emp_cf'][tdy]*emp['prod'][tdy]

emp = emp.dropna().resample('Q').mean()/1000

fig,ax=plt.subplots(ncols=2,figsize=(12,5),gridspec_kw={'width_ratios': [1.2,1]})

ax[0].plot(emp['emp'].rolling(4).mean()   ,color='darkgreen',label='Actual')
ax[0].plot(emp['emp_cf'].rolling(4).mean(),color='darkblue',label='Counterfactual')
ax[0].grid(linestyle='--')
ax[0].set_ylabel('Millions of Persons',fontsize=11)
ax[0].legend(loc=0,fontsize=11)
ax[0].tick_params(labelsize=11)
ax[0].set_title('Mfg. Emp. Levels (BLS/FRED)', fontweight='bold')

d2002 = pd.read_csv(DATA_RAW / '2002_detailed_industry.dat', delimiter='|').iloc[1:][['NAICS2002_MEANING','ESTAB']]
d2022 = pd.read_csv(DATA_RAW / '2022.dat', delimiter='|')
d2022 = d2022.loc[(d2022['GEO_TTL']=='United States')&(d2022['NAICS2022'].str.len()==3)][['NAICS2022_TTL','ESTAB']]

d2002.columns = ['industry','estab']
d2022.columns = ['industry','estab']

d2002['industry'] = d2022['industry'].values

d2002.set_index('industry',inplace=True)
d2022.set_index('industry',inplace=True)

delta = (d2022-d2002)['estab']
delta['Total'] = delta.sum()
delta = delta.sort_values()/1000
delta.index = [i.replace('manufacturing','').replace('Electrical equipment, appliance, and component','Electrical equipment').replace('and related support activities','').replace('Beverage and tobacco product','Beverage and tobacco').replace('Furniture and related product','Furniture').replace('Petroleum and coal products','Oil and coal').replace('Plastics and rubber products','Plastics and rubber').replace('Computer and electronic product','Computer and electronic') for i in delta.index]

ax[1].set_axisbelow(True)
ax[1].grid(linestyle='--')
colors = []
for ind, value in delta.items():
    if ind == 'Total':
        colors.append('darkblue')
    else:
        colors.append('darkgreen')

bars = ax[1].barh(delta.index, delta.values, color=colors)

ax[1].axvline(0, color='black', linewidth=0.8)
ax[1].tick_params(axis='y',labelsize=11)
ax[1].tick_params(axis='x',labelsize=11)
ax[1].set_xlabel(r'$\Delta$ in the # of Est. between 2002 and 2022, thousands',fontsize=11)
ax[1].set_title(r'$\Delta$ in Mfg. Est. in the US (Census Bureau)', fontweight='bold')

for bar, ind in zip(bars, delta.index):
    width  = bar.get_width()

    y_center = bar.get_y() + bar.get_height() / 2

    ax[1].text(width,
            y_center,
            f'{round(delta[ind],1)}',
            va='center',
            ha='center',
            fontsize=11,
            bbox=dict(facecolor='white',
                      edgecolor='gray',
                      boxstyle='round,pad=0.2',
                      alpha=0.9))
plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_fact1.pdf')
plt.close()

# Fact 2: Real FDI position and income with profits after taxes
fdi_pos = pd.read_excel(DATA_RAW / 'bea_bop_direct_investment_position.xlsx', index_col=0).T
fdi_pos.index = pd.to_datetime(fdi_pos.index) + pd.offsets.YearEnd()
fdi_pos = fdi_pos.div(inflation, axis=0).dropna() / 1000

fig, ax = plt.subplots(ncols=3, figsize=(14, 5))

ax[0].set_axisbelow(True)
ax[0].grid(linestyle='--')
ax[0].stackplot(fdi_pos.index, fdi_pos.T, colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
ax[0].legend(['Europe', 'Africa', 'Middle East', 'Asia & Pacific', 'Rest of Americas'], loc=2, fontsize=10)
ax[0].set_title('Real FDI Position (BEA)', fontweight='bold')
ax[0].set_ylabel('Billions of 1999 USD', fontsize=11)
ax[0].tick_params(labelsize=11)

ax[1].set_axisbelow(True)
ax[1].grid(linestyle='--')
ax[1].stackplot(data.index, data.T, colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
ax[1].legend(['Europe', 'Africa', 'Middle East', 'Asia & Pacific', 'Rest of Americas'], loc=2, fontsize=10)
ax[1].set_title('Real FDI Income (BEA)', fontweight='bold')
ax[1].set_ylabel('Billions of 1999 USD', fontsize=11)
ax[1].tick_params(labelsize=11)

ax[2].plot(profits, marker='o', color='darkgreen')
ax[2].grid(linestyle='--')
ax[2].set_title('Real Mfg. Profits after Tax (BEA)', fontweight='bold')
ax[2].set_ylabel('Billions of 1999 USD', fontsize=11)
ax[2].tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_fact2.pdf')
plt.close()

# Fact 3: Real wages and income inequality
fig = plt.figure(figsize=(12, 5))
outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
ax0   = fig.add_subplot(outer[0])
inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.5)
ax1, ax2, ax3 = [fig.add_subplot(inner[i]) for i in range(3)]

ax0.axhline(100, color='black')
ax0.plot(mfg_wage, color='darkgreen', label='Manufacturing')
ax0.plot(pbs_wage, color='darkblue', label='Professional and Business Services')
ax0.plot(fin_wage, color='darkred', label='Financial Services')
ax0.grid(linestyle='--')
ax0.set_title('Real Wage Indices (BLS, FRED)', fontweight='bold')
ax0.set_ylabel('Index (Jan/1979 = 100)', fontsize=11)
ax0.legend(loc=0, fontsize=11)
ax0.tick_params(labelsize=11)

ax1.barh([f"1st QU (100%)",
          f"3rd QU (100%)",
          f"5th QU (100%)"],
         growth_rates['Total ex-Social Security'], color='darkgreen')
ax1.set_axisbelow(True)
ax1.set_title('Real Income Growth (ex-Soc. Sec.)', fontweight='bold')
ax1.grid(linestyle='--')
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax1.tick_params(labelsize=11)

ax2.barh([f"1st QU ({round(d2024_pct['Wages']['p1']*100,1)}%)",
          f"3rd QU ({round(d2024_pct['Wages']['p3']*100,1)}%)",
          f"5th QU ({round(d2024_pct['Wages']['p5']*100,1)}%)"],
         growth_rates['Wages'], color='darkgreen')
ax2.set_axisbelow(True)
ax2.set_title('Wages', fontweight='bold')
ax2.grid(linestyle='--')
ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax2.tick_params(labelsize=11)

ax3.barh([f"1st QU ({round(d2024_pct['Capital']['p1']*100,1)}%)",
          f"3rd QU ({round(d2024_pct['Capital']['p3']*100,1)}%)",
          f"5th QU ({round(d2024_pct['Capital']['p5']*100,1)}%)"],
         growth_rates['Capital'], color='darkgreen')
ax3.set_axisbelow(True)
ax3.set_title('Capital Income', fontweight='bold')
ax3.grid(linestyle='--')
ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax3.tick_params(labelsize=11)
ax3.set_xlabel('2012–2024 Growth in Real Income by Source\n(parentheses = share of 2024 income from source)', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUTS_MOTIVATION / 'motivation_fact3.pdf', bbox_inches='tight')
plt.close()
