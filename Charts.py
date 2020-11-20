import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

os.chdir('P:\ECB business areas\DGE\DMP\MAY\Analytical\Orbis\ORBIS_NEW\Python_charts')
df = pd.read_csv('db_unconsolidated_tangible.csv', sep = ',')
#df = pd.read_csv('db_consolidated.csv', sep = ',')


df = df[df.country=="IT"]
df_debt = df.groupby('year').tot_debt.sum()
df_ta = df.groupby('year').tangible_fixed_assets.sum()













QMA_format = (11.3, 9)
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


# =============================================================================
# CHART 1.2: Joint density of Cash and Debt holdings (relative to assets) in 2018
# =============================================================================


i = sns.JointGrid(df[df.year == 2018].leverage_assets[df[df.year == 2018].leverage_assets<1],
                  df[df.year == 2018].cash_assets[df[df.year == 2018].cash_assets<1],
                  ratio=100,
                  space=0,
                  size=6,
                  xlim=[0,0.6],
                  ylim=[0,0.6])

i.plot_joint(sns.kdeplot, cmap="Blues", shade=True, shade_lowest=False, height=9)
i.set_axis_labels('Debt on assets','Cash on assets', fontsize=18)
plt.tick_params(axis="both", labelsize=18)

i.savefig('Chart_1_2.png')

## Chart 1.2 reloaded: quantile regression  GIULIO


import statsmodels.api as sm
import statsmodels.formula.api as smf

#DataFrame.quantile(q=0.5, axis=0, numeric_only=True, interpolation=’linear’)
#df.head()

# this works only with df = pd.read_csv('db_unconsolidated.csv', sep = ',') and not with new dataset
#res = mod.fit(q=.5)
#print(res.summary())

quantiles = np.arange(.05, .96, .1)


def fit_model(q,moda):
    res = moda.fit(q=q)
    return [q, res.params['Intercept'], res.params['cash_assets']] + \
            res.conf_int().loc['cash_assets'].tolist()
            
moda = smf.quantreg('leverage_assets ~ cash_assets', df[df.year==2007])
models = [fit_model(x,moda) for x in quantiles]
models2007 = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

moda = smf.quantreg('leverage_assets ~ cash_assets', df[df.year==2009])
models = [fit_model(x,moda) for x in quantiles]
models2009 = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

moda = smf.quantreg('leverage_assets ~ cash_assets', df[df.year==2014])
models = [fit_model(x,moda) for x in quantiles]
models2014 = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

moda = smf.quantreg('leverage_assets ~ cash_assets', df[df.year==2018])
models = [fit_model(x,moda) for x in quantiles]
models2018 = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])




plt.figure(figsize=(10,9))
p0 = plt.plot(models2007.q, models2007.b, color='blue', label='2007')
p1 = plt.plot(models2009.q, models2009.b, linestyle='dotted', color='blue', label='2009')
p2 = plt.plot(models2014.q, models2014.b, linestyle='dotted', color='black', label='2014')
p3 = plt.plot(models2018.q, models2018.b, color='black', label='2018')
plt.ylabel(r'$\beta_{Cash}$', fontsize= 16)
plt.xlabel('Quantiles of the conditional gross debt distribution', fontsize= 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.show()




models2018.to_csv('models2018_s.csv', sep=',', encoding='utf-8', index = False)
models2014.to_csv('models2014_s.csv', sep=',', encoding='utf-8', index = False)
models2009.to_csv('models2009_s.csv', sep=',', encoding='utf-8', index = False)
models2007.to_csv('models2007_s.csv', sep=',', encoding='utf-8', index = False)



# =============================================================================
# CHART 1.3: Profit margin by group of firms: debt increasing versus decreasing 
# =============================================================================


# first define groupings 
df['grouping1'] = abs(df.devmarg_up) > 0 
df['grouping2'] = abs(df.devmarg_down) > 0 
df['grouping'] = (df.grouping2 | df.grouping1)

df_cut = df[df.grouping==True]

# now take the prof_margin variable 
aup = np.percentile(df_cut.prof_ultimate_margin,98)
adown = np.percentile(df_cut.prof_ultimate_margin,2)

df_cut_supercut5 = df_cut[df_cut.prof_ultimate_margin<aup]
df_cut_supercut6 = df_cut_supercut5[df_cut_supercut5.prof_ultimate_margin>adown]
# give names to the last  
df_cut_supercut6['Debt'] = df_cut_supercut6.grouping1

df_cut_supercut6.Debt[df_cut_supercut6.Debt==True]='Decrease'
df_cut_supercut6.Debt[df_cut_supercut6.Debt==False]='Increase'

fig, ax = plt.subplots(figsize=QMA_format)

i=sns.boxplot(x="year", 
              y="prof_ultimate_margin", 
              hue="Debt", 
              palette=["m", "g"], 
              data=df_cut_supercut6,
              showfliers=False, 
              showmeans=True,
              ax=ax)

i.set_ylabel("Profit margin", fontsize=22)
i.set_xlabel("")
i.tick_params(labelsize=18) #Use this one to edit the size of the axis ticks (year, profit)

del adown
del aup
del df_cut
del df_cut_supercut5
del df_cut_supercut6



# =============================================================================
# CHART 1.4: Profit margin by group of firms: cash increasing versus decreasing 
# =============================================================================


# con la cash 
df['grouping3'] = abs(df.devmarg_cashup) > 0 
df['grouping4'] = abs(df.devmarg_cashdown) > 0 
df['grouping_c'] = (df.grouping3 | df.grouping4)

df_cut_cash = df[df.grouping_c==True]

# either devmarg or prof_ultimate_margin
aup = np.percentile(df_cut_cash.prof_ultimate_margin,98)
adown = np.percentile(df_cut_cash.prof_ultimate_margin,2)

df_cut_supercut3= df_cut_cash[df_cut_cash.prof_ultimate_margin<aup]
df_cut_supercut4 = df_cut_supercut3[df_cut_supercut3.prof_ultimate_margin>adown]

df_cut_supercut4['Cash'] = df_cut_supercut4.grouping4

df_cut_supercut4.Cash[df_cut_supercut4.Cash==True]='Decrease'
df_cut_supercut4.Cash[df_cut_supercut4.Cash==False]='Increase'

fig, ax = plt.subplots(figsize=QMA_format)

i=sns.boxplot(x="year", 
              y="prof_ultimate_margin", 
              hue="Cash", 
              palette=["m", "g"], 
              data=df_cut_supercut4, 
              showfliers=False, 
              showmeans=True,
              ax=ax)

i.set_ylim([-0.3,0.3])
i.set_ylabel("Profit margin", fontsize=22)
i.set_xlabel("")
i.tick_params(labelsize=18) #Use this one to edit the size of the axis ticks (year, profit)

del adown
del aup
del df_cut_cash
del df_cut_supercut3
del df_cut_supercut4



# =============================================================================
# CHART 1.5: Joint density of Cash and Debt changes (relative to assets) in 2018
# =============================================================================


i = sns.JointGrid(-df.debt_rep[df.year == 2018],
                  df.dcash_pc[df.year == 2018],
                  ratio=100,
                  space=0,
                  size=6,
                  xlim=[-0.15,0.15],
                  ylim=[-0.15,0.15])

i.plot_joint(sns.kdeplot, cmap="Blues", shade=True, shade_lowest=False)
i.set_axis_labels('Debt change','Cash change', fontsize=18)
plt.tick_params(axis="both", labelsize=18)



# =============================================================================
# CHART 1.6: Sectoral developments
# =============================================================================


df['grouping1'] = abs(df.devmarg_up) > 0 
df['grouping2'] = abs(df.devmarg_down) > 0 
df['grouping'] = (df.grouping2 | df.grouping1)


# down are debt increasers 
index_2012_down = df.id[(df.grouping2==True) & (df.year==2012)].tolist()
index_2012_up = df.id[(df.grouping1==True) & (df.year==2012)].tolist()

index_2017_down = df.id[(df.grouping2) & (df.year==2017)].tolist()
index_2017_up = df.id[(df.grouping1) & (df.year==2017)].tolist()

# now we look at firms that crossed group
firms_recovering = pd.Series(list(set(index_2012_down) & set(index_2017_up))).astype(str).values.tolist()
firms_worsening = pd.Series(list(set(index_2012_up) & set(index_2017_down))).astype(str).values.tolist()


# now we create variables in the dataframe to extract them afterwards
df['debt_increaser_2012'] = df['id'].isin(index_2012_up)
df['debt_increaser_2017'] = df['id'].isin(index_2017_up)
df['debt_reducer_2012'] = df['id'].isin(index_2012_down)
df['debt_reducer_2017'] = df['id'].isin(index_2017_down)
df['Worsening'] = df['id'].isin(firms_worsening)
df['Improving'] = df['id'].isin(firms_recovering)

del index_2012_up
del index_2012_down
del index_2017_up
del index_2017_down
del firms_recovering
del firms_worsening

values = df.id[(df.debt_reducer_2017==True) & (df.year==2017) & ((df.country !="FR")&(df.country !="DE")&(df.country !="ES")&(df.country !="IT"))].groupby(df['country']).size()

values = df.id[(df.debt_increaser_2012==True) & (df.year==2012)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index, palette="rocket")
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")
w.set_title("2012 - Debt Increasing Firms")
w.savefig('debt_increasers_2012.png')

values = df.id[(df.debt_increaser_2017==True) & (df.year==2017)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index,palette="rocket")
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")
w.set_title("2017 - Debt Increasing Firms")
w.savefig('debt_increasers_2017.png')

values = df.id[(df.debt_reducer_2012==True) & (df.year==2017)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index)
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")
w.set_title("2012 - Debt Reducing Firms")
w.savefig('debt_reducers_2012.png')

values = df.id[(df.debt_reducer_2017==True) & (df.year==2017)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index)
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")
w.set_title("2017 - Debt Reducing Firms")
w.savefig('debt_reducers_2017.png')


values = df.id[(df.Worsening==True) & (df.year==2017)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index,palette="rocket")
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")

values = df.id[(df.Improving==True) & (df.year==2017)].groupby(df['sector_new']).size().sort_values(ascending=False)
w=sns.barplot(x=values, y=values.index)
w.set_ylabel("Sectors")
w.set_xlabel("Number of companies")



# =============================================================================
# CHART 1.7: Leverage of the two groups
# =============================================================================


df_improving = df[(df.Improving | df.Worsening) & ((df.year==2017) | (df.year==2012))]
df_improving['Margins'] = np.where(df_improving.Worsening, 'Decrease', 'Increase')

fig, ax = plt.subplots(figsize=QMA_format)

i = sns.boxplot(x="year",
                y="leverage_assets", 
                hue="Margins", 
                data=df_improving,
                palette=["g", "m"], 
                showfliers=False, 
                showmeans=True,
                ax=ax)

i.set_ylabel("Leverage", fontsize=22)
i.set_xlabel("Year", fontsize=22)
i.tick_params(labelsize=18)
i.legend(loc='top left', ncol=2, fontsize= 18)



# =============================================================================
# CHART 1.8: Interest coverage by group of firms: debt  increasing versus decreasing
# =============================================================================


# first define groupings 
df['grouping1'] = abs(df.devmarg_up) > 0 
df['grouping2'] = abs(df.devmarg_down) > 0 
df['grouping'] = (df.grouping2 | df.grouping1)

df_cut = df[(df.grouping==True) & (df.intcover)]

# then cut the variable you are interested in doing the chart
aup = np.percentile(df_cut.intcover,98)
adown = np.percentile(df_cut.intcover,2)

df_cut_supercut1= df_cut[df_cut.intcover<aup]
df_cut_supercut2 = df_cut_supercut1[df_cut_supercut1.intcover>adown]
# give names to the last  
df_cut_supercut2['Debt'] = df_cut_supercut2.grouping1

df_cut_supercut2.Debt[df_cut_supercut2.Debt==True]='Decrease'
df_cut_supercut2.Debt[df_cut_supercut2.Debt==False]='Increase'

fig, ax = plt.subplots(figsize=QMA_format)

i=sns.boxplot(x="year", 
              y="intcover", 
              hue="Debt", 
              palette=["m", "g"], 
              data=df_cut_supercut2[df_cut_supercut2.year != 2018], 
              showfliers=False, 
              showmeans=True,
              ax=ax)

i.set_ylabel("Debt burden ratio", fontsize=22)
i.set_xlabel("")
i.tick_params(labelsize=18) #Use this one to edit the size of the axis ticks (year, profit)
i.legend(fontsize= 18)

del adown
del aup
del df_cut
del df_cut_supercut1
del df_cut_supercut2



# =============================================================================
# CHART 1.10: Average interest rate
# =============================================================================


# first define groupings 
df['grouping1'] = abs(df.devmarg_up) > 0 
df['grouping2'] = abs(df.devmarg_down) > 0 
df['grouping'] = (df.grouping2 | df.grouping1)

df_cut = df[(df.grouping==True) & (df.avg_interest_rate)]

# then cut the variable you are interested in doing the chart
aup = np.percentile(df_cut.avg_interest_rate,98)
adown = np.percentile(df_cut.avg_interest_rate,2)

df_cut_supercut1= df_cut[df_cut.avg_interest_rate<aup]
df_cut_supercut2 = df_cut_supercut1[df_cut_supercut1.avg_interest_rate>adown]
# give names to the last  
df_cut_supercut2['Debt'] = df_cut_supercut2.grouping1

df_cut_supercut2.Debt[df_cut_supercut2.Debt==True]='Decrease'
df_cut_supercut2.Debt[df_cut_supercut2.Debt==False]='Increase'

fig, ax = plt.subplots(figsize=QMA_format)

i=sns.boxplot(x="year", 
              y="avg_interest_rate", 
              hue="Debt", 
              palette=["m", "g"], 
              data=df_cut_supercut2[df_cut_supercut2.year != 2018], 
              showfliers=False, 
              showmeans=True,
              ax=ax)

i.set_ylabel("Average interest rate", fontsize=22)
i.set_xlabel("")
i.tick_params(labelsize=18) #Use this one to edit the size of the axis ticks (year, profit)
i.legend(fontsize= 18)

del adown
del aup
del df_cut
del df_cut_supercut1
del df_cut_supercut2



#------------------------------------------------------------------------------
# CHART A.3: LEVERAGE ON ASSETS ACROSS COUNTRIES
#------------------------------------------------------------------------------



#df1 = df[(df.year==2007) & ((df.country =="FR") |(df.country =="DE")| (df.country =="ES")| (df.country =="IT"))]
#df2 = df[(df.year == 2017) & ((df.country =="FR") |(df.country =="DE")| (df.country =="ES")| (df.country =="IT"))]

df['2007'] = df.leverage[df.year==2007]
df['2017'] = df.leverage[df.year==2017]
pal = sns.cubehelix_palette(df.country.nunique(), rot=-.7, light=.7)

g = sns.FacetGrid(df[(df.leverage < 3) & ((df.country =="FR") |(df.country =="DE")| (df.country =="ES")| (df.country =="IT")) ], row="country", hue="country", size=.7, aspect=15, palette=pal)

g.map(sns.kdeplot, "2007", clip_on=False, shade=True, alpha=1, lw=0, bw=.2) # plots the colored line above the white line
g.map(sns.kdeplot, "2007", clip_on=False, color="w", lw=2, bw=.2) # plots the white line
g.map(sns.kdeplot, "2017", clip_on=False, shade=True, alpha=1, lw=0, bw=.2) # plots the colored line above the white line
g.map(sns.kdeplot, "2017", clip_on=False, color="w", lw=2, bw=.2) # plots the white line
g.map(plt.axhline, y=0, lw=1.5, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontsize = 16, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "leverage")

g.fig.subplots_adjust(hspace=-.35)

# Remove axes details that don't play well with overlap
g.set_titles("")

g.set_axis_labels("Leverage")
g.set_xlabels(fontsize = 16)
g.set_xticklabels(fontsize=14)

g.set(yticks=[])
g.despine(bottom=True, left=True)    



#------------------------------------------------------------------------------
# CHART A.4: LEVERAGE ON ASSETS ACROSS COUNTRIES AND YEARS
#------------------------------------------------------------------------------


f, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df.net_leverage_assets[(df.country=="DE") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df.net_leverage_assets[(df.country=="DE") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df.net_leverage_assets[(df.country=="DE") & (df.year==2018)], hist=False, color="#FF4B00", label='2018', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df.net_leverage_assets[(df.country=="ES") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df.net_leverage_assets[(df.country=="ES") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df.net_leverage_assets[(df.country=="ES") & (df.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df.net_leverage_assets[(df.country=="FR") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df.net_leverage_assets[(df.country=="FR") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df.net_leverage_assets[(df.country=="FR") & (df.year==2018)], hist=False, color="#FF4B00", label='2018', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df.net_leverage_assets[(df.country=="IT") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df.net_leverage_assets[(df.country=="IT") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
a = sns.distplot(df.net_leverage_assets[(df.country=="IT") & (df.year==2018)], hist=False, color="#FF4B00", label='2018', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# 

#------------------------------------------------------------------------------
# CHART A.4: NET LEVERAGE ON ASSETS ACROSS COUNTRIES AND YEARS
#------------------------------------------------------------------------------

# FILIPPO: remove outliers here 

#from scipy import stats
#df_cut=df[(np.abs(stats.zscore(df.net_leverage)) < 3).all(axis=1)]

df_cut = df

f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 3)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage[(df.country=="DE") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage[(df.country=="DE") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage[(df.country=="DE") & (df.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 3)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage[(df.country=="ES") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage[(df.country=="ES") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage[(df.country=="ES") & (df.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 3)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage[(df.country=="FR") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage[(df.country=="FR") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage[(df.country=="FR") & (df.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 3)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage[(df.country=="IT") & (df.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage[(df.country=="IT") & (df.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage[(df.country=="IT") & (df.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# energy sector















#------------------------------------------------------------------------------
# ISOLATE REPEATING FIRMS
#------------------------------------------------------------------------------






# down are debt increasers 
firms_2014 = df.id[(df.year==2014)].tolist()
firms_2017 = df.id[(df.year==2017)].tolist()

df['2014'] = df['id'].isin(firms_2014)
df['2017'] = df['id'].isin(firms_2017)

df_selected = df[(df['2014']==True) & (df['2017']==True)] 

df= df_selected



#------------------------------------------------------------------------------
# CHART A.4: LEVERAGE ON ASSETS ACROSS SECTORS AND YEARS
#------------------------------------------------------------------------------

df_cut = df

f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)


axes[0,0].set_title("Durable", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("Non-Durable", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Non Durable") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Non Durable") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Non Durable") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("Real Estate and Construction", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("Energy", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# other chart
f, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True)

axes[0,0].set_title("Information, Communications and R&D", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("Healthcare", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Healthcare") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Healthcare") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df.sector_new=="Healthcare") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("Wholesale and Retail Trade", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("Other", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Other") & (df_cut.year==2007)], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Other") & (df_cut.year==2014)], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Other") & (df_cut.year==2017)], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()








# END 

# now do one sector, many countries 
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2007) & (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2017) & (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2007)& (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2017)& (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Real Estate Services") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
# 
# now do different countries, same industry 
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2007) & (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2017) & (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2007)& (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2017)& (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Non Durable") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()


# now do different countries, same industry 
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2007) & (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2017) & (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2007)& (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2017)& (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Durable") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()



f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

#wholesale
axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("FR", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007) & (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017) & (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("ES", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007)& (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()









f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

#wholesale
axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("FR", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2007) & (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2017) & (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("ES", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2007)& (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2017)& (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Information, Communications and R&D") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# energy 
# now do different countries, same industry 
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2007) & (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2017) & (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2007)& (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2017)& (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Energy") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

## all other stuff 

axes[0,0].set_title("DE", fontsize=15)
axes[0,0].set_ylim(0, 2.5)
axes[0,0].set_xlim(-0.2, 1)
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007) & (df_cut.country=="DE")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="DE")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="DE")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 0])
a.set_xlabel('')
a.xaxis.set_tick_params(labelbottom=True)
a.tick_params(axis='both', which='major', labelsize=12)

axes[0,1].set_title("ES", fontsize=15)
axes[0,1].set_ylim(0, 2.5)
axes[0,1].set_xlim(-0.2, 1)
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007) & (df_cut.country=="ES")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="ES")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017) & (df_cut.country=="ES")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[0, 1])
b.set_xlabel('')
b.xaxis.set_tick_params(labelbottom=True)
b.tick_params(axis='both', which='major', labelsize=12)

axes[1,0].set_title("FR", fontsize=15)
axes[1,0].set_ylim(0, 2.5)
axes[1,0].set_xlim(-0.2, 1)
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007)& (df_cut.country=="FR")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="FR")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="FR")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 0])
c.set_xlabel('')
c.tick_params(axis='both', which='major', labelsize=12)

axes[1,1].set_title("IT", fontsize=15)
axes[1,1].set_ylim(0, 2.5)
axes[1,1].set_xlim(-0.2, 1)
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2007)& (df_cut.country=="IT")], hist=False, color="#003299", label='2007', kde_kws={'linewidth': 3, 'linestyle':'--'}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2014)& (df_cut.country=="IT")], hist=False, color="#FFB400", label='2014', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d = sns.distplot(df_cut.net_leverage_assets[(df_cut.sector_new=="Wholesale and Retail Trade") & (df_cut.year==2017)& (df_cut.country=="IT")], hist=False, color="#FF4B00", label='2017', kde_kws={'linewidth': 3}, ax=axes[1, 1])
d.set_xlabel('')
d.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()





