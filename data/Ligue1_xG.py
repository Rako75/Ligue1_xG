#pip install highlight_text adjustText fuzzywuzzy

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import numpy as np
from math import pi
from urllib.request import urlopen
import matplotlib.patheffects as pe
from highlight_text import fig_text
from adjustText import adjust_text
from tabulate import tabulate
import matplotlib.style as style
import unicodedata
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.image as mpimg
from PIL import Image
import urllib
import os
import math
import matplotlib.gridspec as gridspec
from highlight_text import ax_text

matchs_22_23 = pd.read_csv('Scores_and_Fixtures_Ligue_1_22_23.csv', sep=';')
matchs_23_24 = pd.read_csv('Scores_and_Fixtures_Ligue_1_23_24.csv', sep=';')
matchs_24_25 = pd.read_csv('Scores_and_Fixtures_Ligue_1_24_25.csv', sep=';', encoding='latin-1')

matchs_22_23.head()

matchs_23_24.head()

matchs_24_25.head()

saisons_multi = pd.concat([matchs_24_25, matchs_23_24, matchs_22_23])

saisons_multi

saisons_multi.isna().sum()

saisons_multi['match_id'] = range(1, len(saisons_multi) + 1)
saisons_multi_ext = saisons_multi[['Away','xG', 'Score', 'xG.1','match_id', 'Date']]
saisons_multi_ext.rename(columns={'Away': 'team'}, inplace=True)
saisons_multi_ext.rename(columns={'xG.1': 'xG_for'}, inplace=True)
saisons_multi_ext.rename(columns={'xG': 'xG_ag'}, inplace=True)
saisons_multi_ext['venue'] = 'A'
saisons_multi_ext[['score_ag','score_for']] = saisons_multi['Score'].str.split('–', expand=True)
saisons_multi_dom = saisons_multi[['Home','xG', 'Score', 'xG.1','match_id', 'Date']]
saisons_multi_dom.rename(columns={'Home': 'team'}, inplace=True)
saisons_multi_dom.rename(columns={'xG.1': 'xG_ag'}, inplace=True)
saisons_multi_dom.rename(columns={'xG': 'xG_for'}, inplace=True)
saisons_multi_dom['venue'] = 'H'
saisons_multi_dom[['score_for','score_ag']] = saisons_multi['Score'].str.split('–', expand=True)
columns = saisons_multi_ext.columns
saisons_multi_dom = saisons_multi_dom[columns]

saisons_multi

saisons_multi.isna().sum()

multi_season_expanded = pd.concat([saisons_multi_ext, saisons_multi_dom])

multi_melted_df = multi_season_expanded.melt(id_vars=['match_id','Date', 'venue', 'team'], value_vars=['score_for', 'score_ag', 'xG_for', 'xG_ag',],
                    var_name='variable', value_name='value')

# Obtenir la liste unique des noms d'équipes
team_names = multi_melted_df['team'].unique()

# Créer une liste de tuples (nom d'équipe, ID d'équipe)
team_data = [(team, i + 1) for i, team in enumerate(team_names)]

# Créer le DataFrame fm_ids
fm_ids = pd.DataFrame(team_data, columns=['team', 'team_id'])

Final_df = multi_melted_df.merge(fm_ids, on='team', how='left')

Final_df.rename(columns={'team': 'team_name'}, inplace=True)

Final_df.rename(columns={'Date': 'date'}, inplace=True)

df = Final_df

df

def get_xG_rolling_data(team_id, window=10, data=df):
    '''
    This function returns xG rolling average figures for a specific team.
    '''
    df = data.copy()
    df_xg = df[(df['team_id'] == team_id) & (df['variable'].isin(['xG_for', 'xG_ag']))]
    df_xg = pd.pivot_table(df_xg,
            index=['date', 'match_id', 'team_id', 'team_name'],columns='variable', values='value', aggfunc= 'first'
        ).reset_index().rename_axis(columns=None)

    # Trier par date (du plus ancien au plus récent)
    df_xg['date'] = pd.to_datetime(df_xg['date'], format='%d/%m/%Y')  # Specify the correct format
    #df_xg = df_xg.sort_values(by=['date'], ascending=True)  # Trier par date ascendante

    df_xg.columns = ['date', 'match_id', 'team_id', 'team_name', 'xG_ag', 'xG_for']
    df_xg['rolling_xG_for'] = df_xg['xG_for'].rolling(window=window, min_periods=0).mean()
    df_xg['rolling_xG_ag'] = df_xg['xG_ag'].rolling(window=window, min_periods=0).mean()
    df_xg['rolling_diff'] = df_xg['rolling_xG_for'] - df_xg['rolling_xG_ag']
    return df_xg

get_xG_rolling_data(5)

def get_xG_interpolated_df(team_id, window=10, data=df):
    # --- Get the xG rolling df
    df_xG = get_xG_rolling_data(team_id, window, data)
    # -- Create interpolated series
    df_xG['match_number'] = df_xG.index
    X_aux = df_xG.match_number.copy()
    X_aux.index = X_aux * 10 # 9 aux points in between each match
    last_idx = X_aux.index[-1] + 1
    X_aux = X_aux.reindex(range(last_idx))
    X_aux = X_aux.interpolate()
    # --- Aux series for the xG created (Y_for)
    Y_for_aux = df_xG.rolling_xG_for.copy()
    Y_for_aux.index = Y_for_aux.index * 10
    last_idx = Y_for_aux.index[-1] + 1
    Y_for_aux = Y_for_aux.reindex(range(last_idx))
    Y_for_aux = Y_for_aux.interpolate()
    # --- Aux series for the xG conceded (Y_ag)
    Y_ag_aux = df_xG.rolling_xG_ag.copy()
    Y_ag_aux.index = Y_ag_aux.index * 10
    last_idx = Y_ag_aux.index[-1] + 1
    Y_ag_aux = Y_ag_aux.reindex(range(last_idx))
    Y_ag_aux = Y_ag_aux.interpolate()
    # --- Aux series for the rolling difference in xG
    Z_diff_aux = df_xG.rolling_diff.copy()
    Z_diff_aux.index = Z_diff_aux.index * 10
    last_idx = Z_diff_aux.index[-1] + 1
    Z_diff_aux = Z_diff_aux.reindex(range(last_idx))
    Z_diff_aux = Z_diff_aux.interpolate()
    # -- Create the aux dataframe
    df_aux = pd.DataFrame({
        'X': X_aux,
        'Y_for': Y_for_aux,
        'Y_ag': Y_ag_aux,
        'Z': Z_diff_aux
    })
    return df_aux

get_xG_interpolated_df(5)

top_6_couleurs = {
    #PSG
    '1': {
        'low': '#990000',
        'high': '#004170'
    },

    #OM
    '2': {
        'low':'#d1d3d4',
        'high':'#00AEEF'
    },

    #Monaco
    '11': {
        'low': '#d1d3d4',
        'high': '#ED1C24'
    },

    #Lille
    '3': {
        'low': '#1D2340',
        'high': '#E61937'
    },

    #Nice
    '5': {
        'low': '#ED1C24',
        'high': '#000000'
    },

    #Lyon
    '9':{
        'low':'#E30613',
        'high':'#0047AB'
    },

    #Lens
    '7':{
        'low':'#D00000',
        'high':'#FFCC00'
    },

    #Brest
    '14':{
        'low':'#d1d3d4',
        'high':'#D31145'
    },

    #Rennes
    '15':{
        'low':'#000000',
        'high':'#D40420'
    }
}

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1-mix)*c1 + mix*c2)

def plot_xG_gradient(ax, team_id, window=10, data=df):
    # -- Get the data
    df_xg = get_xG_rolling_data(team_id, window, data)
    df_aux_xg = get_xG_interpolated_df(team_id, window, data)

    # Specify the axes limits
    ax.set_ylim(0,3)
    ax.set_xlim(-0.5,df_xg.shape[0])
    ax.grid(ls='--', color='white', alpha=0.1)
    ax.set_facecolor('#085eff')


    # -- Select the colors
    color_1 = top_6_couleurs[str(team_id)]['low']
    color_2 = top_6_couleurs[str(team_id)]['high']

    ax.plot(df_xg.index, df_xg['rolling_xG_for'], color=color_2,zorder=4)
    ax.plot(df_xg.index, df_xg['rolling_xG_ag'], color=color_1,zorder=4)
    ax.fill_between(x=[-0.5,window], y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], alpha=0.15, color='black', ec='None',zorder=2)
    vmin = df_xg['rolling_diff'].min()
    vmax = df_xg['rolling_diff'].max()
    vmax = max(abs(vmin), abs(vmax))
    vmin = -1*vmax
    for i in range(0, len(df_aux_xg['X']) - 1):
        ax.fill_between(
            [df_aux_xg['X'].iloc[i], df_aux_xg['X'].iloc[i+1]],
            [df_aux_xg['Y_for'].iloc[i], df_aux_xg['Y_for'].iloc[i + 1]],
            [df_aux_xg['Y_ag'].iloc[i], df_aux_xg['Y_ag'].iloc[i + 1]],
            color=colorFader(color_1, color_2, mix=((df_aux_xg['Z'].iloc[i] - vmin)/(vmax - vmin))),
            zorder=3, alpha=0.3
        )
    for x in [38, 38*2]:
        ax.plot([x,x],[ax.get_ylim()[0], ax.get_ylim()[1]], color='white', alpha=0.35, zorder=2, ls='dashdot', lw=0.95)

    for x in [22, 60]:
        if x == 22:
            text = 'Saison 22-23'
        else:
            text = 'Saison 23-24'
        text_ = ax.annotate(
            xy=(x,2.75),
            text=text,
            color='white',
            size=7,
            va='center',
            ha='center',
            weight='bold',
            zorder=4
        )
        text_.set_path_effects(
            [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]
        )
    ax.tick_params(axis='both', which='major', labelsize=7)
    return ax

fig = plt.figure(figsize=(5,3.5), dpi=300)
ax = plt.subplot(111)

plot_xG_gradient(ax,15,10)

# Chemin vers le fichier de police (à adapter si nécessaire)
font_path = 'BebasNeue-Regular.ttf'

# Charger la police
prop = fm.FontProperties(fname=font_path)

# ---- Path effect for stroke on black curves
def path_effect_stroke(**kwargs):
    return [path_effects.Stroke(**kwargs), path_effects.Normal()]
pe = path_effect_stroke(linewidth=1.5, foreground="white")

# ---- Base settings
fig = plt.figure(figsize=(13, 10), dpi=200)
fig.patch.set_facecolor('#085eff')  # Fond général Ligue 1

nrows = 6
ncols = 3
gspec = gridspec.GridSpec(
    ncols=ncols, nrows=nrows, figure=fig,
    height_ratios=[(1/nrows)*2.35 if x % 2 != 0 else (1/nrows)/2.35 for x in range(nrows)],
    hspace=0.3
)

plot_counter = 0
logo_counter = 0

for row in range(nrows):
    for col in range(ncols):
        if row % 2 != 0:
            ax = plt.subplot(gspec[row, col])
            ax.set_facecolor('#085eff')  # Fond axes
            ax.tick_params(axis='both', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.grid(ls='--', color='white', alpha=0.1)
            teamId = int(list(top_6_couleurs.keys())[plot_counter])
            plot_xG_gradient(ax, teamId, 10)
            plot_counter += 1

        else:
            teamId = int(list(top_6_couleurs.keys())[logo_counter])
            color_1 = top_6_couleurs[str(teamId)]['low']
            color_2 = top_6_couleurs[str(teamId)]['high']

            color_1_t = 'black' if color_1 == '#d1d3d4' else 'white'
            color_2_t = 'black' if color_2 == '#97c1e7' else 'white'

            df_for_text = get_xG_rolling_data(teamId, 10)
            teamName = df_for_text['team_name'].iloc[0]
            xG_for = df_for_text['rolling_xG_for'].iloc[-1]
            xG_ag = df_for_text['rolling_xG_ag'].iloc[-1]

            logo_ax = plt.subplot(gspec[row, col], anchor='NW', facecolor='#085eff')
            club_icon = Image.open(f'logos/{teamId}.png')
            logo_ax.imshow(club_icon)
            logo_ax.axis('off')

            # Texte club
            ax_text(
                x=1.2, y=0.7,
                s=f'<{teamName}>\n<xG pour: {xG_for:.1f}> <|> <xG contre: {xG_ag:.1f}>',
                ax=logo_ax,
                highlight_textprops=[
                    {'weight': 'bold','color': 'white'},
                    {'size': '8', 'bbox': {'edgecolor': color_2, 'facecolor': color_2, 'pad': 1}, 'color': color_2_t},
                    {'color': '#EFE9E6'},
                    {'size': '8', 'bbox': {'edgecolor': color_1, 'facecolor': color_1, 'pad': 1}, 'color': color_1_t}
                ],
                ha='left',
                size=10,
                annotationbbox_kw={'xycoords': 'axes fraction'}
            )
            logo_counter += 1

# ---- Titre & sous-titre
fig_text(
    x=0.135, y=.92,
    s='xG : Quels écarts entre attaque et défense dans le Top 9 de Ligue 1 ?',
    va='bottom', ha='left',
    fontsize=19, color='white', weight='bold',fontproperties=prop
)
fig_text(
    x=0.135, y=.9,
    s='Tendances sur 10 matchs glissants selon les xG pour et contre | @Alex Rakotomalala | Data : Fbref',
    va='bottom', ha='left',
    fontsize=10, color='#cbd5e1',fontproperties=prop
)

# ---- Logo Ligue 1
logo_ax = fig.add_axes([.05, .885, .07, .075])
club_icon = Image.open('logos/ligue1.jpg')
logo_ax.imshow(club_icon)
logo_ax.axis('off')


# Mettre logo RKSTS
ax3 = fig.add_axes([0.85, 0.075, 0.07, 1.7])
ax3.axis('off')
img = mpimg.imread('Logo_RKSTS.png')
ax3.imshow(img)

fig.savefig('xG_Ligue1_Top9.png', dpi=300)
