#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns  
import statsmodels.api as sm
import textwrap
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import t

def compare_phases(df, metric, x_cat, phases, ylim, ylabel, xlabels, skip_subjects=[], col_palette='rocket', sat_val=1, figsize=(4,6), font_size = 20, tick_size = 18, showfliers=True, save_fig=False, fig_title=None, notch_param=False):
    pairs_phases = list(combinations(phases, r=2))
    print(pairs_phases)

    alpha = 0.05
    num_tests = len(pairs_phases)
    print(num_tests)
    alpha_bonferroni = alpha / num_tests
    print(f"Bonferroni-corrected alpha: {alpha_bonferroni}")

    significant_combinations = []
    idx_pair_plotting = []

    subjects = df['subject'].unique()
    for pair in pairs_phases:
        idx1 = phases.index(pair[0])
        idx2 = phases.index(pair[1])

        data1 = [np.mean(df[metric][(df[x_cat] == pair[0]) & (df['subject'] == subject)]) for subject in subjects if subject not in skip_subjects]
        data2 = [np.mean(df[metric][(df[x_cat] == pair[1]) & (df['subject'] == subject)]) for subject in subjects if subject not in skip_subjects]
        
        if np.isnan(data2).any():
            print('CHECK FOR MISSING VALUES')
        # Significance
        res = wilcoxon(data1, data2)
        p = res[1]
        print(p)

        if p < alpha_bonferroni:
            significant_combinations.append([pair, p])
            idx_pair_plotting.append([idx1, idx2])

    print(significant_combinations)
    print(idx_pair_plotting)
    print(f'n subjects: {len(data2)}')

    ## PLOT 
    fig, ax = plt.subplots(1,1,figsize=figsize)

    sns.boxplot(data=df, x=x_cat, y=metric, ax=ax, order=phases, palette=col_palette, saturation=sat_val, linewidth=1,
                showfliers=showfliers,
                showmeans=True,
                notch=notch_param,
                boxprops=dict(edgecolor='k'),
                capprops=dict(color='k'),
                medianprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(markeredgecolor='black'),
                meanprops={"markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"8"})
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=font_size, fontname='Arial', c='k')
    ax.set_xlabel('', fontsize=font_size, fontname='Arial')
    ax.set_xticklabels(xlabels, fontsize=font_size, fontname='Arial')
    
    # Tweak the visual presentation
    labels = [item.get_text() for item in ax.get_xticklabels()]
    wrapped_labels = [textwrap.fill(label, width=10) for label in labels] # Adjust 'width' as needed
    ax.set_xticklabels(wrapped_labels)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    ax.tick_params(axis='both', which='major', labelsize=tick_size) 

    ax.spines[['right', 'top']].set_visible(False)

    # Significance bars
    bottom, top = ax.get_ylim()
    y_range = top - bottom

    extra_y = y_range * 0.07 * (len(significant_combinations)+1)
    ax.set_ylim(bottom, top + extra_y)  # extend ylim BEFORE plotting bars/text!

    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        pval = "%.4f" % significant_combination[1]

        level = i + 1
        y_range = df[metric].max() - df[metric].min()
        bar_height = top + (y_range * 0.08 * level)  # Offset for each bar
        bar_tips = bar_height - (y_range * 0.01)
 
        ax.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=0.8, c='k')
    
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (y_range * 0.01)
        
        ax.text((idx_pair_plotting[i][0] + idx_pair_plotting[i][1]) * 0.5 , text_height, f'p={pval}', ha='center', va='bottom', c='k', fontsize=tick_size)
    
    if save_fig == True:
        plt.savefig(fig_title, bbox_inches='tight', dpi=600)

def run_regression(df1, pairs_data, df2=None, arr1=None, arr2=None, idx=[8, 21], n=0, group='all', figsize=(12,5), showplot=True, x_labels=None, y_label=None, scatter_colors=['lightskyblue'], resplot=True, suptitle1=None, suptitle2=None, save_fig=True, savevals=False,  title1=None):
    if group == 'all':
        data = {'AEE_kJkg':df1['AEE_kJkg'],
                'E2_pmolL_Stricker':df1['E2_pmolL_Stricker'], 'E2_pmolL_Dighe':df1['E2_pmolL_Dighe'], 
                'E2_pmolL_Verdonk':df1['E2_pmolL_Verdonk'], 'P4_nmolL_Stricker':df1['P4_nmolL_Stricker']}
    elif group == 'active':
        data = {'AEE_kJkg':arr1, '[E2]':df1['E2_pmolL'], '[P4]':df1['P4_pmolL'], 
        'dAEE':arr2, 'd[E2]':df2['d[E2]'], 'd[P4]':df2['d[P4]']}
    elif group == 'inactive':
        data = {'AEE_kJkg':arr1, '[E2]':df1['E2_pmolL'], '[P4]':df1['P4_pmolL'], 
        'dAEE':arr2, 'd[E2]':df2['d[E2]'], 'd[P4]':df2['d[P4]']}

    idx1_E2 = idx[0]
    idx2_E2 = idx[1]
    if isinstance(n, list):
        idx1_AEE = n[0]
        idx2_AEE = n[1]
    else:
        idx1_AEE = idx1_E2 + n
        idx2_AEE = idx2_E2 + n

    fig, axes = plt.subplots(1,len(pairs_data),figsize=figsize, constrained_layout=True)
    try:
        axes = axes.flatten()
    except AttributeError:
        pass

    if resplot == True:
        fig2, axes2 = plt.subplots(1,len(pairs_data),figsize=figsize, constrained_layout=True)
        try:
            axes2 = axes2.flatten()
        except AttributeError:
            pass

    r2_vals = []
    for i in range(len(pairs_data)):
        pair = pairs_data[i]

        data1_name = pair[0]
        data2_name = pair[1]
        print(data1_name, data2_name)

        data1 = data[data1_name][idx1_E2:idx2_E2].values
        data2 = data[data2_name][idx1_AEE:idx2_AEE]
        vals = {'X':data1, 'Y':data2}
        df = pd.DataFrame(vals)
        
        # pearson Correlation test
        r, p = stats.pearsonr(data1, data2)
        
        # (1) linear regression
        X = sm.add_constant(df['X'])
        y = df['Y']
        model = sm.OLS(y,X)
        results = model.fit()
        m = results.params['X']
        b = results.params['const']
        f_prob = results.f_pvalue 
        r2 = results.rsquared
        r2_vals.append([data1_name, data2_name, r2, f_prob, idx])
        
        # (2) residual errors
        y_pred = results.predict(X)
        residuals = y - y_pred

        plot_df = pd.DataFrame({'x': data1, 'y_line': m * data1 + b})
        plot_df_sorted = plot_df.sort_values(by='x')
       
        # Plot the regression line and scatter plot
        if showplot == True:
            if x_labels is None:
                x_labels = [data1_name]
                y_label = data2_name
            try:
                axes[i].plot(plot_df_sorted['x'], plot_df_sorted['y_line'], color='black', linestyle='dashed', label=f'Regression Line (y = {m:.2f}x + {b:.2f})')
                axes[i].scatter(data1, data2, color=scatter_colors[i], s=70)
                axes[i].set_xlabel(x_labels[i], fontsize=18)
                axes[i].set_ylabel(y_label, fontsize=18)
                axes[i].tick_params(axis='both', which='major', labelsize=16)
                axes[i].spines[['right', 'top']].set_visible(False)
                axes[i].set_title(f'r = {r:.2f}, p={p:.2f}\n r2 = {r2:.2f}, f_pval={f_prob:.2f}', fontsize=16)
            except TypeError:
                plot_regression(axes, fig, data1, data2, m, b, r, p, r2, f_prob, x_labels, y_label, suptitle1, scatter_color=scatter_colors[i])

        # Plot residual error plots
        if resplot == True:
            try:
                plot_residual(axes2[i], fig2, y_pred, residuals, suptitle2)
            except TypeError:
                plot_residual(axes2, fig2, y_pred, residuals, suptitle2)

    if save_fig == True:
        fig.savefig(title1, bbox_inches='tight', dpi=600)
        
    if savevals == True:
        return r2_vals
    
    
def plot_regression(ax, fig, data1, data2, m, b, r, p, r2, f_prob, data1_name, data2_name, suptitle, scatter_color='lightskyblue'):
    plot_df = pd.DataFrame({'x': data1, 'y_line': m * data1 + b})
    plot_df_sorted = plot_df.sort_values(by='x')

    # Plot the regression line and scatter plot
    ax.plot(plot_df_sorted['x'], plot_df_sorted['y_line'], color='black', linestyle='dashed', label=f'Regression Line (y = {m:.2f}x + {b:.2f})')
    ax.scatter(data1, data2, color=scatter_color, s=70)
    ax.set_xlabel(data1_name, fontsize=18)
    ax.set_ylabel(data2_name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f'r = {r:.2f}, p={p:.2f}\n r2 = {r2:.2f}, f_pval={f_prob:.2f}', fontsize=16)
    fig.suptitle(f'{suptitle}', fontsize=12)

def plot_residual(ax, fig, y_pred, residuals, suptitle):
    # Plot residual error plots
    ax.scatter(y_pred, residuals, color='dodgerblue', edgecolor='k')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Fitted values (Y_pred)', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle(f'{suptitle}', fontsize=14)

def linear_regression(df):
    X = sm.add_constant(df['X'])
    y = df['Y']
    model = sm.OLS(y,X)
    results = model.fit()
    m = results.params['X']
    b = results.params['const']
    f_value = results.fvalue
    f_prob = results.f_pvalue 
    r2 = results.rsquared
    return m, b, f_value, f_prob, r2

def compare_r2_dot(LH_mean_df, idx, pairs, LH_diff_df=None, show_fig=True, save_fig=False, savevals=False):
    "Function for plotting R2 and dot product on the same plot."
    N = [0,1,2,3,4]
    for pair in pairs:
        E2_name = pair[0]
        AEE_name = pair[1]

        if 'd[' in E2_name:
            dfx = LH_diff_df 
            dfx['day_from_LH'] = LH_mean_df['day_from_LH'].unique()[:-1]
        else:
            dfx = LH_mean_df

        # get E2 window
        window_E2 = np.array(dfx[E2_name].iloc[idx[0]:idx[1]])
        
        r2_scores, pdt_vals = [], []
        for n in N:
            # get AEE signal
            signal_AEE = np.array(dfx[AEE_name].iloc[idx[0]+n:idx[1]+n])
            vals = {'X':signal_AEE, 'Y':window_E2}
            df = pd.DataFrame(vals)
            
            # linear regression
            m, b, f_value, f_prob, r2 = linear_regression(df)
            
            # get dot product
            pdt = np.dot(window_E2, signal_AEE)
        
            # save vals
            r2_scores.append((n, r2))
            pdt_vals.append((n, pdt))
            
        # plot r2 scores
        if show_fig == True:
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            sns.lineplot(x=[tup[0] for tup in r2_scores], y=[tup[1] for tup in r2_scores], color="black",linewidth=2,label=r'$r^2$', legend=False, marker='o', ax=ax)
            tick_locations = [0, 1, 2, 3, 4]
            tick_labels = [r'$p_0$', r'$p_0 + 1$', r'$p_0 + 2$', r'$p_0 + 3$', r'$p_0 + 4$']
            ax.set_xticks(tick_locations, tick_labels) 

            ax.tick_params(axis='both', labelsize=18) 
            ax.set_xlabel(f'AEE shift from p0', fontsize=20)
            ax.set_ylabel(r'$r^2$ between AEE and Progesterone', fontsize=20)
            ax.set_ylim(-0.01,0.7)
            ax.spines[['right','top']].set_visible(False)
            ax.grid(False)
            
            # plot dot product
            ax2 = ax.twinx()
            sns.lineplot(x=[tup[0] for tup in pdt_vals], y=[tup[1] for tup in pdt_vals], color='grey',linewidth=2, label='dot product',marker='o', ax=ax2)
            ax2.set_xticks([tup[0] for tup in pdt_vals])
            ax2.tick_params(axis='both', labelsize=18) 
            ax2.set_xlabel(f'shift from optimal {E2_name} days', fontsize=20)
            ax2.set_ylabel(r"AEE ($kJ \cdot kg^{-1}$) $\cdot$ Progesterone ($nmol \cdot L^{-1}$)", fontsize=20, color="grey")
            ax2.set_ylim(min([tup[1] for tup in pdt_vals])*0.97, max([tup[1] for tup in pdt_vals])*1.05)
            ax2.spines[['top']].set_visible(False)
            ax2.spines[['right']].set_color("darkgrey")
            ax2.tick_params(axis='y', colors="grey")
            ax2.grid(False)
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc='upper right',frameon=False, fontsize=16)

            if save_fig == True:
                plt.savefig(f'./figures/r2_dot_{AEE_name}_{E2_name}.svg', transparent=True)

        if savevals == True:
            return r2_scores, pdt_vals
        
def ci_95_from_99_t(mean, ci99_upper, n):
    df = n - 1
    alpha_99 = 0.01
    alpha_95 = 0.05

    # Get two-tailed t critical values for 99% and 95% CI
    t_99 = t.ppf(1 - alpha_99/2, df)
    t_95 = t.ppf(1 - alpha_95/2, df)
    
    # Get standard error from 99% CI
    w_99 = ci99_upper - mean  
    se = w_99 / t_99

    # Use standard error and t to get 95% CI
    w_95 = t_95 * se
    ci95_lower = mean - w_95
    ci95_upper = mean + w_95
    return (ci95_lower, ci95_upper)