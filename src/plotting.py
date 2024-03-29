import math as maths

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# This file was used to produce the plots in the associated publication (see readme) and has been left for transparency
# and so that similar plots can be reproduced when using this project, if desired.

# Set seaborn as default and set resolution and style defaults
sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

# Avoids an annoying error on macOS
matplotlib.use('TkAgg')


# Used to produce a scatter plot comparing the time performance of two methods that achieve the same result
def scatter_compare_two_methods(data, x_filter, y_filter, title='', x_label='', y_label='', hue=None, max_val=60,
                                timeout=None):
    # df1['A'] = df1['A'].apply(lambda x: [y if y <= 9 else 11 for y in x])
    if timeout is not None:
        plt.axvline(x=timeout, color='r', linestyle='dashed', label='Timeout')
        # plt.axhline(y=timeout, color='r', linestyle='dashed')
        data[x_filter] = data[x_filter].apply(lambda x: x if x <= timeout else timeout)
        data[y_filter] = data[y_filter].apply(lambda x: x if x <= timeout else timeout)

    g = sns.scatterplot(data=data, x=data[x_filter], y=data[y_filter],
                        hue=data['winner'] if hue is None else hue)

    if max_val == 0:
        max_val_for_axes(data, x_filter, y_filter)

    plot_style(max_val, title, x_label, y_label)

    return g.get_figure()


def max_val_for_axes(data, x_filter, y_filter):
    max_x = data[x_filter].max()
    max_y = data[y_filter].max()
    max_val = max_x if max_x > max_y else max_y
    return maths.floor(max_val)


def plot_style(max_val, title, x_label, y_label):
    # House style settings (set grid, add axis labels, make axes correct size, arrange legend)
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend().set_visible(True)

    # plt.xlim([0, max_val])  # Axes should be same length for easier visual comparison
    # plt.ylim([0, max_val])
    # plt.legend()


def main():
    data = pd.read_csv('/Users/ethanhunter/PycharmProjects/DynamicalGraphModel/src/data/path_equations_data.csv')
    print(data)

    plt.axhline(y=150, color='r', linestyle='dashed', label='Timeout')
    data[f'time (full)'] = data[f'time (full)'].apply(lambda x: x if x <= 150 else 150)
    data[f'time (closed)'] = data[f'time (closed)'].apply(lambda x: x if x <= 150 else 150)

    # Define title and labels for axes
    title = f'Time to generate and solve equations for $SIR$ models\n on paths ' \
            f'up to 25 vertices.'
    g = sns.scatterplot(x=data['num of vertices'], y=data[f'time (full)'], label='Full', alpha=0.4, s=70)
    sns.scatterplot(x=data['num of vertices'], y=data[f'time (closed)'], label='Closed', alpha=0.4, s=70)
    plot_style(0, title, 'Number of vertices', f'Time to generate and solve equations')
    g.get_figure().savefig(f'data/plots/cycle_time.png')
    # plot_er_full_closures(60)


def plot_averages(cycle_data, path_data, tree_data):
    averages = pd.DataFrame()
    max_vals = []
    for d in [(path_data, 'path'), (cycle_data, 'cycle'), (tree_data, 'tree')]:
        x_filter = f'{d[1]} full average'
        y_filter = f'{d[1]} closed average'
        data = d[0]
        full_results, closed_results = {}, {}
        full_averages, closed_averages = [], []
        for index, row in data.iterrows():
            if row['num of vertices'] not in full_results.keys():
                full_results[int(row['num of vertices'])] = []
            if row['num of vertices'] not in closed_results.keys():
                closed_results[int(row['num of vertices'])] = []
            full_results[row['num of vertices']].append(row['time (full)'])
            closed_results[row['num of vertices']].append(row['time (closed)'])
        for entry in full_results:
            avg = sum(full_results[entry]) / len(full_results[entry])
            full_averages.append(avg)
        for entry in closed_results:
            avg = sum(closed_results[entry]) / len(closed_results[entry])
            closed_averages.append(avg)
        if 'num vertices' not in averages.keys():
            averages['num vertices'] = full_results.keys()
        averages[x_filter] = full_averages
        averages[y_filter] = closed_averages
        winners = []
        for index, row in averages.iterrows():
            if row[x_filter] <= row[y_filter]:
                winners.append('full')
            else:
                winners.append('closed')
        averages[f'{d[1]} winner'] = winners
        sns.scatterplot(averages, x=x_filter, y=y_filter, hue=averages[f'{d[1]} winner'])
        max_vals.append(max_val_for_axes(averages, x_filter, y_filter))
    plot_style(max(max_vals), "Averages", "Full", "Closed")
    plt.savefig('avg.png')
    print(averages)


def plot_er_full_closures(timeout=None):
    # Read in from the CSV (just for testing, will come straight from dataframe in future)
    print(f'READING: data/erdos_renyi_equations_data.csv')
    data = pd.read_csv(f'data/erdos_renyi_equations_data.csv')
    print(data)
    time_winners = []
    for index, row in data.iterrows():
        if row['time (full)'] <= row['time (closed)']:
            time_winners.append('full')
        else:
            time_winners.append('closed')
    data['winner'] = time_winners

    # Define title and labels for axes
    title = f'Time to generate and solve equations for $SIR$ models\n on Erdős–Rényi graphs ' \
            f'on 25 vertices up to $p=0.2$.'

    # Send to scatter plot function to compare performance
    if timeout is not None:
        # plt.axvline(x=timeout, color='r', linestyle='dashed', label='Timeout')
        plt.axhline(y=timeout, color='r', linestyle='dashed', label='Timeout')
        data['time (full)'] = data['time (full)'].apply(lambda x: x if x <= timeout else timeout)
        data['time (closed)'] = data['time (closed)'].apply(lambda x: x if x <= timeout else timeout)

    g = sns.scatterplot(x=data['probability'], y=data['time (closed)'], hue=data['average degree'])
    plot_style(0, title, 'Probability', 'Time to generate and solve full system')

    g.get_figure().savefig(f'data/plots/{g}_full_time.png')


def plot_full_vs_closures(graphs: list, timeout=None):
    for g in graphs:
        # Read in from the CSV (just for testing, will come straight from dataframe in future)
        print(f'READING: data/{g}_equations_data.csv')
        data = pd.read_csv(f'data/{g}_equations_data.csv')
        print(data)
        time_winners = []
        for index, row in data.iterrows():
            if row['time (full)'] <= row['time (closed)']:
                time_winners.append('full')
            else:
                time_winners.append('closed')
        data['winner'] = time_winners

        # Define title and labels for axes
        title = f'Time to generate and solve equations for $SIR$ models\n on {"random " + g + "s" if g == "tree" else g + "s"} ' \
                f'up to {data.iloc[-1]["num of vertices"]} vertices.'

        # Send to scatter plot function to compare performance
        scatter_compare_two_methods(data, 'time (full)', 'time (closed)', title, 'Time - full system (s)',
                                    'Time - closed system (s)', time_winners, max_val=20, timeout=timeout)\
            .savefig(f'data/plots/{g}_time.png')



def plot_eq_vs_mc(graphs: list):
    for g in graphs:
        # Read in from the CSV (just for testing, will come straight from dataframe in future)
        data_mc = pd.read_csv(f'data/{g}_mc_same_v_data.csv')
        data_eq = pd.read_csv(f'data/{g}_equations_same_v_data.csv')

        num_eq, num_mc = len(data_eq.index), len(data_mc.index)
        smallest = min(num_eq, num_mc)

        data = pd.concat([data_eq.rename(columns={'time to solve': 'time (eq)'})['time (eq)'].iloc[:smallest],
                         data_mc.rename(columns={'time to solve': 'time (mc)'})['time (mc)'].iloc[:smallest],
                         data_mc['p'].iloc[:smallest]], axis=1)

        time_winners = []
        for index, row_eq in data_eq.iterrows():
            row_mc = data_mc.iloc[index]
            if row_eq['time to solve'] <= row_mc['time to solve']:
                time_winners.append('Equations')
            else:
                time_winners.append('Monte Carlo')
        data['winner'] = time_winners

        # Define title and labels for axes
        title = f'Time to generate and solve equations for $SIR$ models\n on random graphs '\
                f'on {int(data_mc.iloc[-1]["num of vertices"])} vertices.'
        data = data.rename({'p': 'Probability'}, axis=1)
        # Send to scatter plot function to compare performance
        scatter_compare_two_methods(data, 'time (eq)', 'time (mc)', title, 'Time for equations', 'Time for Monte Carlo',
                                    data['Probability'], timeout=60).savefig(f'data/plots/{g}_time.png')


if __name__ == "__main__":
    main()


def plot_mc_averages(mc_averages, soln):
    mc_averages.columns = [f'$I_{i}$' for i in range(len(mc_averages.columns))]
    mc_averages.plot()
    count = 0
    for i in soln:
        plt.axhline(y=i, linestyle='--', label=f'True $I_{count}$')
        count += 1
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
