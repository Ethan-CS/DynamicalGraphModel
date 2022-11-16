import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set seaborn as default and set resolution and style defaults
sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

# Avoids an annoying error on macOS
matplotlib.use('TkAgg')


# Used to produce a scatter plot comparing the time performance of two methods that achieve the same result
def scatter_compare_two_methods(data, x_filter, y_filter, title='', x_label='', y_label=''):
    g = sns.scatterplot(data=data, x=data[x_filter], y=data[y_filter], hue=data['winner'])

    plot_style(max_val_for_axes(data, x_filter, y_filter), title, x_label, y_label)

    return g.get_figure()


def max_val_for_axes(data, x_filter, y_filter):
    max_x = data[x_filter].max()
    max_y = data[y_filter].max()
    max_val = max_x if max_x > max_y else max_y
    return max_val


def plot_style(max_val, title, x_label, y_label):
    # House style settings (set grid, add axis labels, make axes correct size, arrange legend)
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xlim([0, max_val])  # Axes should be same length for easier visual comparison
    # plt.ylim([0, max_val])
    plt.legend(title='Fastest Time', loc='lower right')


def main():
    plot_averages(pd.read_csv(f'data/path_data.csv'), pd.read_csv(f'data/cycle_data.csv'), pd.read_csv(f'data/tree_data.csv'))
    plot_full_vs_closures(['tree'])


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
            if row['number of vertices'] not in full_results.keys():
                full_results[int(row['number of vertices'])] = []
            if row['number of vertices'] not in closed_results.keys():
                closed_results[int(row['number of vertices'])] = []
            full_results[row['number of vertices']].append(row['time (full)'])
            closed_results[row['number of vertices']].append(row['time (closed)'])
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


def plot_full_vs_closures(graphs: list):
    for g in graphs:
        # Read in from the CSV (just for testing, will come straight from dataframe in future)
        data = pd.read_csv(f'data/{g}_data.csv')
        time_winners = []
        for index, row in data.iterrows():
            if row['time (full)'] <= row['time (closed)']:
                time_winners.append('full')
            else:
                time_winners.append('closed')
        data['winner'] = time_winners
        # For testing - summarise the pandas dataframe
        print(data)

        # Define title and labels for axes
        title = f'Time to generate equations for $SIR$ models\n on {"random " + g + "s" if g == "tree" else g + "s"} ' \
                f'up to {data.iloc[-1]["number of vertices"]} vertices.'

        # Send to scatter plot function to compare performance
        scatter_compare_two_methods(data, x_filter='time (full)', y_filter='time (closed)', title=title,
                                    x_label='Time for full system', y_label='Time for closed system'
                                    ).savefig(f'data/plots/{g}_time.png')


if __name__ == "__main__":
    main()
