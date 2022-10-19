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

    max_x = data[x_filter].max()
    max_y = data[y_filter].max()
    max_val = max_x if max_x > max_y else max_y

    # House style settings (set grid, add axis labels, make axes correct size, arrange legend)
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([0, max_val])  # Axes should be same length for easier visual comparison
    plt.ylim([0, max_val])
    plt.legend(title='Fastest Time', loc='lower right')

    return g.get_figure()


def main():
    # Get the results
    path_data = pd.read_csv(f'data/path_data.csv')
    cycle_data = pd.read_csv(f'data/cycle_data.csv')
    tree_data = pd.read_csv(f'data/tree_data.csv')
    for data in [path_data, cycle_data, tree_data]:
        full_results = {}
        closed_results = {}
        full_averages = []
        for index, row in data.iterrows():
            full_results[row['number of vertices']].append(row['time (full)'])
            closed_results[row['number of vertices']].append(row['time (full)'])
        for entry in full_results:
            full_averages.append()

        data['full average'] = full_averages
        print(data)

    # plot_full_vs_closures(['path', 'tree', 'cycle'])


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
        title = f'Time to generate equations for $SIR$ models\n' \
                f'on {"random " + g + "s" if g == "tree" else g + "s"} up to 10 vertices.'

        # Send to scatter plot function to compare performance
        scatter_compare_two_methods(data, x_filter='time (full)', y_filter='time (closed)', title=title,
                                    x_label='Time for full system', y_label='Time for closed system'
                                    ).savefig(f'data/plots/{g}_time.png')


if __name__ == "__main__":
    main()
