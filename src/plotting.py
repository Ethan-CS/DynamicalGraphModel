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


def plot(data):
    # For testing - summarise the pandas dataframe
    print(data)

    # Define title and labels for axes
    title = 'Time taken to generate and solve equations\nfor $SIR$ models on various size paths.'

    # Send to scatter plot function to compare performance
    scatter_compare_two_methods(data, x_filter='time (full)', y_filter='time (closed)', title=title,
                                x_label='Time for full system', y_label='Time for closed system').savefig('time.png')


# Used to produce a scatter plot comparing the time performance of two methods that achieve the same result
def scatter_compare_two_methods(data, x_filter, y_filter, title='', x_label='', y_label=''):
    g = sns.scatterplot(data=data, x=data[x_filter], y=data[y_filter], hue=data['winner'])
    g.legend().set_title('Fastest time')

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
    plt.legend(loc='lower right')

    return g.get_figure()


def main():
    # Read in from the CSV (just for testing, will come straight from dataframe in future)
    data = pd.read_csv('/data/PathData.csv')
    time_winners = []
    for index, row in data.iterrows():
        if row['time (full)'] < row['time (closed)']:
            time_winners.append('full')
        else:
            time_winners.append('closed')
    data['winner'] = time_winners
    plot(data)


if __name__ == "__main__":
    main()
