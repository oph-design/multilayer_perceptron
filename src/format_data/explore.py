import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

colnames = [
    "ID",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_SE",
    "texture_SE",
    "perimeter_SE",
    "area_SE",
    "smoothness_SE",
    "compactness_SE",
    "concavity_SE",
    "concave_points_SE",
    "symmetry_SE",
    "fractal_dimension_SE",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

hue_colors = {
    "M": "red",
    "B": "blue",
}


def main() -> None:
    """main function"""
    data = pd.read_csv("data.csv", names=colnames, header=None)
    print(data.describe())
    selected_columns = [data.columns[1]] + list(data.columns[3:12])
    if len(sys.argv) > 1 and sys.argv[1] == "SE":
        selected_columns = [data.columns[1]] + list(data.columns[13:22])
    elif len(sys.argv) > 1 and sys.argv[1] == "worst":
        selected_columns = [data.columns[1]] + list(data.columns[23:32])
    data = pd.DataFrame(data[selected_columns])
    plt.rcParams.update({"font.size": 8})
    pair = sns.pairplot(
        data,
        hue="diagnosis",
        palette=hue_colors,
        plot_kws=dict(linewidth=0.1),
        corner=True,
    )
    pair.figure.set_size_inches(15, 13)
    pair.figure.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
