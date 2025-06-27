import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sns.set_theme()
add = 12
def draw(dict_,fold,model):

    # Normalize the importance values
    total_sum = sum(dict_.values())
    normalized_importance = {key: value / total_sum for key, value in dict_.items()}


    # Sort the features by importance in descending order
    sorted_features = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features)

    # Normalize values for colormap
    norm = Normalize(vmin=min(importance), vmax=max(importance))
    cmap = plt.cm.winter
    colors = cmap(norm(importance))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(features, importance, color=colors)


    # Add values at the end of the bars
    for bar, value in zip(bars, importance):
        plt.text(
            bar.get_width() + 0.001,  # Position slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Vertically center the text
            f'{value:.2f}',  # Format to two decimal places
            va='center',
            fontsize=12+add
        )




    plt.xlabel('Importance Scores', fontsize=16+add)
    if fold==5:
        plt.ylabel('Features', fontsize=16+add)
    plt.title(f'Feature Importance - VMA ({model.upper()})', fontsize=16+add, fontweight="bold")
    plt.yticks(fontsize=16+add)
    plt.xticks(fontsize=16+add)
    plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for colorbar to show
    # cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    # cbar.set_label('Normalized Importance', fontsize=14)
    # cbar.ax.tick_params(labelsize=12+add)
    if fold==1:
        plt.text(0.3, 7.5, f'80% - 20% Split',size=14+add,bbox=dict(boxstyle='round,pad=0.3', fc="#7b93eb"),color="white")
    else:
        plt.text(0.3, 7.5, f'{fold}-fold cross validation',size=14+add,bbox=dict(boxstyle='round,pad=0.3', fc="#7b93eb"),color="white")
    plt.savefig(f"{model}_{fold}.png",bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()