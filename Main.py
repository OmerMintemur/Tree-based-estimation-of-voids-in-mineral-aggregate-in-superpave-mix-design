import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.inspection import permutation_importance
import xgboost
import lightgbm
from sklearn.metrics import mean_squared_error
from Draw_Importance import draw
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("data.xlsx")
# Get Input and Output
features = ["Pb", "Gmm", "Gmb", "Gb", "Gsb", "Pba", "Pbe", "Gse"]
output = ["VMA"]
saved_feature_importances= {"Pb":0,"Gmm":0,"Gmb":0,"Gb":0,"Gsb":0,"Pba":0,"Pbe":0,"Gse":0}
data_for_feature_importance = data[features] # To see the features' importance
input = data[features].to_numpy()
output = data[output].to_numpy()

average_mse_on_train = []
average_mse_on_test = []
average_rmse_on_train = []
average_rmse_on_test = []
average_feature_importance = []
per_fold_importances = []

KFOLD = 20
model = xgboost.XGBRegressor()
#model = lightgbm.LGBMRegressor()
model_name = "XGBOOST"



# Kfold
kf = KFold(n_splits=KFOLD,shuffle=True,random_state=42)
for train_index, val_index in kf.split(input):
    # Get splitted data
    X_train, X_val = input[train_index], input[val_index]
    Y_train, Y_val = output[train_index], output[val_index]

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Prepare the model
    model.fit(X_train,Y_train)

    # Save the models errors

    # MSE
    average_mse_on_train.append(mean_squared_error(model.predict(X_train),Y_train))
    average_mse_on_test.append(mean_squared_error(model.predict(X_val), Y_val))

    #RMSE
    average_rmse_on_train.append(np.sqrt(mean_squared_error(model.predict(X_train),Y_train)))
    average_rmse_on_test.append(np.sqrt(mean_squared_error(model.predict(X_val), Y_val)))


    # Model feature importance
    fimportance = permutation_importance(model, X_val, Y_val, n_repeats=30, random_state=0)
    indexes = fimportance.importances_mean.argsort()[::-1]
    # print(data_for_feature_importance.columns[indexes])

    for i in indexes:
        saved_feature_importances[data_for_feature_importance.columns[i]] += fimportance.importances_mean[i]


    fold_importance = {}
    for i in indexes:
        feature_name = data_for_feature_importance.columns[i]
        importance_value = fimportance.importances_mean[i]
        fold_importance[feature_name] = importance_value
        saved_feature_importances[feature_name] += importance_value

    per_fold_importances.append(fold_importance)

print(f"{model_name} - {KFOLD}")
print(f"Average MSE on train {np.average(average_mse_on_train)}")
print(f"Average MSE on test {np.average(average_mse_on_test)}")
print(f"Average RMSE on train {np.average(average_rmse_on_train)}")
print(f"Average RMSE on test {np.average(average_rmse_on_test)}")
# print(saved_feature_importances)
draw(saved_feature_importances,KFOLD,model_name)

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Prepare data
importance_df = pd.DataFrame(per_fold_importances).fillna(0)
importance_df = importance_df.div(importance_df.sum(axis=1), axis=0)
features = importance_df.columns.tolist()

# Transpose for plotting
importance_df_T = importance_df.T

# Color map setup
cmap = cm.get_cmap('viridis', KFOLD)
norm = mcolors.Normalize(vmin=0, vmax=KFOLD - 1)
colors = [cmap(i) for i in range(KFOLD)]

# Plot settings
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.8 / KFOLD  # width per bar per fold
x = np.arange(len(features))  # x locations for features

# Plot each fold's bars
for i in range(KFOLD):
    ax.bar(
        x + i * bar_width,
        importance_df_T.iloc[:, i],
        width=bar_width,
        color=colors[i],
        edgecolor='none',
        label=f'Fold {i+1}' # Optional if not using legend
    )

# Axes labels and title
ax.set_title(f"Normalized Feature Importance per Fold ({model_name})", fontsize=32, fontweight='bold', pad=20)
ax.set_ylabel("Normalized Permutation Importance", fontsize=28)
ax.set_xlabel("Feature", fontsize=28)
ax.set_xticks(x + bar_width * (KFOLD / 2 - 0.5))
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=26)
'''
# Colorbar as legend substitute
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Fold Number', fontsize=14)

# Set integer ticks (e.g., 1 through KFOLD)
tick_values = np.arange(0, KFOLD)
cbar.set_ticks(tick_values)
cbar.set_ticklabels([str(i + 1) for i in tick_values])  # Fold numbers start from 1
cbar.ax.tick_params(labelsize=14)
'''

# Grid and beautify
sns.despine()
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
ax.tick_params(axis='y', labelsize=24)

plt.tight_layout()
plt.savefig(f"{model_name}_{KFOLD}_per_fold.png", format='png',bbox_inches='tight', dpi=600)
plt.show()



