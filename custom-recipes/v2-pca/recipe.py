# Code for custom code recipe v2-pca (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.


input_dataset_name = get_input_names_for_role('input')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
dataset_pca_df = input_dataset.get_dataframe()

# For outputs, the process is the same:
output_eigen_vectors = get_output_names_for_role('output eigen vectors')
output_eigen_vectors_datasets = [dataiku.Dataset(name) for name in output_eigen_vectors]

# For outputs, the process is the same:
output_variance_names = get_output_names_for_role('output eigen variance')
output_variance_datasets = [dataiku.Dataset(name) for name in output_variance_names]


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
n_components = get_recipe_config()['number of components']


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_floats = dataset_pca_df.select_dtypes(np.float64)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_normalized=(df_floats - df_floats.mean()) / df_floats.std()
pca = PCA(n_components=n_components)
x = pca.fit(df_normalized)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
eigen_vectors = pd.DataFrame(pca.components_.T,columns=PCnames)
# eigen_vectors['Columns'] = df_floats.columns
# print(eigen_vectors)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
eigen_variance = pd.DataFrame(pca.explained_variance_,columns=["Variance"],index=PCnames)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
eigen_variance_ratio = pd.DataFrame(pca.explained_variance_ratio_,columns=["Variance_ratio"],index=PCnames)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
frame_combined = pd.concat([eigen_variance, eigen_variance_ratio],axis=1)
frame_combined['PCA Components'] = PCnames

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Write recipe outputs
output_eigen_vectors_final = get_output_names_for_role('output_eigen_vectors')
print(" SSSSS output_eigen_vectors_final ")
output_dataset_final_vectors = dataiku.Dataset(output_eigen_vectors_final)
output_dataset_final_vectors.write_with_schema(eigen_vectors)

# Write recipe outputs
eigen_variance_ratio_final = get_output_names_for_role('output_eigen_variance')
output_dataset_variance_final = dataiku.Dataset(eigen_variance_ratio_final)
output_dataset_variance_final.write_with_schema(frame_combined)
