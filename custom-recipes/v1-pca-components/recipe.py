# Code for custom code recipe v1-pca-components (imported from a Python recipe)

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

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_A_names = get_input_names_for_role('input_dataset')
print('NNNNN input_A_names')
# The dataset objects themselves can then be created like this:
input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]
print(" SSSSS input_A_datasets")

# For outputs, the process is the same:
output_A_names = get_output_names_for_role('out_dataset')
output_A_datasets = [dataiku.Dataset(name) for name in output_A_names]


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
my_variable = get_recipe_config()['parameter_name']

# For optional parameters, you should provide a default value in case the parameter is not present:
my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
avocado_prepared_v4_prepared = dataiku.Dataset("avocado_prepared_v4_prepared")
avocado_prepared_v4_prepared_df = avocado_prepared_v4_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
plugin_df = avocado_prepared_v4_prepared_df # For this sample code, simply copy input to output
plugin_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_floats = plugin_df.select_dtypes(np.float64)
print(df_floats)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# scaling=StandardScaler()
 
# # Use fit and transform method 
# scaling.fit(df_floats)
# Scaled_data=scaling.transform(df_floats)
df_normalized=(df_floats - df_floats.mean()) / df_floats.std()
# # Set the n_components=3
# principal=PCA(n_components=3)
# principal.fit(df_normalized)
# x=principal.transform(df_normalized)
pca = PCA(n_components=3)
x = pca.fit(df_normalized)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pca.components_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
eigen_vectors = pd.DataFrame(pca.components_.T,columns=PCnames,index=df_floats.columns)
print(eigen_vectors)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
print(PCnames)
eigen_variance = pd.DataFrame(pca.explained_variance_,columns=["Variance"],index=PCnames)
# eigen_variance.columns = 'Variance'
eigen_variance

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
eigen_variance_ratio = pd.DataFrame(pca.explained_variance_ratio_,columns=["Variance_ratio"],index=PCnames)
# eigen_variance.columns = 'Variance'
eigen_variance_ratio

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
frame_combined = pd.concat([eigen_variance, eigen_variance_ratio],axis=1)
frame_combined

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
plugin_df = avocado_prepared_v4_prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
plugin = dataiku.Dataset("plugin")
plugin.write_with_schema(plugin_df)