# -*- coding: utf-8 -*-
import os
global basep_path, output_path,data_path, fig_path, model_path, elements
test_count = str("CLR_all_element")
base_path = "../"
output_path = "../result/"
data_path = "../data/"
fig_path = "../fig/"
model_path = "../result/models/"

# 9 ELEMENTS selected
elements_brev = ['P', 'Y', 'Ce', 'Sm', 'Eu', 'Dy', 'Lu', 'Th', 'U']
elements = ['P (μmol/g)', 'Y (μmol/g)', 'Ce (μmol/g)', 'Sm (μmol/g)', 'Eu (μmol/g)', 'Dy (μmol/g)', 'Lu (μmol/g)', 'Time-corrected Th (μmol/g)', 'Time-corrected U (μmol/g)']
#elements = ['P (μmol/g)', 'Ce (μmol/g)', 'Sm (μmol/g)', 'Eu (μmol/g)', 'Dy (μmol/g)', 'Lu (μmol/g)', 'Time-corrected Th (μmol/g)', 'Time-corrected U (μmol/g)']
#elements = ['P (μmol/g)', 'Ce (μmol/g)', 'Sm (μmol/g)', 'Eu (μmol/g)', 'Lu (μmol/g)', 'Time-corrected Th (μmol/g)', 'Time-corrected U (μmol/g)']
#elements = ['P (μmol/g)', 'Y (μmol/g)', 'Ce (μmol/g)', 'Sm (μmol/g)', 'Eu (μmol/g)', 'Dy (μmol/g)', 'Lu (μmol/g)', 'Th (μmol/g)', 'U (μmol/g)']
if not os.path.exists(model_path ):
    os.makedirs(model_path )
model_path = model_path

if not os.path.exists(output_path):
    os.makedirs(output_path)
output_path = output_path

if not os.path.exists(fig_path):
    os.makedirs(fig_path)
fig_path = fig_path


info_list = ["Reference", "Set", "Zircon", "Note", "Age(Ma)"]
file_name = "Database 2"

