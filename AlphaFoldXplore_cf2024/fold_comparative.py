from AlphaFoldXplore_cf2024 import alphafoldxplore as afx
from AlphaFoldXplore_cf2024 import prediction_results

from zipfile import ZipFile
import os
import re
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
#from google.colab import files
import pandas as pd
import numpy as np
import math
from Bio.PDB import PDBParser
#---------------------------------Funciones FoldXplorer-------------------------


############### Funciones para cargar archivos multiples [af3].zip --> [af3].afxt :
import os
import shutil
import json
from zipfile import ZipFile
from Bio.PDB import MMCIFParser, PDBIO
from prediction_results import prediction_results
#from google.colab import files  # Solo si usas Google Colab

# Función para manejar la carga del archivo en Google Colab
def upload_zip_colab():
    uploaded = files.upload()
    input_directory = 'input_af3'
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

    for filename in uploaded.keys():
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded[filename])

    # Devuelve la ruta del archivo cargado
    return file_path

# Función principal que integra la carga de archivo y el procesamiento
def load_af3_interactive_colab():
    # Solicitar al usuario que cargue el archivo .zip desde la computadora
    print("Por favor, carga tu archivo .zip desde tu computadora local:")
    location = upload_zip_colab()  # Usamos la función de carga

    # Solicitar al usuario que ingrese un nombre para la estructura
    name = input("Por favor, ingrese un nombre para la estructura: ")

    # Verificar si el archivo es un .zip válido
    if not location.endswith('.zip'):
        print("Error: El archivo proporcionado no es un archivo .zip.")
        return

    # Llamar a la función original load_af3 con los parámetros proporcionados
    Z = load_af3(name, location)

    return Z

# Función load_af3 (tu función original)
def load_af3(name, location):
    Z = {}

    if os.path.exists(name):
        print(f"Error: la carpeta con el nombre '{name}' ya existe en esta ubicación. Por favor, elige otro nombre o elimina/renombra la carpeta.")
        return

    folder = os.path.basename(location[:-4])
    extract_folder = f'AF3_files/{folder}'
    results_folder = f'AF3_files/{folder}_fx'
    os.makedirs(results_folder, exist_ok=True)

    with ZipFile(location, 'r') as fz:
        fz.extractall(extract_folder)  # Extraer el archivo .zip de AF3

    # Procesar los archivos extraídos
    for path in os.listdir(extract_folder):
        long_path = os.path.join(extract_folder, path)
        if long_path.endswith("_summary_confidences_0.json"):  # Obtener pTMscore
            with open(long_path, 'r') as file:
                data = json.load(file)
                ptmscore = float(data['ptm'])

        if long_path.endswith("_model_0.cif"):  # Convertir el CIF en PDB
            file_parser = MMCIFParser()
            structure = file_parser.get_structure("base", long_path)
            io = PDBIO()
            io.set_structure(structure)
            io.save(f'{results_folder}/{name}_relaxed.pdb')  # Guardar el archivo PDB

        if long_path.endswith("_full_data_0.json"):  # Obtener PAE
            with open(long_path, 'r') as file:
                data = json.load(file)
                distance = {"distance": data['pae']}
            with open(f'{results_folder}/{name}_pae.json', 'w', encoding='utf-8') as f:
                json.dump(distance, f, ensure_ascii=False)

    # Crear el reporte en la carpeta con el nombre proporcionado
    directory = f'{name}/{name}.zip'
    os.makedirs(f'{name}', exist_ok=True)
    with open(f'{name}/{name}_report.txt', 'w', encoding='utf-8') as file:
        file.write(name + '\n')
        file.write(directory + '\n')
        file.write("0" + '\n')
        file.write("no info" + '\n')
        file.write("pTMScore=" + str(ptmscore) + '\n')
        file.write("version=af3")

    # Mover y comprimir archivos
    os.system(f"mv '{results_folder}/{name}_pae.json' '{name}/{name}_pae.json'")
    os.system(f"mv '{results_folder}/{name}_relaxed.pdb' '{name}/{name}_relaxed.pdb'")
    os.system(f"zip -FSr '{name}.zip' '{name}'")
    shutil.rmtree(f'{name}')

    # Crear la entrada de predicción y guardarla en el diccionario
    prediction_entry = prediction_results(name, directory, "0", "no info", ptmscore)
    Z['p1'] = prediction_entry

    # Crear el archivo de lista
    os.makedirs(f'{name}', exist_ok=True)
    with open(f'{name}/{name}_list.txt', 'w', encoding='utf-8') as file:
        for result in list(Z.values()):
            file.write(f"{result.directory}\n")

    # Finalizar y mover el archivo .afxt
    os.system(f"mv '{name}.zip' '{name}/'")
    os.system(f"zip -FSr -D '{name}.zip' '{name}'")
    os.system(f"mv '{name}.zip' '{name}.afxt'")

    print(f"Guardado en tu computadora local. Nombre: \"{name}.afxt\"")

    return Z
#************************************FoldComparative************************************


def instances_objects_AF3(directory_path):
    """
    Concatenates predictions from afxt files named af3_1, af3_2, ..., af3_181 into a single dictionary.

    Parameters:
    - directory_path (str): Path to the directory containing afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # List of afxt files in order af3_1.afxt, af3_2.afxt, ..., af3_181.afxt
    afxt_files = [os.path.join(directory_path, f'af3_{index}.afxt') for index in range(1, 30)]

    # Iterate over each file path in the provided list
    for file_path in afxt_files:
        # Check if the file exists before attempting to load
        if os.path.exists(file_path):
            afxt_entry_part = afx.load(file_path)
            # Iterate over predictions in the current part
            for prediction_name, prediction_data in afxt_entry_part.items():
                afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
                prediction_counter += 1  # Increment the global prediction counter
        else:
            print(f"File {file_path} does not exist and will be skipped.")

    return afxt_entry

def instances_objects(file_paths):
    """
    Concatenates predictions from a list of afxt files into a single dictionary.

    Parameters:
    - file_paths (list of str): List of paths to the afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # Iterate over each file path
    for file_path in file_paths:
        afxt_entry_part = afx.load(file_path)
        # Iterate over predictions in the current part
        for prediction_name, prediction_data in afxt_entry_part.items():
            afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
            prediction_counter += 1  # Increment the global prediction counter

    return afxt_entry
#----------------------Funciones SGLT----------------
import os
 # Asumo que fx es el módulo que utilizas para cargar los archivos .afxt

def instances_objects_SGLT(directory_path):
    """
    Concatenates predictions from afxt files in the specified directory into a single dictionary.

    Parameters:
    - directory_path (str): Path to the directory containing afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # List all .afxt files in the directory
    afxt_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.afxt')]

    # Iterate over each file path in the provided list
    for file_path in afxt_files:
        # Check if the file exists before attempting to load
        if os.path.exists(file_path):
            afxt_entry_part = afx.load(file_path)
            # Iterate over predictions in the current part
            for prediction_name, prediction_data in afxt_entry_part.items():
                afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
                prediction_counter += 1  # Increment the global prediction counter
        else:
            print(f"File {file_path} does not exist and will be skipped.")

    return afxt_entry
#***********************************FoldComparative(EXPERIMENTAL VS PREDICT)****************************************************


def visualize_structures(pdb_file1, pdb_file2):
    # Cargar los archivos PDB y visualizarlos con py3Dmol
    with open(pdb_file1, 'r') as file:
        pdb_data1 = file.read()

    with open(pdb_file2, 'r') as file:
        pdb_data2 = file.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data1, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})

    view.addModel(pdb_data2, 'pdb')
    view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})

    view.zoomTo()
    view.show()

import py3Dmol

def visualize_two_structures(pdb_file1, pdb_file2):
    # Leer los datos del archivo PDB experimental
    with open(pdb_file1, 'r') as f:
        pdb_data1 = f.read()

    # Leer los datos del archivo PDB superimpuesto
    with open(pdb_file2, 'r') as f:
        pdb_data2 = f.read()

    # Configurar la visualización usando py3Dmol
    view = py3Dmol.view(width=800, height=600)

    # Añadir la estructura experimental
    view.addModel(pdb_data1, 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}})  # Estilo Cartoon con color azul para la experimental

    # Añadir la estructura superpuesta
    view.addModel(pdb_data2, 'pdb')
    view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})  # Estilo Cartoon con color rojo para la predicción superpuesta

    # Ajustar el zoom para que se vea toda la molécula
    view.zoomTo()

    # Mostrar la visualización en el notebook o en un archivo HTML
    view.show()
#------------------------- PARA VISUALIZAR UMBRALES ---------------------------------------
def get_residues_below_threshold(pdb_file, threshold):
    """
    Extrae los residuos con plDDT menor o igual a un umbral del archivo PDB.
    """
    residues_to_color = []
    
    # Leer el archivo PDB
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                # El plDDT suele estar en el campo B-factor (columna 61-66)
                plDDT = float(line[60:66].strip())
                
                if plDDT <= threshold:
                    # Extraer el ID del residuo (columna 23-26)
                    residue_id = int(line[22:26].strip())
                    
                    # Añadir a la lista si cumple con el umbral
                    if residue_id not in residues_to_color:
                        residues_to_color.append(residue_id)
    
    return residues_to_color



#_________________________ Superimposicion con argumento para threshold_____________**********
from Bio.PDB import PDBParser, Superimposer, PDBIO, Select

    
def superimposed_pdb(pdb_file_exp, pdb_file_pred, guardo_pdb__pred_superimposed, threshold):
    """
    Alinea dos estructuras PDB basadas en átomos CA con pLDDT mayor a 'threshold' y guarda las estructuras alineadas y no alineadas.

    :param pdb_file_exp: Ruta al archivo PDB de la estructura experimental.
    :param pdb_file_pred: Ruta al archivo PDB de la estructura predicha.
    :param output_superimposed_full: Ruta al archivo PDB donde guardo la estructura alineada completa.
    :param output_file_no_aligned: Ruta al archivo PDB donde guardo la estructura con solo los átomos no alineados.
    :param threshold: Umbral de pLDDT para seleccionar residuos =0 (todos los residuos que coinciden).
    :return: Objeto Superimposer con la alineación realizada.
    """
    # Cargar las estructuras desde los archivos PDB
    parser = PDBParser(QUIET=True)
    structure_exp = parser.get_structure('exp', pdb_file_exp)  # Estructura experimental
    structure_pred = parser.get_structure('pred', pdb_file_pred)  # Estructura predicha

    # Obtener los residuos y átomos de CA correspondientes para la alineación
    exp_atoms = []
    pred_atoms = []
    aligned_residue_ids = []  # Lista para almacenar IDs de residuos alineados
    aligned_atoms_pred = []

    # Iterar sobre los residuos de la estructura predicha
    for pred_chain in structure_pred.get_chains():
        for pred_res in pred_chain.get_residues():
            if pred_res.has_id('CA') and pred_res['CA'].get_bfactor() >= threshold:  # Filtrar por pLDDT (B-factor)
                pred_res_id = pred_res.get_id()[1]  # ID del residuo predicho

                # Buscar el residuo correspondiente en la estructura experimental
                for exp_chain in structure_exp.get_chains():
                    for exp_res in exp_chain.get_residues():
                        exp_res_id = exp_res.get_id()[1]  # ID del residuo experimental

                        # Comparar si los residuos coinciden en ID
                        if exp_res_id == pred_res_id and exp_res.has_id('CA'):
                            exp_atoms.append(exp_res['CA'])  # Agregar CA experimental
                            pred_atoms.append(pred_res['CA'])  # Agregar CA predicho
                            aligned_atoms_pred.append(pred_res['CA'])  # Para mantenerlos en visualización completa
                            aligned_residue_ids.append((exp_res_id, pred_res_id))  # Guardar IDs alineados
                            break

    # Verificar si se encontraron átomos CA para alinear
    if not exp_atoms or not pred_atoms:
        raise ValueError("No se encontraron átomos CA alineados para realizar el alineamiento.")

    # Alinear las estructuras usando los átomos CA coincidentes
    super_imposer = Superimposer()
    super_imposer.set_atoms(exp_atoms, pred_atoms)
    super_imposer.apply(structure_pred.get_atoms())  # Aplica la alineación a la estructura predicha

    # Guardar la estructura alineada completa
    io = PDBIO()
    io.set_structure(structure_pred)
    io.save(guardo_pdb__pred_superimposed)


    # Imprimo  los IDs de los residuos alineados
    pred_residue_ids = [pred_id for _, pred_id in aligned_residue_ids]
    print(f"Lista de IDs de residuos predichos alineados (pLDDT ≥ {threshold}):")
    print(pred_residue_ids)
    longitud=len(pred_residue_ids)
    print(f"Longitud de la lista de IDs de residuos predichos alineados: {longitud}")
     # Longitud en términos de número de residuos para estructuras
    longitud_exp = len(list(structure_exp.get_residues()))
    longitud_pred = len(list(structure_pred.get_residues()))

    print(f'Longitud de la estructura experimental: {longitud_exp}')
    print(f'Longitud de la estructura predicha: {longitud_pred}')


    return super_imposer
#____________________________Funcion calculo de distancias por CA__________________
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def aadistances_rmsd(pdb_file_exp, pdb_file_pred, start=0, end=0, plot=False):
    """
    Calcula las distancias RMSD entre átomos CA de residuos alineados de dos estructuras PDB
    y devuelve un DataFrame con la información relevante.

    :param pdb_file_exp: Ruta al archivo PDB de la estructura experimental.
    :param pdb_file_pred: Ruta al archivo PDB de la estructura predicha.
    :param start: Índice inicial para el cálculo de distancias (opcional).
    :param end: Índice final para el cálculo de distancias (opcional).
    :param plot: Booleano para decidir si se debe graficar el perfil de RMSD.
    :return: Un DataFrame con la información de los átomos alineados y sus distancias RMSD.
    """
    # Cargar las estructuras desde los archivos PDB
    parser = PDBParser(QUIET=True)
    structure_exp = parser.get_structure('exp', pdb_file_exp)  # Estructura experimental
    structure_pred = parser.get_structure('pred', pdb_file_pred)  # Estructura predicha

    # Obtener los residuos y átomos de CA correspondientes para el cálculo de distancias
    ref_atoms_coords = []
    sample_atoms_coords = []
    atom_info = []  # Para almacenar información de cada átomo
    rmsd_individual = []  # Para almacenar las distancias RMSD individuales

    # Iterar sobre los residuos de la estructura experimental
    for exp_chain in structure_exp.get_chains():
        for exp_res in exp_chain.get_residues():
            if exp_res.has_id('CA'):  # Asegurarse de que el residuo tenga un átomo CA
                exp_res_id = exp_res.get_id()[1]  # ID del residuo experimental

                # Buscar el residuo correspondiente en la estructura predicha
                for pred_chain in structure_pred.get_chains():
                    for pred_res in pred_chain.get_residues():
                        pred_res_id = pred_res.get_id()[1]  # ID del residuo predicho

                        # Comparar si el residuo predicho coincide en ID
                        if pred_res_id == exp_res_id:
                            if pred_res.has_id('CA'):  # Asegurarse de que también tenga CA
                                ref_atoms_coords.append(exp_res['CA'].get_coord())
                                sample_atoms_coords.append(pred_res['CA'].get_coord())

                                # Almacenar la información del átomo alineado
                                atom_info.append({
                                    'atom_name': exp_res['CA'].get_name(),
                                    'residue_name': exp_res.get_resname(),
                                    'residue_seq': exp_res.get_id()[1],
                                    'chain_id': exp_chain.get_id(),
                                    'pLDDT': pred_res['CA'].get_bfactor(),  # Usamos el bFactor como pLDDT del átomo predicho
                                })
                            break

    # Verificar si se encontraron átomos CA alineados
    if not ref_atoms_coords or not sample_atoms_coords:
        raise ValueError("No se encontraron átomos CA alineados para calcular distancias RMSD.")

    # Calcular distancias RMSD entre átomos CA
    if end == 0:
        end = len(ref_atoms_coords)  # Por defecto, hasta el final de las coordenadas

    for i in range(start, end):
        if i < len(ref_atoms_coords) and i < len(sample_atoms_coords):
            dist = np.sqrt(np.sum((ref_atoms_coords[i] - sample_atoms_coords[i]) ** 2))
            rmsd_individual.append(dist)
            atom_info[i]['CA_RMSD'] = dist  # Agregar la distancia RMSD al diccionario
    #RMSD GLOBAL
    suma=np.mean(rmsd_individual)
    print(f'Global RMSD: {suma:.2f} Å')

    # Crear un DataFrame con la información de los átomos y el RMSD
    df = pd.DataFrame(atom_info)

    # Graficar RMSD individual si el argumento `plot` es True
    if plot:
        plt.figure(figsize=(15, 6))
        plt.plot(rmsd_individual, label="Local CA-Distances")
        plt.xlabel('Atom Position')
        plt.ylabel('Distances (Å)')
        plt.title('Distances between CA Atoms')
        plt.legend()
        plt.show()

    # Retornar el DataFrame
    return df

## EJEMPLO de uso
# pdb_exp_NIS='C:/Users/flor_/FoldComparative/7UUY_NIS.pdb'
# pdb_pred_CF='C:/Users/flor_/FoldComparative/pdb_files_predicts/NIST_wt_unrelaxed_CF.pdb'
# pdb_superimposed_predict_CF = 'C:/Users/flor_/FoldComparative/7UUY_predict_superimposed_CF.pdb'
# pdb_superimposed_predict_no_aligned_CF = 'C:/Users/flor_/FoldComparative/7UUY_predict_no_aligned_CF.pdb'
# threshold=00.0

# super_imposer_CF = superimposed_pdb(pdb_exp_NIS, pdb_pred_CF, pdb_superimposed_predict_CF, pdb_superimposed_predict_no_aligned_CF,threshold)
#______________________________________Dataframe Distances(EXP-PRED)

def calc_distances_individual_p(p1_AFF, p1_fit_CF, start=0, end=0, plot=False):
    """
    Calcula las distancias euclidianas entre átomos CA de residuos alineados de dos estructuras PDB
    y devuelve un DataFrame con las distancias por posición de CA.
    
    :param p1_AFF: Ruta al archivo PDB de la estructura experimental (ej. AFF).
    :param p1_fit_CF: Ruta al archivo PDB de la estructura predicha (ej. CF ajustada).
    :param start: Índice inicial para el cálculo de distancias (opcional).
    :param end: Índice final para el cálculo de distancias (opcional).
    :param plot: Booleano para decidir si se debe graficar el perfil de las distancias (opcional).
    :return: Un DataFrame con las distancias euclidianas calculadas entre los átomos CA.
    """
    
    # Cargar las estructuras desde los archivos PDB
    parser = PDBParser(QUIET=True)
    structure_exp = parser.get_structure('exp', p1_AFF)  # Estructura experimental
    structure_pred = parser.get_structure('pred', p1_fit_CF)  # Estructura predicha

    # Obtener los residuos y átomos de CA correspondientes para el cálculo de distancias
    ref_atoms_coords = []
    sample_atoms_coords = []
    distances = []  # Lista para almacenar las distancias calculadas
    
    # Iterar sobre los residuos de la estructura experimental
    for exp_chain in structure_exp.get_chains():
        for exp_res in exp_chain.get_residues():
            if exp_res.has_id('CA'):  # Asegurarse de que el residuo tenga un átomo CA
                exp_res_id = exp_res.get_id()[1]  # ID del residuo experimental

                # Buscar el residuo correspondiente en la estructura predicha
                for pred_chain in structure_pred.get_chains():
                    for pred_res in pred_chain.get_residues():
                        pred_res_id = pred_res.get_id()[1]  # ID del residuo predicho

                        # Comparar si el residuo predicho coincide en ID
                        if pred_res_id == exp_res_id:
                            if pred_res.has_id('CA'):  # Asegurarse de que también tenga CA
                                ref_atoms_coords.append(exp_res['CA'].get_coord())
                                sample_atoms_coords.append(pred_res['CA'].get_coord())
                            break

    # Verificar si se encontraron átomos CA alineados
    if not ref_atoms_coords or not sample_atoms_coords:
        raise ValueError("No se encontraron átomos CA alineados para calcular distancias euclidianas.")

    # Calcular distancias euclidianas entre átomos CA
    if end == 0:
        end = len(ref_atoms_coords)  # Por defecto, hasta el final de las coordenadas

    for i in range(start, end):
        if i < len(ref_atoms_coords) and i < len(sample_atoms_coords):
            dist = np.sqrt(np.sum((ref_atoms_coords[i] - sample_atoms_coords[i]) ** 2))
            distances.append(dist)

    # Crear un DataFrame con las distancias
    df = pd.DataFrame([distances], columns=[f'αC_{i}' for i in range(start, end)])

    return df

#______________________________________Calculo TM (EXP-PRED)____________
from Bio.PDB import PDBParser
import numpy as np
import os

def calc_tmscore_aligned(pdb_file_exp, pdb_file_pred, names=None, silent=False):
    """
    Calcular el TM-score entre dos estructuras PDB usando átomos CA alineados.

    Parámetros:
    - pdb_file_exp: Ruta al archivo PDB de la estructura experimental.
    - pdb_file_pred: Ruta al archivo PDB de la estructura predicha (o lista de archivos PDB predichos).
    - names: Lista de nombres para las estructuras.
    - silent: Si es True, suprime la salida impresa.

    Retorna:
    - Lista con los valores de TM-score.
    """
    # Asignar nombres por defecto si no se proporcionan
    if names is None or len(names) < 2:
        names = ["Exp", "Pred"]

    pdb_parser = PDBParser(QUIET=True)

    # Cargar la estructura experimental
    structure_exp = pdb_parser.get_structure("reference", pdb_file_exp)

    # Extraer los átomos CA de la estructura experimental
    ref_atoms = {}
    for chain in structure_exp.get_chains():
        for res in chain.get_residues():
            if res.has_id('CA'):
                ref_atoms[res.get_id()] = res['CA'].get_coord()  # Guardar usando el ID del residuo

    if isinstance(pdb_file_pred, str):
        pdb_file_pred = [pdb_file_pred]

    tmscore_list = []

    for j, sample_path in enumerate(pdb_file_pred):
        # Cargar la estructura predicha
        try:
            structure_pred = pdb_parser.get_structure("sample", sample_path)
        except:
            structure_pred = pdb_parser.get_structure("sample", f"superimposed_{os.path.basename(sample_path)[13:]}")

        # Extraer los átomos CA de la estructura predicha
        sample_atoms = {}
        for chain in structure_pred.get_chains():
            for res in chain.get_residues():
                if res.has_id('CA'):
                    sample_atoms[res.get_id()] = res['CA'].get_coord()  # Guardar usando el ID del residuo

        # Calcular las distancias solo para los residuos que están en ambas estructuras
        common_residues = set(ref_atoms.keys()).intersection(set(sample_atoms.keys()))
        if len(common_residues) == 0:
            print("No se encontraron residuos alineados.")
            return tmscore_list

        distancia_euclidiana = []
        for res_id in common_residues:
            dist = np.sum((ref_atoms[res_id] - sample_atoms[res_id]) ** 2)
            distancia_euclidiana.append(dist)

        # Cálculo del TM-score basado en las distancias
        end = len(common_residues)
        sum_lcommon = 0
        d0_ltarget = 1.24 * np.cbrt(end - 15) - 1.8  # Constante d0 para TM-score

        for dist in distancia_euclidiana:
            sum_lcommon += 1 / (1 + (dist / d0_ltarget) ** 2)

        tm_score = float(sum_lcommon / end)
        tmscore_list.append(tm_score)

        if not silent:
            print(f"TM-score entre {names[0]} y {names[1]}: {tm_score:.4f}")

    return tmscore_list

############################################################################################

#*********************************FoldComparative between methods **********************


def save_afxt(data, file_path):
    """
    Saves the processed data to an .afxt file.

    Parameters:
    - data (any): Data to be saved, returned from fx.load_af3.
    - file_path (str): Path to the .afxt file to be created.

    Returns:
    - None
    """
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Convert data to a JSON string (or any other format)
        # Adjust the serialization based on the format required for .afxt
        json.dump(data, file, indent=4)
    
    print(f"Saved data to {file_path}")
def convert_zip_to_afxt(directory_path):
    """
    Converts all .zip files in the specified directory to .afxt files using fx.load_af3,
    and saves the .afxt files in a separate 'AF3_afxt' directory.

    Parameters:
    - directory_path (str): Path to the directory containing .zip files.

    Returns:
    - None
    """
    # Define the path for the AF3_afxt directory
    afxt_directory = os.path.join(directory_path, 'AF3_afxt')

    # Create the AF3_afxt directory if it does not exist
    if not os.path.exists(afxt_directory):
        os.makedirs(afxt_directory)

    # List all .zip files in the directory
    zip_files = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if file_name.endswith('.zip')]

    # Sort the files to maintain order if necessary
    # zip_files.sort()

    # Iterate over each .zip file with an index
    for index, zip_file in enumerate(zip_files, start=1):
        # Generate the af3 name iteratively as af3_1, af3_2, ..., af3_181
        af3_name = f'af3_{index}'

        # Derive the output .afxt file name and path
        afxt_file_name = os.path.splitext(os.path.basename(zip_file))[0] + '.afxt'
        afxt_file_path = os.path.join(afxt_directory, afxt_file_name)

        # Load the .zip file with fx.load_af3
        af3_data = fx.load_af3(af3_name, zip_file)

        # Save the .afxt file using save_afxt
        save_afxt(af3_data, afxt_file_path)

        print(f"Converted {zip_file} to {afxt_file_path} with af3_name {af3_name}")


def instances_objects_AF3(directory_path):
    """
    Concatenates predictions from afxt files named af3_1, af3_2, ..., af3_181 into a single dictionary.

    Parameters:
    - directory_path (str): Path to the directory containing afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # List of afxt files in order af3_1.afxt, af3_2.afxt, ..., af3_181.afxt
    afxt_files = [os.path.join(directory_path, f'af3_{index}.afxt') for index in range(1, 182)]

    # Iterate over each file path in the provided list
    for file_path in afxt_files:
        # Check if the file exists before attempting to load
        if os.path.exists(file_path):
            afxt_entry_part = afx.load(file_path)
            # Iterate over predictions in the current part
            for prediction_name, prediction_data in afxt_entry_part.items():
                afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
                prediction_counter += 1  # Increment the global prediction counter
        else:
            print(f"File {file_path} does not exist and will be skipped.")

    return afxt_entry
#--------------------------------Funcion load _ Result de referencia----------------------------------
from zipfile import ZipFile

def load_REF(filedir):
  from prediction_results import prediction_results
  Z = {}
  protein_count = 0
  extract_folder = os.path.basename(filedir[:-5])
  #os.makedirs(extract_folder, exist_ok=True)
  with ZipFile(filedir,'r') as fz:
    fz.extractall(".")

  if os.path.isdir(extract_folder):
    pass
  else:
    os.system(f"cp -R prediction_{extract_folder} {extract_folder}") #compatibility with old afxt files

  for path in os.listdir(extract_folder):
    long_path = os.path.join(extract_folder, path)
    if long_path.endswith(".txt"):
      with open(long_path,'r') as file:
        lines = file.readlines()
        file.close()
      for zipf in lines:
        zipf = zipf[:-1]
        if not "/" in zipf:
          zipf = os.path.join(extract_folder, zipf)
        if os.path.exists(zipf) == True: #Excluding linebreaks
              protein_count = protein_count + 1
              with ZipFile(zipf, 'r') as fz:
                for zip_info in fz.infolist():
                  if zip_info.filename[-1] == '/':
                    continue
                  tab = os.path.basename(zip_info.filename)
                  if tab.endswith(".txt"):
                    #zip_info.filename = os.path.basename(zip_info.filename)
                    with fz.open(zip_info.filename) as pred_info:
                      pred_lines = pred_info.readlines()
                      uncut_pred_lines = pred_info.read()
                      pred_info.close()
                    #details = pred_lines.values()
                    try:
                      ptmscore = float(re.findall(r"pTMScore=?([ \d.]+)",uncut_pred_lines)[0])
                    except:
                      ptmscore = 0
                    prediction_entry = prediction_results(pred_lines[0].strip().decode('UTF-8'),pred_lines[1].strip().decode('UTF-8'),pred_lines[2].strip().decode('UTF-8'),pred_lines[3].strip().decode('UTF-8'),ptmscore)
                    Z[f'REF_p{protein_count}'] = prediction_entry
  print("Loaded successfully.")
  #print(Z)
  predictions_AF=Z
  return predictions_AF


def instances_objects_CF(directory_path):
    """
    Concatenates predictions from all .afxt files in a directory into a single dictionary.

    Parameters:
    - directory_path (str): Path to the directory containing .afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions from all .afxt files.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file has the .afxt extension
        if file_name.endswith('.afxt'):
            file_path = os.path.join(directory_path, file_name)  # Full path to the .afxt file
            afxt_entry_part = afx.load(file_path)  # Load the .afxt file using afx.load()

            # Iterate over predictions in the current file and add them to the dictionary
            for prediction_name, prediction_data in afxt_entry_part.items():
                afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
                prediction_counter += 1  # Increment the global prediction counter

    return afxt_entry
#----------------------------------------------Carga AFL:----------------------------------------------

def instances_objects_AFL(directory_path):
    """
    Concatenates predictions from all .afxt files in a directory into a single dictionary.

    Parameters:
    - directory_path (str): Path to the directory containing .afxt files.

    Returns:
    - dict: Dictionary containing concatenated predictions from all .afxt files.
    """
    afxt_entry = {}  # Dictionary to store concatenated predictions
    prediction_counter = 1  # Global counter for predictions

    # Verificar si el directorio existe
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"El directorio '{directory_path}' no existe.")
    
    # Verificar si el directorio está vacío
    if not os.listdir(directory_path):
        raise FileNotFoundError(f"El directorio '{directory_path}' está vacío.")
    
    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file has the .afxt extension
        if file_name.endswith('.afxt'):
            file_path = os.path.join(directory_path, file_name)  # Full path to the .afxt file
            try:
                afxt_entry_part = afx.load(file_path)  # Load the .afxt file using afx.load()

                # Iterate over predictions in the current file and add them to the dictionary
                for prediction_name, prediction_data in afxt_entry_part.items():
                    afxt_entry[f'p{prediction_counter}'] = prediction_data  # Add the prediction to the dictionary
                    prediction_counter += 1  # Increment the global prediction counter
            
            except Exception as e:
                print(f"Error al cargar el archivo {file_path}: {e}")
        else:
            print(f"El archivo {file_name} no tiene la extensión '.afxt', se omite.")

    if not afxt_entry:
        print("No se encontraron archivos '.afxt' válidos en el directorio.")

    return afxt_entry

##****************** funcion para ordenar directorio de predicciones **********************************
def sort_results(result, order_p):
    """
    Reorganiza y renombra las claves de `result_X` según el orden proporcionado en `order_p`.

    Parámetros:
    - result_X (dict): El diccionario original con las claves a reorganizar.
    - order_p (list): Lista de enteros que indica el orden deseado de las claves.

    Retorna:
    - dict: Un nuevo diccionario con las claves reorganizadas y renombradas.
    """
    # Crear un diccionario para almacenar los resultados ordenados
    results_order = {}

    # Filtrar las claves y añadirlas al nuevo objeto
    for key in order_p:
        new_key = f'p{key}'
        if new_key in result:
            results_order[new_key] = result[new_key]

    # Crear un nuevo diccionario para las claves renombradas
    results_sort = {}

    # Iterar sobre las claves ordenadas y asignar nuevos nombres
    for i, old_key in enumerate(order_p, start=1):
        new_key = f'p{i}'  # Crear la nueva clave con el formato deseado
        if f'p{old_key}' in results_order:
            results_sort[new_key] = results_order[f'p{old_key}']  # Asignar el valor al diccionario con la nueva clave

    return results_sort
###################################### menos los AWS ########################################################
def get_ptmscore(results):
    """
    Función para extraer los valores de pTMscore y los nombres de los objetos prediction_results en el diccionario results.

    Parameters:
    results (dict): Diccionario que contiene objetos prediction_results.

    Returns:
    pd.DataFrame: DataFrame con los nombres de los archivos y sus correspondientes valores de pTMscore.
    """
    # Lista para almacenar los datos
    data = []

    # Recorrer cada entrada en el objeto results
    for key, result in results.items():
        # Acceder al valor de pTMscore
        ptmscore = result.ptmscore
        # Acceder al nombre del archivo
        name = result.name

        # Añadir el nombre del archivo, pTMscore y key al diccionario
        data.append({'key': key, 'Variant': name, 'pTMscore': ptmscore})

    # Crear un DataFrame con los datos
    df = pd.DataFrame(data)

    return df

#**************get_pTMscore_AWS**********************************
#colab PERO PROBAR OCN OTRAS 
from zipfile import ZipFile
import os
import re
import pandas as pd

# Función para extraer pTM score por archivo .afxt
def get_pTMscore(afxt_filepaths):
    """
    Extrae los valores de pTMScore de archivos .afxt que se encuentran dentro de archivos .zip.

    Parameters:
    - afxt_filepaths (list): Lista de rutas de los archivos .afxt.

    Returns:
    - DataFrame: Un DataFrame que contiene los nombres de archivo y los valores de pTMScore.
    """
    from prediction_results import prediction_results  # Asegúrate de tener importada esta clase

    Z = {}
    protein_count = 0
    ptmscore_values = []  # Lista para almacenar los valores de pTMScore
    txt_files = []

    # Itera sobre cada archivo en la lista de rutas de archivos
    for afxt_filepath in afxt_filepaths:
        if isinstance(afxt_filepath, str):  # Asegúrate de que sea una cadena válida
            extract_folder = os.path.basename(afxt_filepath)[:-5]  # Extrae el nombre del archivo sin la extensión .afxt
            
            # Extraer archivos del .zip
            with ZipFile(afxt_filepath, 'r') as fz:
                fz.extractall(".")

            # Verifica si el folder existe, si no, lo copia
            if not os.path.isdir(extract_folder):
                os.system(f"cp -R prediction_{extract_folder} {extract_folder}")

            # Recorre los archivos extraídos
            for path in os.listdir(extract_folder):
                long_path = os.path.join(extract_folder, path)
                if long_path.endswith(".txt"):
                    with open(long_path, 'r') as file:
                        lines = file.readlines()

                    for zipf in lines:
                        zipf = zipf.strip()
                        if not "/" in zipf:
                            zipf = os.path.join(extract_folder, zipf)

                        if os.path.exists(zipf):
                            protein_count += 1

                            ptmscore = 0  # Inicializar ptmscore antes del bloque try
                            with ZipFile(zipf, 'r') as fz_inner:
                                for zip_info in fz_inner.infolist():
                                    if zip_info.filename[-1] == '/':
                                        continue
                                    tab = os.path.basename(zip_info.filename)

                                    if tab.endswith(".txt"):
                                        with fz_inner.open(zip_info.filename) as pred_info:
                                            txt_files.append(tab)
                                            pred_lines = pred_info.readlines()
                                            for line in pred_lines:
                                                line = line.decode('utf-8')
                                                if line.startswith('pTMScore='):
                                                    ptmscore = float(line.split('=')[1].strip())
                                                    ptmscore_values.append(ptmscore)  # Almacenar el valor de pTMScore en la lista

                                                    break  # Rompe el bucle cuando se encuentra pTMScore

                            prediction_entry = prediction_results(
                                pred_lines[0].strip().decode('UTF-8'),
                                pred_lines[1].strip().decode('UTF-8'),
                                pred_lines[2].strip().decode('UTF-8'),
                                pred_lines[3].strip().decode('UTF-8'),
                                ptmscore
                            )
                            Z[f'p{protein_count}'] = prediction_entry

    # Crear un DataFrame con los valores de pTMScore
    df = pd.DataFrame({'Variant': txt_files, 'pTMscore': ptmscore_values})
    print("Loaded successfully.")
    return df

#**********************Funciones para el Curado de datos**********************************************************************
def sort_results(result, order_key):
    """
    Reorganiza y renombra las claves de `result` según el orden proporcionado en `order_key`.

    Parámetros:
    - result (dict): El diccionario original con las claves a reorganizar.
    - order_key (list): Lista de cadenas que indica el orden deseado de las claves.

    Retorna:
    - dict: Un nuevo diccionario con las claves reorganizadas y renombradas.
    """
    # Crear un diccionario para almacenar los resultados ordenados
    results_order = {}

    # Filtrar las claves y añadirlas al nuevo objeto
    for key in order_key:
        if key in result:
            results_order[key] = result[key]

    # Crear un nuevo diccionario para las claves renombradas
    results_sort = {}

    # Iterar sobre las claves ordenadas y asignar nuevos nombres
    for i, old_key in enumerate(order_key, start=1):
        new_key = f'p{i}'  # Crear la nueva clave con el formato deseado
        if old_key in results_order:
            results_sort[new_key] = results_order[old_key]  # Asignar el valor al diccionario con la nueva clave

    return results_sort

def filter_metadata(df):
    """
    Filtra y ordena un DataFrame de metadatos eliminando ciertas variantes y ordenando el resultado por 'NIS_Variant'.

    Parámetros:
    - df (DataFrame): El DataFrame original a ser procesado.

    Retorna:
    - DataFrame: El DataFrame filtrado y ordenado.
    """
    # Hacer una copia del DataFrame para evitar modificar el original
    df = df.copy()

    # Lista de variantes a eliminar
    variants_to_delete = [
        'NIST_p.Gly543Lys_Pathogenic', 'NIST_p.Gly543Ala_Benign', 'NIST_p.Ser547Arg_Pathogenic',
        'NIST_p.Gly561Glu_Pathogenic', 'NIST_p.Gly561Gln_Pathogenic', 'NIST_p.Arg569Trp_Benign',
        'NIST_p.Thr575Ala_Benign', 'NIST_p.Thr577Asp_Benign', 'NIST_p.Val580Ala_Benign',
        'NIST_p.Ser581Ala_Benign', 'NIST_p.Ser581Asp_Benign', 'NIST_p.Leu587Ala_Benign',
        'NIST_p.Val588Ala_Benign', 'NIST_p.Leu594Phe_Benign', 'NIST_p.Leu594Ala_Benign',
        'NIST_p.Asn608Ser_Benign', 'NIST_p.Leu612Ala_Benign', 'NIST_p.Glu621Ala_Benign'
    ]

    # Filtrar el DataFrame
    df_curada = df[~df['Variant'].isin(variants_to_delete)]

    # Ordenar por 'NIS_Variant'
    df_sorted = df_curada.sort_values(by='Variant')

    # Retornar el DataFrame resultante
    return df_sorted
####------------------Funciones para Metricas comparativas de distancias --------------
# ##-----------------------------------obtener PDBS--------------------------------------
# # Asegúrate de que fx.extract_zips está definido y funciona como se espera
# def extract_all_files(results):
#     # Itera sobre los números de 1 a 199 para generar las claves p1, p2, ..., p199
#     for i in range(1, 200):
#         key = f'p{i}'
        
#         # Verifica si la clave existe en el diccionario
#         if key in results:
#             result_obj = results[key]
            
#             try:
#                 # Obtén el directorio del objeto de resultados
#                 directory = result_obj.directory.partition("/")[0]  # Asegúrate de que esta sea la forma correcta de obtener el directorio
                
#                 if not os.path.isdir(directory):
#                     print(f"El directorio no existe: {directory}")
#                     continue

#                 # Llama a la función para extraer los archivos .pdb
#                 afx.extract_zips(directory)
#                 print(f"Archivos extraídos para {key} en {directory}")

#             except Exception as e:
#                 print(f"Error al extraer archivos para {key} en {directory}: {e}")
#         else:
#             print(f"Clave {key} no encontrada en results")
#----------------------------------------------------------------------------------------------------------
def superimposed_HC(aff_dir, cf_dir, output_superimposed_dir, output_no_aligned_dir, threshold):
    """
    Procesa archivos PDB en los directorios AFF y CF, alineándolos en pares.
    
    :param aff_dir: Directorio con archivos PDB AFF.
    :param cf_dir: Directorio con archivos PDB CF.
    :param output_superimposed_dir: Directorio para guardar estructuras alineadas.
    :param output_no_aligned_dir: Directorio para guardar estructuras con átomos no alineados.
    :param threshold: Umbral de pLDDT para seleccionar residuos.
    """
    # Obtener los archivos en ambos directorios
    aff_files = sorted([f for f in os.listdir(aff_dir) if f.endswith('.pdb')])
    cf_files = sorted([f for f in os.listdir(cf_dir) if f.endswith('.pdb')])
    
    if len(aff_files) != len(cf_files):
        raise ValueError("Los directorios AFF y CF deben contener el mismo número de archivos.")

    # Crear directorios de salida si no existen
    os.makedirs(output_superimposed_dir, exist_ok=True)
    os.makedirs(output_no_aligned_dir, exist_ok=True)

    # Procesar los pares de archivos
    for aff_file, cf_file in zip(aff_files, cf_files):
        aff_file_path = os.path.join(aff_dir, aff_file)
        cf_file_path = os.path.join(cf_dir, cf_file)
        
        # Definir los nombres para los archivos de salida
        base_name = os.path.splitext(aff_file)[0]
        output_superimposed_file = os.path.join(output_superimposed_dir, f"{base_name}_superimposed.pdb")
        output_no_aligned_file = os.path.join(output_no_aligned_dir, f"{base_name}_no_aligned.pdb")
        
        # Llamar a la función de superposición
        try:
            superimposed_pdb(
                aff_file_path,
                cf_file_path,
                output_superimposed_file,
                output_no_aligned_file,
                threshold
            )
        
        except Exception as e:
            print(f"Error al procesar el archivo {aff_file} y {cf_file}: {e}")



import os  # Importa el módulo para manejar operaciones con el sistema de archivos (e.., listados de archivos).
import pandas as pd  # Importa pandas para la manipulación de dataframes.g

def calc_distances(p1_REF, p1_fit_X, start=0, end=0, plot=False):
    """
    Función que calcula las distancias entre los átomos Cα de dos estructuras PDB.
    
    Parámetros:
    p1_AFF: ruta del archivo PDB de la estructura experimental.
    p1_fit_CF: ruta del archivo PDB de la estructura superpuesta.
    start: índice inicial para calcular las distancias.
    end: índice final para calcular las distancias.
    plot: si es True, genera un gráfico de las distancias (opcional).
    
    Retorna:
    Serie de pandas con las distancias RMSD entre átomos Cα.
    """
    
    parser = PDBParser(QUIET=True)  # Crea un parser para leer archivos PDB en modo silencioso (sin mostrar advertencias).
    structure_exp = parser.get_structure('exp', p1_REF)  # Lee la estructura experimental desde el archivo PDB.
    structure_pred = parser.get_structure('pred', p1_fit_X)  # Lee la estructura predicha desde el archivo PDB superpuesto.

    ref_atoms_coords = []  # Lista para almacenar las coordenadas de los átomos Cα de la estructura experimental.
    sample_atoms_coords = []  # Lista para almacenar las coordenadas de los átomos Cα de la estructura superpuesta.
    rmsd_individual = []  # Lista para almacenar las distancias RMSD individuales entre átomos Cα.

    # Recorre las cadenas y residuos de la estructura experimental.
    for exp_chain in structure_exp.get_chains():
        for exp_res in exp_chain.get_residues():
            if exp_res.has_id('CA'):  # Verifica si el residuo tiene un átomo Cα.
                exp_res_id = exp_res.get_id()[1]  # Obtiene el ID del residuo.
                # Recorre las cadenas y residuos de la estructura superpuesta.
                for pred_chain in structure_pred.get_chains():
                    for pred_res in pred_chain.get_residues():
                        pred_res_id = pred_res.get_id()[1]  # Obtiene el ID del residuo superpuesto.
                        # Verifica que los residuos tengan el mismo ID y ambos tengan un átomo Cα.
                        if pred_res_id == exp_res_id and pred_res.has_id('CA'):
                            # Agrega las coordenadas de los átomos Cα correspondientes a las listas.
                            ref_atoms_coords.append(exp_res['CA'].get_coord())
                            sample_atoms_coords.append(pred_res['CA'].get_coord())
                            break  # Detiene el ciclo cuando encuentra la coincidencia.

    # Verifica si se encontraron átomos Cα en ambas estructuras.
    if not ref_atoms_coords or not sample_atoms_coords:
        raise ValueError("No se encontraron átomos CA alineados.")  # Lanza un error si no hay átomos alineados.

    # Si no se proporciona un valor para 'end', se usa el total de átomos.
    if end == 0:
        end = len(ref_atoms_coords)

    # Calcula la distancia RMSD entre los átomos Cα dentro del rango de índices.
    for i in range(start, end):
        if i < len(ref_atoms_coords) and i < len(sample_atoms_coords):
            dist = np.sqrt(np.sum((ref_atoms_coords[i] - sample_atoms_coords[i]) ** 2))  # Calcula la distancia euclidiana.
            rmsd_individual.append(dist)  # Agrega la distancia a la lista.

    return pd.Series(rmsd_individual)  # Retorna las distancias como una Serie de pandas.
import os
import pandas as pd



def i_TMSCORE(pdb_file_exp, pdb_file_pred, names=None, silent=False):
    """
    Calcular el TM-score entre dos estructuras PDB usando átomos CA alineados.
    """
    if names is None or len(names) < 2:
        names = ["Exp", "Pred"]

    pdb_parser = PDBParser(QUIET=True)

    # Cargar la estructura experimental
    structure_exp = pdb_parser.get_structure("reference", pdb_file_exp)

    # Extraer los átomos CA de la estructura experimental
    ref_atoms = {}
    for chain in structure_exp.get_chains():
        for res in chain.get_residues():
            if res.has_id('CA'):
                ref_atoms[res.get_id()] = res['CA'].get_coord()

    if isinstance(pdb_file_pred, str):
        pdb_file_pred = [pdb_file_pred]

    tmscore_list = []

    for j, sample_path in enumerate(pdb_file_pred):
        try:
            structure_pred = pdb_parser.get_structure("sample", sample_path)
        except:
            structure_pred = pdb_parser.get_structure("sample", f"superimposed_{os.path.basename(sample_path)[13:]}")

        # Extraer los átomos CA de la estructura predicha
        sample_atoms = {}
        for chain in structure_pred.get_chains():
            for res in chain.get_residues():
                if res.has_id('CA'):
                    sample_atoms[res.get_id()] = res['CA'].get_coord()

        common_residues = set(ref_atoms.keys()).intersection(set(sample_atoms.keys()))
        if len(common_residues) == 0:
            print("No se encontraron residuos alineados.")
            return tmscore_list

        distancia_euclidiana = []
        for res_id in common_residues:
            dist = np.sum((ref_atoms[res_id] - sample_atoms[res_id]) ** 2)
            distancia_euclidiana.append(dist)

        end = len(common_residues)
        sum_lcommon = 0
        d0_ltarget = 1.24 * np.cbrt(end - 15) - 1.8  # Constante d0 para TM-score

        for dist in distancia_euclidiana:
            sum_lcommon += 1 / (1 + (dist / d0_ltarget) ** 2)

        tm_score = float(sum_lcommon / end)
        tmscore_list.append(tm_score)

        if not silent:
            print(f"TM-score entre {names[0]} y {names[1]}: {tm_score:.4f}")

    return tmscore_list


def TMSCORE(dir_ref, dir_x, name='TM(REF-X)'):
    """
    Iterar sobre archivos PDB en dos directorios y calcular TM-score para cada par.
    
    Parámetros:
    - dir_ref: Directorio que contiene las estructuras de referencia.
    - dir_x: Directorio que contiene las estructuras a comparar.
    - name: Nombre de la columna en el DataFrame que contiene los TM-scores.

    Retorna:
    - DataFrame con columnas 'Variant' y una columna con el nombre definido por 'name'.
    """
    # Obtener listas de archivos PDB en cada directorio
    ref_files = sorted([f for f in os.listdir(dir_ref) if f.endswith('.pdb')])
    x_files = sorted([f for f in os.listdir(dir_x) if f.endswith('.pdb')])

    # Verificar que haya la misma cantidad de archivos en ambos directorios
    if len(ref_files) != len(x_files):
        raise ValueError("El número de archivos en los directorios no coincide.")
    
    # Crear DataFrame vacío para almacenar los resultados
    df_tmscore = pd.DataFrame(columns=['Variant', name])

    # Iterar sobre los archivos pares
    for ref_file, x_file in zip(ref_files, x_files):
        ref_path = os.path.join(dir_ref, ref_file)
        x_path = os.path.join(dir_x, x_file)
        
        # Calcular TM-score entre los dos archivos PDB
        tm_scores = i_TMSCORE(ref_path, x_path, names=[ref_file, x_file], silent=True)
        
        # Añadir el resultado al DataFrame
        if tm_scores:
            df_tmscore = pd.concat([df_tmscore, pd.DataFrame({'Variant': [ref_file], name: [tm_scores[0]]})], ignore_index=True)
        else:
            print(f"No se pudo calcular el TM-score para el par {ref_file} y {x_file}.")
    
    return df_tmscore

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

def RMSD(p1_AFF, p1_fit_CF, residuos_ids):
    """
    Calcula las distancias RMSD entre los átomos Cα de residuos específicos entre dos estructuras PDB.
    
    Parámetros:
    p1_AFF: ruta del archivo PDB de la estructura experimental.
    p1_fit_CF: ruta del archivo PDB de la estructura superpuesta.
    residuos_ids: lista de IDs de los residuos que se desea comparar.
    plot: si es True, genera un gráfico de las distancias (opcional).
    
    Retorna:
    rmsd_individual: Serie de pandas con las distancias RMSD entre los residuos seleccionados.
    rmsd_global: Valor numérico que representa el RMSD global (promedio) entre los residuos seleccionados.
    """
    
    parser = PDBParser(QUIET=True)  # Inicializa el parser de PDB.
    structure_exp = parser.get_structure('exp', p1_AFF)  # Carga la estructura experimental.
    structure_pred = parser.get_structure('pred', p1_fit_CF)  # Carga la estructura superpuesta.
    
    ref_atoms_coords = []  # Coordenadas de los átomos Cα de la estructura experimental.
    sample_atoms_coords = []  # Coordenadas de los átomos Cα de la estructura superpuesta.
    rmsd_individual = []  # Lista para almacenar las distancias individuales entre residuos seleccionados.
    
    # Recorre las cadenas y residuos de la estructura experimental.
    for exp_chain in structure_exp.get_chains():
        for exp_res in exp_chain.get_residues():
            exp_res_id = exp_res.get_id()[1]  # Obtiene el ID del residuo.
            # Verifica si el residuo está en la lista de IDs especificada por el usuario.
            if exp_res_id in residuos_ids and exp_res.has_id('CA'):
                # Busca el residuo correspondiente en la estructura superpuesta.
                for pred_chain in structure_pred.get_chains():
                    for pred_res in pred_chain.get_residues():
                        pred_res_id = pred_res.get_id()[1]
                        if pred_res_id == exp_res_id and pred_res.has_id('CA'):
                            # Agrega las coordenadas de los átomos Cα de los residuos correspondientes.
                            ref_atoms_coords.append(exp_res['CA'].get_coord())
                            sample_atoms_coords.append(pred_res['CA'].get_coord())
                            break

    # Verifica si se encontraron residuos en ambas estructuras.
    if not ref_atoms_coords or not sample_atoms_coords:
        raise ValueError("No se encontraron átomos CA alineados para los residuos especificados.")

    # Calcula la distancia RMSD entre los residuos seleccionados.
    for i in range(len(ref_atoms_coords)):
        dist = np.sqrt(np.sum((ref_atoms_coords[i] - sample_atoms_coords[i]) ** 2))  # Distancia euclidiana.
        rmsd_individual.append(dist)  # Almacena la distancia individual.
    
    # Convierte la lista de distancias individuales a una Serie de pandas.
    rmsd_individual_series = pd.Series(rmsd_individual)
    
    # Calcula el RMSD global (promedio de las distancias individuales).
    rmsd_global = rmsd_individual_series.mean()
    
    # Si el parámetro plot es True, se puede añadir un gráfico opcional aquí (aún no implementado).
    
    return rmsd_individual_series, rmsd_global  # Retorna las distancias individuales y el RMSD global.
#--------------Calculo de distanciaeuclidiana por CA o RMSD individual por HC 
import pandas as pd
import os
from Bio.PDB import PDBParser
import numpy as np

def calc_distances(p1_AFF, p1_fit_CF, start=0, end=0, plot=False):
    
    parser = PDBParser(QUIET=True)
    structure_exp = parser.get_structure('exp', p1_AFF)
    structure_pred = parser.get_structure('pred', p1_fit_CF)

    ref_atoms_coords = []
    sample_atoms_coords = []
    rmsd_individual = []

    for exp_chain in structure_exp.get_chains():
        for exp_res in exp_chain.get_residues():
            if exp_res.has_id('CA'):
                exp_res_id = exp_res.get_id()[1]
                for pred_chain in structure_pred.get_chains():
                    for pred_res in pred_chain.get_residues():
                        pred_res_id = pred_res.get_id()[1]
                        if pred_res_id == exp_res_id and pred_res.has_id('CA'):
                            ref_atoms_coords.append(exp_res['CA'].get_coord())
                            sample_atoms_coords.append(pred_res['CA'].get_coord())
                            break

    if not ref_atoms_coords or not sample_atoms_coords:
        raise ValueError("No se encontraron átomos CA alineados.")

    if end == 0:
        end = len(ref_atoms_coords)

    for i in range(start, end):
        if i < len(ref_atoms_coords) and i < len(sample_atoms_coords):
            dist = np.sqrt(np.sum((ref_atoms_coords[i] - sample_atoms_coords[i]) ** 2))
            rmsd_individual.append(dist)

    return pd.Series(rmsd_individual)

# Función para iterar sobre todos los archivos y crear el DataFrame
def distances_profile(aff_dir, fit_cf_dir):
    # Obtener lista de archivos PDB en ambos directorios
    aff_files = sorted([f for f in os.listdir(aff_dir) if f.endswith('.pdb')])
    fit_cf_files = sorted([f for f in os.listdir(fit_cf_dir) if f.endswith('.pdb')])

    # Verificar que ambos directorios tengan el mismo número de archivos
    if len(aff_files) != len(fit_cf_files):
        raise ValueError("Los directorios no tienen el mismo número de archivos PDB.")

    # Crear un DataFrame vacío para almacenar las distancias
    combined_df = pd.DataFrame()

    # Iterar sobre los archivos y calcular las distancias
    for i, (aff_file, fit_cf_file) in enumerate(zip(aff_files, fit_cf_files)):
        p1_AFF = os.path.join(aff_dir, aff_file)
        p1_fit_CF = os.path.join(fit_cf_dir, fit_cf_file)
        
        # Calcular distancias para el par actual de archivos
        distances = calc_distances(p1_AFF, p1_fit_CF)

        # Agregar el resultado como una nueva fila al DataFrame
        combined_df = pd.concat([combined_df, distances.to_frame().T], ignore_index=True)

    # Renombrar las columnas con las posiciones de CA
    combined_df.columns = [f'αC_{i}' for i in range(1, combined_df.shape[1] + 1)]
    
    return combined_df

#--------------------------------------calculo RMSD tomando CA con plddt > threshold ----------------------------------------------------


def calculo_TMSCORE(distancias_euclidiana, end):
    """
    Calcula el TM-score entre dos conjuntos de coordenadas basándose en las distancias euclidianas.
    
    Parámetros:
    - distancias_euclidiana: lista de distancias euclidianas entre residuos.
    - end: número de residuos comunes.

    Retorna:
    - El valor del TM-score normalizado entre 0 y 1.
    """
    if end <= 0:
        return 0.0

    sum_lcommon = 0
    d0_ltarget = 1.24 * np.cbrt(end - 15) - 1.8  # Constante d0 para TM-score

    for dist in distancias_euclidiana:
        sum_lcommon += 1 / (1 + (dist / d0_ltarget) ** 2)

    tm_score = float(sum_lcommon / end)
    
    return tm_score

def get_local_distances(p1_ref, p1_fit_x, threshold=70):
    """
    Calcula las distancias RMSD entre los átomos Cα de residuos específicos entre dos estructuras PDB
    basándose en los residuos de p1_ref que tengan un valor de pLDDT mayor al umbral definido por el usuario.

    Parámetros:
    - p1_ref: ruta del archivo PDB de la estructura experimental.
    - p1_fit_x: ruta del archivo PDB de la estructura superpuesta.
    - threshold: valor del umbral de pLDDT para seleccionar residuos.

    Retorna:
    - Un diccionario con 'residuos_ids', 'distancias' y 'tm_score'.
    """
    
    parser = PDBParser(QUIET=True)
    structure_ref = parser.get_structure('ref', p1_ref)
    structure_fit = parser.get_structure('fit', p1_fit_x)
    
    ref_atoms_coords = []
    fit_atoms_coords = []
    residuos_ids = []
    rmsd_individual = []
    distancia_euclidiana = []
    
    for ref_chain in structure_ref.get_chains():
        for ref_res in ref_chain.get_residues():
            if ref_res.has_id('CA'):
                ca_atom = ref_res['CA']
                b_factor = ca_atom.get_bfactor()
                
                if b_factor >= threshold:
                    ref_res_id = ref_res.get_id()[1]
                    residuos_ids.append(ref_res_id)

                    for fit_chain in structure_fit.get_chains():
                        for fit_res in fit_chain.get_residues():
                            fit_res_id = fit_res.get_id()[1]
                            
                            if fit_res_id == ref_res_id and fit_res.has_id('CA'):
                                ref_atoms_coords.append(ref_res['CA'].get_coord())
                                fit_atoms_coords.append(fit_res['CA'].get_coord())
                                
                                dist = np.sqrt(np.sum((ref_res['CA'].get_coord() - fit_res['CA'].get_coord()) ** 2))
                                distancia_euclidiana.append(dist)
                                rmsd_individual.append(dist)
                                break

    if not ref_atoms_coords or not fit_atoms_coords:
        raise ValueError("No se encontraron residuos con pLDDT superior al umbral en ambas estructuras.")

    end = len(residuos_ids)
    tm_score = calculo_TMSCORE(distancia_euclidiana, end)

    return {
        'residuos_ids': residuos_ids,
        'distancias': rmsd_individual,
        'tm_score': tm_score
    }

def process_pdb_directories(ref_dir, fit_dir, threshold=70, rmsd_name='RMSD', tm_score_name='TM-score'):
    """
    Procesa pares de archivos PDB desde dos directorios y concatena los resultados en un DataFrame.

    Parámetros:
    - ref_dir: directorio que contiene los archivos PDB de referencia.
    - fit_dir: directorio que contiene los archivos PDB superpuestos.
    - threshold: valor umbral de pLDDT para seleccionar residuos.
    - rmsd_name: nombre para la columna de distancias RMSD.
    - tm_score_name: nombre para la columna de TM-score.

    Retorna:
    - Un DataFrame con la información de las variantes, residuos seleccionados, distancias y TM-score.
    """
    
    results_list = []
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.pdb')])
    fit_files = sorted([f for f in os.listdir(fit_dir) if f.endswith('.pdb')])

    if len(ref_files) != len(fit_files):
        raise ValueError("Los directorios no contienen la misma cantidad de archivos PDB.")

    for ref_file, fit_file in zip(ref_files, fit_files):
        p1_ref = os.path.join(ref_dir, ref_file)
        p1_fit_x = os.path.join(fit_dir, fit_file)
        variant = os.path.splitext(ref_file)[0]

        distances_info = get_local_distances(p1_ref, p1_fit_x, threshold)

        df_variant = pd.DataFrame({
            'variant': variant,
            'residuos_ids': distances_info['residuos_ids'],
            rmsd_name: distances_info['distancias'],
            tm_score_name: distances_info['tm_score']
        })

        results_list.append(df_variant)

    final_df = pd.concat(results_list, ignore_index=True)
    
    return final_df
#-----------------------------------metrica bland altman-------------------------------------------------
# Definir los datos y calcular las características del gráfico de Bland-Altman con LOESS
def bland_altman_subplot(ax, x, y, title, xlabel, ylabel):
     # Cálculo de la media de las diferencias y los límites de acuerdo
    mean_diff = np.mean(y - x)
    LoA = 1.96 * np.std(y - x)

    # Calcular LOESS
    smoothed = lowess(y - x, (x + y) / 2, frac=0.03)

    ax.scatter((x + y) / 2, y - x, color='black', alpha=0.5, label='Diferencias')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='orange', linestyle='-', linewidth=2, label='LOESS')
    ax.axhline(mean_diff, color='skyblue', linestyle='-', linewidth=2, label='Media de las diferencias')
    ax.axhline(mean_diff + LoA, color='green', linestyle='--', linewidth=2, label='Límites de acuerdo (±1.96 SD)')
    ax.axhline(mean_diff - LoA, color='green', linestyle='--', linewidth=2)

   
    ax.set_title(title, fontsize=16, fontweight='bold')  # Título en negrita y tamaño aumentado
    ax.set_xlabel(xlabel, fontsize=14,fontweight='bold')  # Etiqueta del eje X en tamaño aumentado
    ax.set_ylabel(ylabel, fontsize=14,fontweight='bold')  # Etiqueta del eje Y en tamaño aumentado
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    plt.grid(True)  
