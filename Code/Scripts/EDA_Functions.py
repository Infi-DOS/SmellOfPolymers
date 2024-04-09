import os
import matplotlib.pyplot as plt

def save_figure(output, title):
    """
    Save the current matplotlib figure to the specified directory with the given title.
    
    Parameters:
        output (str): The directory path where the figure should be saved.
        title (str): The filename for the figure.
    
    Returns:
        None
    """
    file_path = os.path.join(output, title)
    if not os.path.exists(output):
        os.makedirs(output)
    plt.savefig(file_path, dpi=300)

def sort_words(sentence):
    """
    Sort the words in a sentence alphabetically. The words in the sentence are comma-separated.
    
    Parameters:
        sentence (str): The sentence containing the words to be sorted.
        
    Returns:
        str: A string containing the sorted words, joined by commas.
    """
    words = sentence.split(',')
    return ','.join(sorted(words))

def check_duplicates(sentence):
    """
    Check if there are any duplicate words in a sentence. The words in the sentence are comma-separated.
    
    Parameters:
        sentence (str): The sentence to check for duplicates.
        
    Returns:
        bool: True if there are duplicates, False otherwise.
    """
    classes = sentence.split(',')
    return len(classes) != len(set(classes))

def remove_duplicates(sentence):
    """
    Remove duplicate words from a sentence. The words in the sentence are comma-separated.
    
    Parameters:
        sentence (str): The sentence from which to remove duplicates.
        
    Returns:
        str: A string with duplicates removed, joined by commas.
    """
    words = sentence.split(',')
    unique_words = set(words)
    return ','.join(unique_words)

def check_graph_2d(id1, id2, data):
    """
    Display 2D images of molecules identified by their IDs in the given dataset.
    
    Parameters:
        id1 (int): The ID of the first molecule.
        id2 (int): The ID of the second molecule.
        data (DataFrame): The dataset containing the SMILES representation of molecules.
        
    Returns:
        None
    """
    smile1 = data['SMILES'][id1]
    m1 = Chem.MolFromSmiles(smile1)
    img1 = Draw.MolToImage(m1)

    smile2 = data['SMILES'][id2]
    m2 = Chem.MolFromSmiles(smile2)
    img2 = Draw.MolToImage(m2)

    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')
    plt.show()

def display_molecule(smiles):
    """
    Generate and display a 3D model of a molecule from its SMILES representation.
    
    Parameters:
        smiles (str): The SMILES representation of the molecule.
        
    Returns:
        view (py3Dmol.view): A 3Dmol view object of the molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)
    molecule_3D = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule_3D, AllChem.ETKDG())

    block = Chem.MolToMolBlock(molecule_3D)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(block, format='sdf')
    view.setStyle({'stick': {}})
    view.setBackgroundColor('white')
    view.zoomTo()
    return view

def check_graph_3d(id1, id2, data):
    """
    Display 3D models of molecules identified by their IDs in the given dataset.
    
    Parameters:
        id1 (int): The ID of the first molecule.
        id2 (int): The ID of the second molecule.
        data (DataFrame): The dataset containing the SMILES representation of molecules.
        
    Returns:
        None
    """
    smile1 = data['SMILES'][id1]
    smile2 = data['SMILES'][id2]

    view1 = display_molecule(smile1)
    view2 = display_molecule(smile2)

    display(view1)
    display(view2)