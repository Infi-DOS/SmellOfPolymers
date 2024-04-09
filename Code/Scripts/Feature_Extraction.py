# RDKit library
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, RDKFingerprint, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
import pandas as pd

###################
# Descriptors
###################

def add_all_descriptors(df, smiles_col="SMILES"):
    """
    Compute molecular descriptors for each molecule in the DataFrame and append them as new columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the SMILES representations of molecules.
        smiles_col (str, optional): The name of the column in df that contains the SMILES strings. Defaults to "SMILES".

    Returns:
        pd.DataFrame: The input DataFrame with molecular descriptors appended as new columns.
    """
    # df['Mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    df.loc[:, 'Mol'] = df.loc[:, smiles_col].apply(Chem.MolFromSmiles)

    def compute_descriptors(mol):
        if mol:
            return Descriptors.CalcMolDescriptors(mol)
        else:
            return [None] * len(Descriptors.descList)

    descriptor_names = [name for name, _ in Descriptors.descList]

    descriptors = df['Mol'].apply(compute_descriptors)
    descriptor_df = pd.DataFrame(descriptors.tolist(), columns=descriptor_names)

    df_with_descriptors = pd.concat([df, descriptor_df], axis=1).drop(columns=['Mol'])

    return df_with_descriptors

###################
# Fingerprints
###################

def add_selected_fingerprints(df, smiles_col='SMILES', rdkit_fp=False, atompair_fp=False, torsion_fp=False, maccs_fp=False, morgan_fp=False):
    """
    Compute selected molecular fingerprints for each molecule in the DataFrame and append them as new columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the SMILES representations of molecules.
        smiles_col (str, optional): The name of the column in df that contains the SMILES strings. Defaults to "SMILES".
        rdkit_fp (bool, optional): Whether to compute RDKit fingerprints. Defaults to False.
        atompair_fp (bool, optional): Whether to compute Atom Pair fingerprints. Defaults to False.
        torsion_fp (bool, optional): Whether to compute Topological Torsion fingerprints. Defaults to False.
        maccs_fp (bool, optional): Whether to compute MACCS keys fingerprints. Defaults to False.
        morgan_fp (bool, optional): Whether to compute Morgan fingerprints. Defaults to False.

    Returns:
        pd.DataFrame: The input DataFrame with selected molecular fingerprints appended as new columns.
    """
    # df['Mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    df.loc[:, 'Mol'] = df.loc[:, smiles_col].apply(Chem.MolFromSmiles)


    if rdkit_fp:
        rdkit_fps = df['Mol'].apply(lambda mol: list(RDKFingerprint(mol)) if mol else [None]*2048)
        rdkit_df = pd.DataFrame(rdkit_fps.tolist(), columns=[f'RDKit_FP_{i}' for i in range(2048)])
        df = pd.concat([df, rdkit_df], axis=1)

    if atompair_fp:
        atompair_fps = df['Mol'].apply(lambda mol: list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)) if mol else [None]*2048)
        atompair_df = pd.DataFrame(atompair_fps.tolist(), columns=[f'AtomPair_FP_{i}' for i in range(2048)])
        df = pd.concat([df, atompair_df], axis=1)

    if torsion_fp:
        torsion_fps = df['Mol'].apply(lambda mol: list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)) if mol else [None]*2048)
        torsion_df = pd.DataFrame(torsion_fps.tolist(), columns=[f'TopologicalTorsion_FP_{i}' for i in range(2048)])
        df = pd.concat([df, torsion_df], axis=1)

    if maccs_fp:
        maccs_fps = df['Mol'].apply(lambda mol: list(MACCSkeys.GenMACCSKeys(mol)) if mol else [None]*166)
        maccs_df = pd.DataFrame(maccs_fps.tolist(), columns=[f'MACCS_FP_{i}' for i in range(166)])
        df = pd.concat([df, maccs_df], axis=1)

    if morgan_fp:
        morgan_fps = df['Mol'].apply(lambda mol: list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) if mol else [None]*2048)
        morgan_df = pd.DataFrame(morgan_fps.tolist(), columns=[f'Morgan_FP_{i}' for i in range(2048)])
        df = pd.concat([df, morgan_df], axis=1)

    df = df.drop(columns=['Mol'])

    return df
