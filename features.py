# atom_features: encode atom symbol, num bonded atoms, num hydrogens, implicit valence, if aromatic

import numpy as np
from rdkit import Chem

def encoding(feat, featArray):
    if feat not in featArray:
        feat = featArray[0]
    return list(map(lambda f: int(feat == f), featArray))

def getAtomFeatures(atom, just_structure=False):
    symbolArray = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    degArray = [x for x in range(6)]
    if just_structure:
        return np.array([
            *encoding(atom.GetSymbol(), symbolArray),
            *encoding(atom.GetTotalNumHs(), degArray[:-1]),
        ])
    else:
        return np.array([
            *encoding(atom.GetSymbol(), symbolArray),
            *encoding(atom.GetDegree(), degArray),
            *encoding(atom.GetTotalNumHs(), degArray[:-1]),
            *encoding(atom.GetImplicitValence(), degArray),
            int(atom.GetIsAromatic())
        ])

def getBondFeatures(bond, just_structure=False):
    bondType = bond.GetBondType()
    single = (bondType == Chem.rdchem.BondType.SINGLE)
    double = (bondType == Chem.rdchem.BondType.DOUBLE)
    triple = (bondType == Chem.rdchem.BondType.TRIPLE)
    none = False  # bond exists so false; feature needed so autoreg is able to predict 'none'
    aromatic = (bondType == Chem.rdchem.BondType.AROMATIC)
    conjugated = bond.GetIsConjugated()
    in_ring = bond.IsInRing()

    if just_structure:
        return np.array(list(map(int, [none,single,double,triple])))
    else:
        return np.array(list(map(int, [single,double,triple,aromatic,conjugated,in_ring])))
    
def num_atom_features(just_structure=False):
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(getAtomFeatures(a,just_structure=just_structure))

def num_bond_features(just_structure=False):
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(getBondFeatures(simple_mol.GetBonds()[0],just_structure=just_structure))