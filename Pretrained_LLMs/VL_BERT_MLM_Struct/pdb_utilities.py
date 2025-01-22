from Bio.PDB import PDBParser, Polypeptide
import numpy as np 


def print_chain_ids(pdb_file):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    chain_ids = [chain.id for chain in structure[0]]
    print("Chain IDs:", chain_ids)
    
    return chain_ids 

def parse_pdb_chain(pdb_file, chain_id):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    chain_ids = [chain.id for chain in structure[0]]
    
    if chain_id not in chain_ids:
        chain_id = chain_id.swapcase()
    # Find the specified chain
    chain = structure[0][chain_id]
    
    # Collect atom information for the specified chain
    atoms = []
    for atom in chain.get_atoms():
        atoms.append({
            'name': atom.get_name(),                    # Atom name
            'element': atom.element,                    # Atom element
            'coord': atom.coord,                        # Coordinates (x, y, z)
            'residue': atom.get_parent().get_resname()  # Residue name
        })
    
    # Build the amino acid sequence for the specified chain
    residues = [residue for residue in chain if Polypeptide.is_aa(residue)]
    sequence = Polypeptide.Polypeptide(residues).get_sequence()
    
    return str(sequence), atoms


def get_atom_distance_matrix(pdb_file, chain_id):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    chain_ids = [chain.id for chain in structure[0]]
    if chain_id not in chain_ids:
        chain_id = chain_id.swapcase()
    
    # Extract the specified chain
    chain = structure[0][chain_id]
    
    # Collect atom coordinates in the specified chain
    atom_coords = []
    for atom in chain.get_atoms():
        atom_coords.append(atom.coord)
    
    # Convert to numpy array
    atom_coords = np.array(atom_coords)
    
    # Calculate the pairwise distance matrix manually
    num_atoms = atom_coords.shape[0]
    dist_matrix = np.zeros((num_atoms, num_atoms))  # Initialize a distance matrix
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            # Calculate Euclidean distance
            dist_matrix[i, j] = np.sqrt(np.sum((atom_coords[i] - atom_coords[j]) ** 2))
    
    return dist_matrix 


def get_residue_distance_matrix(pdb_file, chain_id):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    chain_ids = [chain.id for chain in structure[0]]
    
    if chain_id not in chain_ids:
        chain_id = chain_id.swapcase()
    
    # Extract the specified chain
    chain = structure[0][chain_id]
    
    # Get a list of standard residues in the chain
    standard_residues = [residue for residue in chain if Polypeptide.is_aa(residue)]
    num_residues = len(standard_residues)

    # Initialize the distance matrix
    dist_matrix = np.zeros((num_residues, num_residues))

    # Calculate the distance between each pair of standard residues
    for i in range(num_residues):
        for j in range(num_residues):
            if i != j:  # Skip distance to itself
                # Get atoms for both residues
                atoms_i = standard_residues[i].get_atoms()
                atoms_j = standard_residues[j].get_atoms()
                
                # Compute the minimum distance between all pairs of atoms
                min_distance = float('inf')
                for atom_i in atoms_i:
                    for atom_j in atoms_j:
                        distance = np.sqrt(np.sum((atom_i.coord - atom_j.coord) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                
                # Store the minimum distance in the distance matrix
                dist_matrix[i, j] = min_distance 
    
    sequence = Polypeptide.Polypeptide(standard_residues).get_sequence()
    
    return dist_matrix, sequence

def get_ca_distance_matrix(pdb_file, chain_id):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    
    chain_ids = [chain.id for chain in structure[0]]
    if chain_id not in chain_ids:
        chain_id = chain_id.swapcase()
    
    # Extract the specified chain
    chain = structure[0][chain_id]
    
    # Collect CA atom coordinates in the specified chain
    ca_coords = []
    for residue in chain:
        if 'CA' in residue:
            ca_coords.append(residue['CA'].coord)

    # Convert to numpy array
    ca_coords = np.array(ca_coords)
    
    # Calculate the pairwise distance matrix manually
    num_residues = ca_coords.shape[0]
    dist_matrix = np.zeros((num_residues, num_residues))  # Initialize a distance matrix
    
    for i in range(num_residues):
        for j in range(num_residues):
            # Calculate Euclidean distance
            dist_matrix[i, j] = np.sqrt(np.sum((ca_coords[i] - ca_coords[j]) ** 2))
    
    return dist_matrix



    
    