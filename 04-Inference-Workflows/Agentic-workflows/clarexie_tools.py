"""
Custom tools for scientific research assistance.
These tools help gather information from various sources.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
import json


@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for scientific papers.

    Parameters
    ----------
    query : str
        Search query string (keywords, title, author, etc.)
    max_results : int, optional
        Maximum number of results to return, by default 3

    Returns
    -------
    str
        JSON string containing paper titles, authors, abstracts, and arXiv IDs

    Examples
    --------
    >>> search_arxiv("machine learning protein folding", max_results=2)
    """
    import arxiv
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary[:300] + "...",  # Truncate for brevity
                "arxiv_id": paper.entry_id.split('/')[-1],
                "published": paper.published.strftime("%Y-%m-%d"),
                "url": paper.entry_id
            })
        
        return json.dumps({
            "status": "success",
            "query": query,
            "num_results": len(results),
            "papers": results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to search arXiv: {str(e)}"
        })


@tool
def lookup_molecular_properties(molecule_name: str) -> str:
    """Look up basic molecular properties from PubChem.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule to look up

    Returns
    -------
    str
        JSON string containing molecular formula, weight, SMILES, and other properties

    Examples
    --------
    >>> lookup_molecular_properties("caffeine")
    """
    import pubchempy as pcp
    
    try:
        compounds = pcp.get_compounds(molecule_name, 'name')
        
        if not compounds:
            return json.dumps({
                "status": "error",
                "message": f"No compound found for '{molecule_name}'"
            })
        
        compound = compounds[0]
        
        # Get available properties
        properties = {
            "status": "success",
            "molecule_name": molecule_name,
            "cid": compound.cid,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "canonical_smiles": compound.canonical_smiles,
            "isomeric_smiles": compound.isomeric_smiles,
            "iupac_name": compound.iupac_name,
            "xlogp": compound.xlogp,
            "exact_mass": compound.exact_mass,
            "charge": compound.charge,
            "complexity": compound.complexity,
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
        }
        
        return json.dumps(properties, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to lookup properties: {str(e)}"
        })


@tool
def calculate_simple_statistics(numbers: List[float], operation: str = "all") -> str:
    """Calculate basic statistics on a list of numbers.

    Parameters
    ----------
    numbers : List[float]
        List of numerical values
    operation : str, optional
        Type of statistic to calculate: "mean", "median", "std", "min", "max", or "all"

    Returns
    -------
    str
        JSON string containing the requested statistics

    Examples
    --------
    >>> calculate_simple_statistics([1.5, 2.3, 3.7, 4.2], "all")
    """
    import statistics
    
    try:
        if not numbers:
            return json.dumps({
                "status": "error",
                "message": "Empty list provided"
            })
        
        results = {
            "status": "success",
            "count": len(numbers),
            "data": numbers
        }
        
        if operation in ["mean", "all"]:
            results["mean"] = statistics.mean(numbers)
        
        if operation in ["median", "all"]:
            results["median"] = statistics.median(numbers)
        
        if operation in ["std", "all"] and len(numbers) > 1:
            results["std_dev"] = statistics.stdev(numbers)
        
        if operation in ["min", "all"]:
            results["min"] = min(numbers)
        
        if operation in ["max", "all"]:
            results["max"] = max(numbers)
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to calculate statistics: {str(e)}"
        })


@tool
def search_protein_database(protein_name: str) -> str:
    """Search for protein information using UniProt.

    Parameters
    ----------
    protein_name : str
        Name or ID of the protein to search for

    Returns
    -------
    str
        JSON string containing protein information including sequence length, organism, function

    Examples
    --------
    >>> search_protein_database("insulin")
    """
    import requests
    
    try:
        # Search UniProt REST API
        search_url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=json&size=3"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('results'):
            return json.dumps({
                "status": "error",
                "message": f"No protein found for '{protein_name}'"
            })
        
        results = []
        for entry in data['results'][:3]:
            protein_info = {
                "accession": entry.get('primaryAccession', 'N/A'),
                "protein_name": entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A'),
                "organism": entry.get('organism', {}).get('scientificName', 'N/A'),
                "gene": entry.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A') if entry.get('genes') else 'N/A',
                "sequence_length": entry.get('sequence', {}).get('length', 'N/A'),
                "mass": entry.get('sequence', {}).get('molWeight', 'N/A'),
            }
            results.append(protein_info)
        
        return json.dumps({
            "status": "success",
            "query": protein_name,
            "num_results": len(results),
            "proteins": results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to search protein database: {str(e)}"
        })
