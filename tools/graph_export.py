# gephi_exporter.py

import os
import sys
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from collections import defaultdict

# Assumes utils.py, config.py, and link_notes.py are in the same directory
import utils
import config
import link_notes

def export_to_graphml(vault_root: str, similarity_threshold: float, chronological_weight: float, output_filepath: str):
    """
    Exports all notes and their semantic links to a Gephi-compatible GraphML file.

    Args:
        vault_root (str): The root path to the Obsidian vault.
        similarity_threshold (float): The cosine similarity score to consider a link.
        chronological_weight (float): The fixed weight to assign to chronological edges.
        output_filepath (str): The path to save the generated GraphML file.
    """
    if not os.path.exists(vault_root):
        print(f"Error: Vault root not found at {vault_root}.")
        return

    print("--- Zettelpal Gephi Exporter ---")
    print(f"Scanning vault for notes at: {vault_root}")
    print(f"Using semantic similarity threshold: {similarity_threshold:.2f}")
    print(f"Assigning chronological links a fixed weight of: {chronological_weight:.2f}")

    # Load all note data from cache, which includes metadata like source and created
    all_vault_notes_data = link_notes.update_and_load_vault_embeddings(vault_root, config.EMBEDDINGS_CACHE_FILE)

    if len(all_vault_notes_data) < 2:
        print("Not enough notes with valid embeddings to create a graph. Exiting.")
        return

    # Filter out notes without embeddings and with source/created metadata for linking
    notes_to_process = [
        n for n in all_vault_notes_data
        if n["embedding"] is not None
    ]
    if not notes_to_process:
        print("No notes with valid embeddings found. Exiting.")
        return
        
    print(f"Found {len(notes_to_process)} notes with embeddings and metadata. Building graph...")
    
    # --- Group notes by source for chronological linking (same logic as link_notes.py) ---
    notes_by_source = defaultdict(list)
    for note_data in notes_to_process:
         source = note_data.get("source")
         created_dt = note_data.get("created")
         if source and isinstance(source, str) and source.strip() and created_dt is not None:
              notes_by_source[source].append(note_data)

    source_sorted_notes_map = {}
    for source, notes_list in notes_by_source.items():
         sorted_list = sorted(notes_list, key=lambda x: (x["created"], x["filename_stem"]))
         source_sorted_notes_map[source] = sorted_list

    # --- Generate GraphML structure ---
    ns = "http://graphml.graphdrawing.org/xmlns"
    xsi_ns = "http://www.w3.org/2001/XMLSchema-instance"
    schema_loc = "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    
    graphml = ET.Element("graphml", {
        "xmlns": ns,
        "xmlns:xsi": xsi_ns,
        "xsi:schemaLocation": schema_loc
    })
    
    # Define keys for node and edge attributes
    ET.SubElement(graphml, "key", {"id": "label", "for": "node", "attr.name": "label", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "weight", "for": "edge", "attr.name": "weight", "attr.type": "double"})
    ET.SubElement(graphml, "key", {"id": "link_type", "for": "edge", "attr.name": "link_type", "attr.type": "string"})

    graph = ET.SubElement(graphml, "graph", {"id": "G", "edgedefault": "undirected"})
    
    # --- Add nodes ---
    print("\nAdding nodes...")
    for note in notes_to_process:
        node_id = note["filename_stem"]
        node = ET.SubElement(graph, "node", {"id": node_id})
        ET.SubElement(node, "data", {"key": "label"}).text = note["display_title"]
    print(f"  Nodes exported: {len(notes_to_process)}")

    # --- Add edges ---
    print("\nAdding edges...")
    edge_count = 0
    # Use a set to track edges to prevent duplicates
    edges = set() 
    
    # Pre-calculate all pairwise similarities
    embeddings = np.array([n["embedding"] for n in notes_to_process])
    filepaths = [n["filepath"] for n in notes_to_process]
    filepath_to_index = {fp: i for i, fp in enumerate(filepaths)}
    similarity_matrix = cosine_similarity(embeddings)
    
    # Add semantic edges
    print("  Adding semantic edges...")
    for i in range(len(notes_to_process)):
        for j in range(i + 1, len(notes_to_process)):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                source_node_id = notes_to_process[i]["filename_stem"]
                target_node_id = notes_to_process[j]["filename_stem"]
                # Store edge as a sorted tuple to prevent duplicates
                edge_tuple = tuple(sorted((source_node_id, target_node_id)))
                if edge_tuple not in edges:
                    edge = ET.SubElement(graph, "edge", {"source": source_node_id, "target": target_node_id})
                    ET.SubElement(edge, "data", {"key": "weight"}).text = str(similarity)
                    ET.SubElement(edge, "data", {"key": "link_type"}).text = "semantic"
                    edges.add(edge_tuple)
                    edge_count += 1
    
    # Add chronological edges
    print("  Adding chronological edges...")
    chrono_edge_count = 0
    for source_id, notes_list in source_sorted_notes_map.items():
         if len(notes_list) > 1:
             for i in range(len(notes_list) - 1):
                 source_node_id = notes_list[i]["filename_stem"]
                 target_node_id = notes_list[i+1]["filename_stem"]
                 # Store edge as a sorted tuple
                 edge_tuple = tuple(sorted((source_node_id, target_node_id)))
                 if edge_tuple not in edges:
                     edge = ET.SubElement(graph, "edge", {"source": source_node_id, "target": target_node_id})
                     ET.SubElement(edge, "data", {"key": "weight"}).text = str(chronological_weight)
                     ET.SubElement(edge, "data", {"key": "link_type"}).text = "chronological"
                     edges.add(edge_tuple)
                     edge_count += 1
                     chrono_edge_count += 1
    
    tree = ET.ElementTree(graphml)
    tree.write(output_filepath, encoding="utf-8", xml_declaration=True)
    
    print(f"\nExport complete. A GraphML file was created at: {output_filepath}")
    print(f"  Semantic edges exported: {edge_count - chrono_edge_count}")
    print(f"  Chronological edges exported: {chrono_edge_count}")
    print(f"  Total edges exported: {edge_count}")
    print(f"  Total nodes exported: {len(notes_to_process)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Zettelpal semantic graph to GraphML for Gephi visualization.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for creating a link (default: {config.SIMILARITY_THRESHOLD:.2f})."
    )
    parser.add_argument(
        "--chrono_weight",
        type=float,
        default=0.60,
        help=f"Fixed weight for chronological links (default: 0.60)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="zettelpal_graph.graphml",
        help="Path and filename for the output GraphML file (default: zettelpal_graph.graphml)."
    )
    args = parser.parse_args()

    # Get the output path, resolving relative paths
    output_path = os.path.abspath(args.output)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    export_to_graphml(config.OBSIDIAN_VAULT_ROOT, args.threshold, args.chrono_weight, output_path)