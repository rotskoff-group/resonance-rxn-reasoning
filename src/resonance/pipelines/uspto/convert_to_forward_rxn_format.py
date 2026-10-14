#!/usr/bin/env python3
"""
Convert USPTO-50k JSONL files to ForwardRxnParser compatible format.

Ported from run_dir('uspto')/convert_to_forward_rxn_format.py.
CLI: python -m resonance.pipelines.uspto.convert_to_forward_rxn_format INPUT [-o OUTPUT]

This script converts JSONL files with structure:
    {"reaction_id": "...", "reactants": [...], "products": [...]}

To JSON format expected by ForwardRxnParser:
    [
        {
            "metadata": {"reaction_id": "..."},
            "original_smiles": {
                "Starting Material": "...",
                "Reagents": "...",
                "End Product": "..."
            }
        },
        ...
    ]
"""

import argparse
import json
from pathlib import Path
from typing import Any


def convert_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single USPTO-50k entry to ForwardRxnParser format.

    Args:
        entry: Dictionary with keys 'reaction_id', 'reactants', 'products'

    Returns:
        Dictionary in ForwardRxnParser format
    """
    reactants = entry.get("reactants", [])
    products = entry.get("products", [])

    # Assume first reactant is the starting material, rest are reagents
    # If there's only one reactant, it's the starting material with no reagents
    starting_material = reactants[0] if reactants else ""
    reagents = ".".join(reactants[1:]) if len(reactants) > 1 else ""

    # Join multiple products with '.' (though typically there's only one)
    end_product = ".".join(products) if products else ""

    return {
        "metadata": {
            "reaction_id": entry.get("reaction_id", "")
        },
        "original_smiles": {
            "Starting Material": starting_material,
            "Reagents": reagents,
            "Reaction Conditions": "NA",
            "End Product": end_product
        }
    }


def convert_jsonl_to_json(input_path: Path, output_path: Path) -> None:
    """
    Convert a JSONL file to ForwardRxnParser JSON format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSON file
    """
    converted_entries = []

    print(f"Reading from {input_path}")
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                converted = convert_entry(entry)
                converted_entries.append(converted)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    print(f"Converted {len(converted_entries)} entries")
    print(f"Writing to {output_path}")

    with open(output_path, 'w') as f:
        json.dump(converted_entries, f, indent=2)

    print(f"Successfully wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert USPTO-50k JSONL files to ForwardRxnParser format"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file or directory (default: same as input with .json extension)"
    )

    args = parser.parse_args()

    input_path = args.input

    if input_path.is_file():
        # Single file conversion
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.with_suffix('.json')

        convert_jsonl_to_json(input_path, output_path)

    elif input_path.is_dir():
        # Directory conversion
        output_dir = args.output if args.output else input_path
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_files = list(input_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {input_path}")
            return

        print(f"Found {len(jsonl_files)} JSONL files to convert")

        for jsonl_file in jsonl_files:
            output_file = output_dir / jsonl_file.with_suffix('.json').name
            convert_jsonl_to_json(jsonl_file, output_file)
            print()
    else:
        print(f"Error: {input_path} is not a file or directory")
        return


if __name__ == "__main__":
    main()
