"""Shared checkpoint identity, chemistry, and durable-output helpers."""
import hashlib
import json
import os
import re
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(path):
    """Verify the final consolidated checkpoint, never a partial trainer save."""
    path = Path(path)
    assert (path / 'config.json').is_file(), path
    index = path / 'model.safetensors.index.json'
    files = (set(json.loads(index.read_text())['weight_map'].values())
             if index.exists() else {'model.safetensors'})
    for name in files:
        assert (path / name).is_file() and (path / name).stat().st_size > 0, path / name
    assert (path / 'tokenizer_config.json').is_file(), path
    return path


def iter_jsonl(path, repair=False):
    """Reject corruption except for an interrupted, unterminated final write."""
    path = Path(path)
    if not path.exists():
        return
    mode = 'r+b' if repair else 'rb'
    with path.open(mode) as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    f.truncate(offset)
                    break
                raise ValueError(f'Incomplete final row in {path}')
            yield json.loads(line)


def read_jsonl(path, repair=False):
    return list(iter_jsonl(path, repair=repair))


def canonical(smiles):
    from rdkit import Chem
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def prediction(text):
    # The generation prompt contains the opening think tag. Never score an
    # intermediate answer inside reasoning or an unfinished thought as final.
    if '</think>' not in text:
        return None
    final = text.rsplit('</think>', 1)[1].strip()
    for tag in ('answer', 'product', 'forward_reaction'):
        values = re.findall(fr'<{tag}>\s*(.*?)\s*</{tag}>', final, re.S)
        if values:
            return canonical(values[-1].strip())
    return canonical(final)


def similarity(pred, gold):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    if pred is None or gold is None:
        return 0.0
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(DataStructs.TanimotoSimilarity(
        fp.GetFingerprint(Chem.MolFromSmiles(pred)), fp.GetFingerprint(Chem.MolFromSmiles(gold))))


def records(path, prompts):
    raw = json.loads(Path(path).read_text())
    result = []
    for idx, row in enumerate(raw):
        smi = row['original_smiles']
        gold = canonical(smi['End Product'])
        assert gold is not None, f'Invalid ground truth at row {idx}'
        result.append({'row_idx': idx, 'ground_truth': gold, 'messages': [
            {'role': 'system', 'content': prompts['system']},
            {'role': 'user', 'content': prompts['user'].format(
                starting_material=smi['Starting Material'],
                reagent_list=smi.get('Reagents') or 'No reagents').strip()}]})
    return result
