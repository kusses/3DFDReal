import hashlib
import os

def hash_file(filepath, block_size=65536):
    """SHA256 hash"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def hash_directory(directory, exts=None):
    """
    all hashs in directory
    exts: ['.py', '.npy', '.txt'] setting possible
    """
    hash_list = []
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            if exts is None or any(fname.endswith(ext) for ext in exts):
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    hash_list.append(hash_file(fpath))
    combined = ''.join(sorted(hash_list))
    return hashlib.sha256(combined.encode()).hexdigest()
