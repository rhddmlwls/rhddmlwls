"""Content-addressed research caches with atomic files and checksum manifests."""
from hashlib import sha256
import json
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def _replace_with_retry(temporary,path):
    """Keep atomic replacement when Windows briefly locks a cache reader.

    Only retry PermissionError; persistent permission failures still surface.
    Never remove the old completed file to work around a transient reader.
    """
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(.05*2**attempt)


def fingerprint(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def frame_digest(frame):
    return fingerprint(dict(columns=[str(c) for c in frame.columns],
        dtypes=[str(t) for t in frame.dtypes],
        values=sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()))


def array_digest(values):
    array=np.ascontiguousarray(values)
    return fingerprint(dict(shape=array.shape,dtype=str(array.dtype),bytes=sha256(array.tobytes()).hexdigest()))


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    _replace_with_retry(temporary,path)


def atomic_npz(path,**arrays):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,**arrays)
    _replace_with_retry(temporary,path)


def read_manifest(folder,key,required):
    folder=Path(folder)
    try:
        metadata=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        files=metadata["files"]
        if metadata["schema"]!=1 or metadata["key"]!=key or set(files)!=set(required):
            return None
        for name,digest in files.items():
            if Path(name).name!=name or file_digest(folder/name)!=digest:
                return None
        return metadata
    except (OSError,ValueError,KeyError,TypeError):
        return None


def commit_manifest(folder,key,files,**metadata):
    folder=Path(folder)
    atomic_json(folder/"manifest.json",dict(schema=1,key=key,
        files={name:file_digest(folder/name) for name in files},**metadata))


def save_frame(path,frame,index=True):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{uuid4().hex}.tmp")
    frame.to_csv(temporary,index=index,index_label="datetime" if index else None)
    _replace_with_retry(temporary,path)
