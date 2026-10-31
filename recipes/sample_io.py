"""A sample is visible only after image, metadata and a final hash-bound READY marker."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    p=Path(path)
    return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))


def checked(record):
    p=Path(record['path'])
    if not p.is_absolute()or p.is_symlink()or not p.is_file()or p.stat().st_size!=record['bytes']or sha(p)!=record['sha256']:
        raise ValueError('Artifact hash, size or path differs')
    return p


def write_sample(output,rgb,metadata):
    output=Path(output);meta=output.with_suffix('.json');ready=output.with_suffix('.READY.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix.lower()not in('.jpg','.jpeg'):raise ValueError('JPEG output required')
    temporaries=[]
    with output.with_suffix('.sample.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(p.exists()for p in [output,meta,ready]):raise ValueError('Refuse overwrite; use a new sample revision')
        def temporary(suffix):
            fd,name=tempfile.mkstemp(prefix='.'+output.stem+'.',suffix=suffix,dir=output.parent)
            os.fchmod(fd,0o640);os.close(fd);p=Path(name);temporaries.append(p);return p
        try:
            image_tmp=temporary('.jpg');Image.fromarray(rgb).save(image_tmp,format='JPEG',quality=95,subsampling=0)
            with image_tmp.open('rb')as f:os.fsync(f.fileno())
            image=dict(file=output.name,bytes=image_tmp.stat().st_size,sha256=sha(image_tmp))
            attributes=metadata(image_tmp)if callable(metadata)else metadata
            payload=dict(attributes,generation_state='GENERATED',training_release_ready=False,image=image)
            metadata_tmp=temporary('.json');metadata_tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
            with metadata_tmp.open('rb')as f:os.fsync(f.fileno())
            completed=dict(schema='corruption-sample-ready-v1',generation_state='GENERATED',image=image,
                metadata=dict(file=meta.name,bytes=metadata_tmp.stat().st_size,sha256=sha(metadata_tmp)),training_release_ready=False)
            ready_tmp=temporary('.READY.json');ready_tmp.write_text(json.dumps(completed,indent=2,sort_keys=True)+'\n')
            with ready_tmp.open('rb')as f:os.fsync(f.fileno())
            # Hard-link publication is exclusive. A failed second link leaves no READY marker.
            os.link(image_tmp,output);os.link(metadata_tmp,meta);os.link(ready_tmp,ready)
            directory=os.open(output.parent,os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
            return ref(ready)
        finally:
            for p in temporaries:
                if p.exists():p.unlink()


def read_completed(ready):
    ready=Path(ready)
    if ready.is_symlink():raise ValueError('READY marker must be a regular sample file')
    manifest=json.loads(ready.read_text())
    if manifest.get('schema')!='corruption-sample-ready-v1':raise ValueError('Completed sample marker required')
    for field in ['image','metadata']:
        r=manifest[field];name=Path(r['file'])
        if name.name!=r['file']or name.is_absolute():raise ValueError('Unsafe sample member')
        checked(dict(path=str((ready.parent/name).absolute()),bytes=r['bytes'],sha256=r['sha256']))
    payload=json.loads((ready.parent/manifest['metadata']['file']).read_text())
    if payload['image']!=manifest['image']:raise ValueError('Image/metadata binding differs')
    return dict(payload,sample_ready=ref(ready))
