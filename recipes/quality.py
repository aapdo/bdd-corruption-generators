"""Immutable quality receipts; creation and CPU tests never grant dataset release."""
import json
from pathlib import Path
from sample_io import checked,read_completed,ref

STATES=('GENERATED','STATIC_PASS','TEMPORAL_PASS','RELEASED','REJECTED')
REQUIRED={'STATIC_PASS':('static_visual','physical_eligibility'),
          'TEMPORAL_PASS':('temporal_consistency',),
          'RELEASED':('annotation_validity','split_integrity','detector_effect','adapter_recovery','protocol')}


def advance_quality(previous,target,evidence,output):
    raw=json.loads(Path(previous).read_text())
    if raw.get('schema')=='corruption-sample-ready-v1':
        sample=read_completed(previous);state='GENERATED';ready=sample['sample_ready'];history=[]
    else:
        state=raw['quality_state'];ready=raw['sample_ready'];sample=read_completed(checked(ready));history=list(raw['evidence'])
        seen=set()
        for r in history:
            report=json.loads(checked(r).read_text());seen.add(report.get('kind'))
            if report.get('passed')is not True or report.get('image_sha256')!=sample['image']['sha256']:raise ValueError('Prior evidence does not pass for this image')
        if state not in STATES:raise ValueError('Unknown quality state')
        inherited=set(sum((list(REQUIRED[s])for s in STATES[1:STATES.index(state)+1]if s in REQUIRED),[]))
        if not inherited<=seen:raise ValueError('Prior quality stages have no complete evidence')
    if target not in STATES or state in('RELEASED','REJECTED')or(target!='REJECTED'and STATES.index(target)!=STATES.index(state)+1):
        raise ValueError('Invalid quality state transition')
    needed=('rejection_reason',)if target=='REJECTED'else REQUIRED[target]
    if set(evidence)!=set(needed):raise ValueError('Explicit quality, label, performance and split evidence required')
    for kind,path in evidence.items():
        r=ref(path);report=json.loads(checked(r).read_text())
        if report.get('kind')!=kind or report.get('image_sha256')!=sample['image']['sha256']or report.get('passed')is not True:
            raise ValueError('Quality evidence must pass and bind the exact generated image')
        history.append(dict(r,kind=kind))
    result=dict(schema='corruption-quality-receipt-v1',quality_state=target,training_release_ready=target=='RELEASED',
        sample_ready=ready,previous=ref(previous),evidence=history)
    with Path(output).open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result


def read_released(receipt):
    record=json.loads(Path(receipt).read_text())
    if record.get('schema')!='corruption-quality-receipt-v1'or record.get('quality_state')!='RELEASED'or record.get('training_release_ready')is not True:
        raise ValueError('Sample is not RELEASED for training/evaluation')
    sample=read_completed(checked(record['sample_ready']));seen=set()
    for r in record['evidence']:
        evidence=json.loads(checked(r).read_text());seen.add(evidence.get('kind'))
        if evidence.get('image_sha256')!=sample['image']['sha256']or evidence.get('passed')is not True:raise ValueError('Release evidence mismatch')
    if not set(sum((list(v)for v in REQUIRED.values()),[]))<=seen:raise ValueError('Incomplete release evidence')
    return sample
