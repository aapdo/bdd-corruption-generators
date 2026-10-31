"""Run the source filter on an explicit chronological 5fps frame/object manifest."""
import argparse,hashlib,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).parent/'recipes'))
    from glare_eligibility import EligibilityTracker,summarize_segments,POLICY
    import numpy as np
    import cv2
    from PIL import Image
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if a.output.exists():raise ValueError('Refuse overwrite')
    data=json.loads(a.manifest.read_text());tracker=EligibilityTracker();records=[]
    for f in data['frames']:
        path=Path(f['path']);raw=path.read_bytes();image=np.asarray(Image.open(path).convert('RGB'))
        h,w=image.shape[:2];rgb=cv2.resize(image,(640,360),interpolation=cv2.INTER_CUBIC);objects=[]
        for o in f.get('objects',[]):
            obj=dict(o);obj['bbox']=(np.asarray(o['bbox'])*[640/w,360/h,640/w,360/h]).tolist();objects.append(obj)
        records.append(tracker.observe(rgb,objects,f['timestamp_s'],f['frame_index'],hashlib.sha256(raw).hexdigest(),data.get('attributes')))
    result=dict(video_id=data['video_id'],frames=records,segments=summarize_segments(records),policy=POLICY,
        training_release_ready=False,render_quality='NOT_RENDERED',manual_audit_required=True)
    with a.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
