"""Local detected-vehicle motion, lamp candidates and separate source quotas."""
from collections import defaultdict
import os
from pathlib import Path
import cv2
import numpy as np
import hashlib


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):result.update(block)
    return result.hexdigest()

POLICY=dict(vehicle_detection_score=.4,vehicle_classes=[3,6,8],detector_batch=2,
    source_confirm_frames=2,minimum_lamp_observations=3,maximum_missing_frames=2,
    vehicle_light_slots=4,environment_light_slots=2,maximum_sources=6,
    lamp_admission='local_photometry_geometry_and_local_CLIP_track_votes',
    ranking='observed_contrast_times_sqrt_area_times_peak; per_vehicle_fairness',
    vehicle_context='local_COCO_detector_not_annotation_boxes',offline_track_representatives=True)


class LocalVehicles:
    def __init__(self,ready,request):
        if len(request['gpus'])!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=request['gpus'][0]:
            raise ValueError('One scheduler-owned GPU required')
        if sha(ready['path'])!=ready['sha256']:raise ValueError('Detector weight changed')
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.manual_seed(1101);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        if torch.cuda.device_count()!=1:raise ValueError('Unexpected GPU visibility')
        self.torch=torch;self.frames=0
        self.model=fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=720,max_size=1280,
            box_score_thresh=.4,box_detections_per_img=100)
        self.model.load_state_dict(torch.load(ready['path'],map_location='cpu',weights_only=True),strict=True)
        self.model=self.model.float().to('cuda').eval();self.weight_sha256=ready['sha256']

    def predict(self,images):
        t=self.torch
        tensors=[t.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).to('cuda',dtype=t.float32)/255 for rgb in images]
        with t.inference_mode():predictions=self.model(tensors)
        rows=[]
        for prediction in predictions:
            boxes=prediction['boxes'].cpu().numpy();scores=prediction['scores'].cpu().numpy();labels=prediction['labels'].cpu().numpy()
            rows.append([dict(box=b.tolist(),score=float(s),category=int(c))for b,s,c in zip(boxes,scores,labels)
                         if c in POLICY['vehicle_classes'] and s>=.4 and b[2]-b[0]>=18 and b[3]-b[1]>=14])
        self.frames+=len(images);return rows

    def stats(self):
        return dict(model='torchvision.fasterrcnn_resnet50_fpn.COCO_V1',weight_sha256=self.weight_sha256,
            frames=self.frames,API_calls=0,annotation_boxes_used=False,
            peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20)

    def close(self):del self.model;self.torch.cuda.empty_cache()


def iou(a,b):
    x0,y0=np.maximum(a[:2],b[:2]);x1,y1=np.minimum(a[2:],b[2:])
    inter=max(0,x1-x0)*max(0,y1-y0)
    return float(inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1))


class VehicleTracker:
    def __init__(self):self.tracks={};self.next_id=1;self.previous=None

    def prediction(self,gray,track):
        box=np.asarray(track['box'],np.float32)
        shift=np.asarray(track['velocity'],np.float32)*(track['gap']+1)
        if self.previous is not None and track['gap']==0:
            h,w=gray.shape;x0,y0,x1,y1=box.astype(int);mask=np.zeros_like(gray)
            mask[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=255
            points=cv2.goodFeaturesToTrack(self.previous,40,.01,4,mask=mask)
            if points is not None and len(points)>=3:
                nxt,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None)
                if nxt is not None:
                    back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None)
                    if back is not None:
                        keep=ok.ravel().astype(bool)&valid.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<2.5)
                        if keep.sum()>=3:shift=np.median((nxt-points).reshape(-1,2)[keep],axis=0)
        return box+np.r_[shift,shift]

    def step(self,rgb,detections):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);edges=[]
        for key,track in self.tracks.items():
            if track['gap']>2:continue
            predicted=self.prediction(gray,track)
            for index,detection in enumerate(detections):
                box=np.asarray(detection['box'],np.float32);overlap=iou(predicted,box)
                distance=np.linalg.norm((predicted[:2]+predicted[2:]-box[:2]-box[2:])/2)
                norm=max(15.,float(np.linalg.norm(box[2:]-box[:2])))
                if overlap>.08 and distance/norm<.65:edges.append((1-overlap+.2*distance/norm,key,index))
        used=set();matched=set();rows=[]
        for _,key,index in sorted(edges):
            if key in used or index in matched:continue
            detection=detections[index];track=self.tracks[key];box=np.asarray(detection['box'],np.float32)
            old=np.asarray(track['box'],np.float32)
            velocity=(box[:2]+box[2:]-old[:2]-old[2:])/2/(track['gap']+1)
            count=track['count']+1
            self.tracks[key]=dict(box=box.tolist(),velocity=velocity.tolist(),gap=0,count=count)
            rows.append(dict(detection,vehicle_id=key,observations=count));used.add(key);matched.add(index)
        for key,track in self.tracks.items():
            if key not in used:track['gap']+=1
        for index,detection in enumerate(detections):
            if index in matched:continue
            key=self.next_id;self.next_id+=1
            self.tracks[key]=dict(box=detection['box'],velocity=[0.,0.],gap=0,count=1)
            rows.append(dict(detection,vehicle_id=key,observations=1))
        self.previous=gray;return rows


def lamp_candidates(rgb,vehicle):
    h,w=rgb.shape[:2];box=vehicle['box'];x0,y0,x1,y1=map(int,np.rint(box))
    x0,y0=max(0,x0),max(0,y0);x1,y1=min(w,x1),min(h,y1)
    bw,bh=x1-x0,y1-y0
    if bw<18 or bh<14:return []
    value=rgb[y0:y1,x0:x1].astype(np.float32)/255;r,g,b=np.moveaxis(value,-1,0)
    lum=.2126*r+.7152*g+.0722*b
    neutral=(lum>.82)&(value.max(axis=2)>.94)
    red=(r>.78)&(r-g>.2)&(r-b>.2)
    count,labels,stats,_=cv2.connectedComponentsWithStats((neutral|red).astype(np.uint8),8)
    rows=[]
    for index in range(1,count):
        x,y,sw,sh,area=map(int,stats[index]);maximum=max(12,.055*bw*bh)
        if area<2 or area>maximum or max(sw,sh)/max(1,min(sw,sh))>6 or area/(sw*sh)<.25:continue
        cy,cx=np.nonzero(labels==index);is_red=bool(red[cy,cx].mean()>.5)
        signal=r if is_red else lum;radius=max(1.,float(np.sqrt(area/np.pi)))
        border=int(max(4,2.5*radius));a,c=max(0,x-border),min(bw,x+sw+border);d,e=max(0,y-border),min(bh,y+sh+border)
        ring=np.ones((e-d,c-a),bool);ring[y-d:y+sh-d,x-a:x+sw-a]=False
        ambient=float(np.median(signal[d:e,a:c][ring]))if ring.any()else 1.
        peak=float(np.quantile(signal[cy,cx],.8));contrast=peak-ambient
        if contrast<.12 or peak/(ambient+.02)<1.18:continue
        weight=np.maximum(signal[cy,cx]-ambient,.001)
        px=float(np.average(cx,weights=weight));py=float(np.average(cy,weights=weight))
        nx,ny=px/bw,py/bh
        # Side lamps and an upper central red brake lamp; exclude plate/roof/wheels.
        side=nx<=.40 or nx>=.60
        central_brake=is_red and .40<nx<.60 and .12<ny<.4
        plausible_height=(.25<=ny<=.78)if is_red else(.28<=ny<=.72)
        if not(plausible_height and side)and not central_brake:continue
        rows.append(dict(xy=[px+x0,py+y0],relative_xy=[nx,ny],radius=radius,area=area,
            color=value[cy,cx].mean(axis=0).tolist(),peak=peak,contrast=contrast,
            score=float(contrast*np.sqrt(area)*peak),is_red=is_red,paired=False,
            bbox=[x+x0,y+y0,x+x0+sw,y+y0+sh],vehicle_id=vehicle['vehicle_id'],vehicle_box=box,
            evidence='vehicle_associated_red_peak'if is_red else'vehicle_associated_white_peak',
            forward_backward_error_pixels=None,identity_verified=False))
    for a in rows:
        for b_ in rows:
            if a is b_ or a['is_red']!=b_['is_red']:continue
            ax,ay=a['relative_xy'];bx,by=b_['relative_xy']
            if abs(ax-bx)>.25 and abs(ay-by)<.18 and .15<a['area']/b_['area']<6:
                a['paired']=True
    rows.sort(key=lambda s:-(s['score']*(1.2 if s['paired']else 1)))
    return rows[:8]


class LampTracker:
    def __init__(self):self.tracks={};self.next_id=1

    def step(self,candidates):
        edges=[]
        for key,track in self.tracks.items():
            if track['gap']>2:continue
            for index,candidate in enumerate(candidates):
                if candidate['vehicle_id']!=track['last']['vehicle_id'] or candidate['is_red']!=track['last']['is_red']:continue
                distance=float(np.linalg.norm(np.asarray(candidate['relative_xy'])-track['last']['relative_xy']))
                if distance<.20:edges.append((distance,key,index))
        used=set();matched=set();rows=[]
        for distance,key,index in sorted(edges):
            if key in used or index in matched:continue
            candidate=candidates[index];track=self.tracks[key]
            streak=track['streak']+1 if track['gap']==0 else 1
            track.update(last=candidate,gap=0,streak=streak,count=track['count']+1)
            rows.append(dict(candidate,track_id=key,confirmed=streak>=2,consecutive=streak,
                             relative_association_error=distance))
            used.add(key);matched.add(index)
        for key,track in self.tracks.items():
            if key not in used:track['gap']+=1;track['streak']=0
        for index,candidate in enumerate(candidates):
            if index in matched:continue
            key='vehicle_lamp_'+str(self.next_id);self.next_id+=1
            self.tracks[key]=dict(last=candidate,gap=0,streak=1,count=1)
            rows.append(dict(candidate,track_id=key,confirmed=False,consecutive=1,relative_association_error=None))
        return rows


def select_sources(vehicle,environment,previous):
    def rank(s):return s['score']*(1.15 if s['track_id']in previous else 1.)
    groups=defaultdict(list)
    for source in sorted(vehicle,key=rank,reverse=True):groups[source['vehicle_id']].append(source)
    ordered=sorted(groups.values(),key=lambda v:rank(v[0]),reverse=True)
    chosen=[]
    for layer in range(2):
        for group in ordered:
            if len(group)>layer and len(chosen)<4:chosen.append(group[layer])
    chosen.extend(sorted(environment,key=rank,reverse=True)[:2])
    return chosen


def automatic_vehicle_sources(images,detections,old_rows,classifier,report_progress=None):
    vehicles=VehicleTracker();lamps=LampTracker();frames=[];groups=defaultdict(list)
    for i,(rgb,detected)in enumerate(zip(images,detections)):
        boxes=vehicles.step(rgb,detected);candidates=[]
        for vehicle in boxes:candidates.extend(lamp_candidates(rgb,vehicle))
        tracked=lamps.step(candidates)
        frames.append(dict(vehicle_boxes=boxes,vehicle_candidates=tracked))
        for source in tracked:groups[source['track_id']].append((i,source))
        if report_progress and i%20==0:report_progress('vehicle_relative_lamp_tracking',frame=i)
    vehicle_groups=defaultdict(list)
    for i,frame in enumerate(frames):
        for vehicle in frame['vehicle_boxes']:vehicle_groups[vehicle['vehicle_id']].append((i,vehicle))
    context_keys=[];context_queries=[]
    for key,rows in vehicle_groups.items():
        if len(rows)<3:continue
        for j in sorted(set(np.linspace(0,len(rows)-1,3).round().astype(int).tolist())):
            i,vehicle=rows[j];context_keys.append((key,i));context_queries.append((images[i],vehicle['box']))
    context_results=[]
    for start in range(0,len(context_queries),64):
        context_results.extend(classifier.score_vehicle_boxes(context_queries[start:start+64]))
        if report_progress:report_progress('whole_vehicle_context_check',queries_done=min(start+64,len(context_queries)),total_queries=len(context_queries))
    context_decisions=defaultdict(list)
    for (key,i),score in zip(context_keys,context_results):context_decisions[key].append(dict(frame=i,**score))
    supported_vehicles={key for key,rows in context_decisions.items()if sum(r['accepted']for r in rows)>=2}
    queries=[];keys=[]
    for key,rows in groups.items():
        if len(rows)<3 or rows[0][1]['vehicle_id']not in supported_vehicles:continue
        for j in sorted(set(np.linspace(0,len(rows)-1,3).round().astype(int).tolist())):
            i,source=rows[j];keys.append((key,i));queries.append((images[i],source))
    scores=[]
    for start in range(0,len(queries),64):
        scores.extend(classifier.score_queries(queries[start:start+64]))
        if report_progress:report_progress('local_vehicle_lamp_semantics',queries_done=min(start+64,len(queries)),total_queries=len(queries))
    decisions=defaultdict(list)
    for (key,i),score in zip(keys,scores):
        source=next(s for frame,s in groups[key]if frame==i)
        generic=score['positive_cosine']>=.21 and score['margin']>=.002 and score['minimum_scale_margin']>=-.025
        # Coarse nearest-text labels are unreliable for tiny lamps. Vehicle context,
        # current compact peak and lamp geometry establish the source association.
        single=score['margin']>=.008 and score['center_delta']>=-.01
        supported=bool(generic and(source['paired']or single))
        decisions[key].append(dict(frame=i,score=score,supported=supported,paired=source['paired']))
    accepted={key for key,rows in decisions.items()if sum(s['supported']for s in rows)>=2}
    previous=set();env_streak={};output=[]
    for i,frame in enumerate(frames):
        vehicle=[]
        for source in frame['vehicle_candidates']:
            key=source['track_id']
            if key in accepted and source['confirmed']:
                vehicle.append(dict(source,fade=min(1.,(source['consecutive']-1)/3),source_group='vehicle',
                    automatic_source=dict(accepted=True,positive_type='taillight'if source['is_red']else'headlight',
                        basis='local_vehicle_detection_photometry_geometry_track_CLIP_votes',human_identity_verified=False)))
        environment=[];old=old_rows[i]['tracking'];semantics={q['track_id']:q for q in old['semantic_candidates']};seen=set()
        for source in old['sources']:
            x,y=source['xy'];q=semantics.get(source['track_id'])
            inside=any(b['box'][0]<=x<=b['box'][2] and b['box'][1]<=y<=b['box'][3]for b in frame['vehicle_boxes'])
            key='environment_'+str(source['track_id'])
            if not inside and y<images[i].shape[0]*.85 and source['confirmed'] and q and q['accepted']:
                seen.add(key);env_streak[key]=env_streak.get(key,0)+1
                if env_streak[key]>=2:
                    environment.append(dict(source,track_id=key,fade=min(1.,(env_streak[key]-1)/3),
                        source_group='environment',automatic_source=q))
        env_streak={k:v for k,v in env_streak.items()if k in seen}
        chosen=select_sources(vehicle,environment,previous);previous={s['track_id']for s in chosen}
        output.append(dict(frame,active_sources=chosen,eligible_vehicle_sources=vehicle,
            eligible_environment_sources=environment,effect_applied=bool(chosen),manual_anchors_used=False))
    summary=dict(total_frames=len(output),active_frames=sum(bool(f['active_sources'])for f in output),
        vehicle_light_frames=sum(any(s['source_group']=='vehicle'for s in f['active_sources'])for f in output),
        mixed_source_frames=sum({s['source_group']for s in f['active_sources']}=={'vehicle','environment'}for f in output),
        accepted_vehicle_lamp_tracks=sorted(accepted),vehicle_lamp_track_candidates=len(groups),
        supported_vehicle_context_ids=sorted(supported_vehicles),whole_vehicle_context_decisions=dict(context_decisions),
        maximum_simultaneous_sources=max((len(f['active_sources'])for f in output),default=0),
        policy=POLICY,manual_anchors_used=False,annotation_boxes_used=False)
    return output,summary,dict(decisions)
