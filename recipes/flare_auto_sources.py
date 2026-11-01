"""Offline local CLIP source decisions; no manual coordinates or remote inference."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

POSITIVE={
    'streetlight':['a close-up photo of a glowing street lamp bulb at night','a bright street light shining against a dark background'],
    'headlight':['a close-up photo of a shining car headlight','a bright vehicle headlamp emitting light'],
    'taillight':['a close-up photo of an illuminated red car brake light','a glowing red vehicle tail lamp'],
    'traffic_light':['a close-up photo of an illuminated traffic signal lamp','a glowing red or green traffic light'],
    'sun':['a dazzling bright sun in the sky','a small overexposed sun shining through the sky'],
    'specular_glint':['a tiny intense specular glint of sunlight on metal','a dazzling concentrated reflection of light on glass'],
    'illuminated_sign':['a close-up photo of an illuminated shop sign at night','a glowing lightbox sign emitting light'],
}
NEGATIVE={
    'road_paint':['white arrows painted on dark asphalt','white letters and lane markings painted on a road'],
    'car_paint':['a close-up photo of the painted side door of a car','orange painted bodywork of a yellow taxi'],
    'white_surface':['a flat white painted wall in sunlight','a white truck panel or white painted vehicle body'],
    'daytime_sky_gap':['bright daytime sky visible between tree leaves','a bright hole in foliage showing the sky'],
    'unlit_sign':['printed white letters on a street sign','a flat unlit advertising billboard'],
    'window_surface':['ordinary unlit building windows in daylight','a flat building facade reflecting daylight'],
    'road_texture':['a close-up photo of textured asphalt','a diffuse patch of light on a road surface'],
    'unlit_reflector':['an unlit red plastic car reflector','a red painted surface without an illuminated lamp'],
}
POLICY=dict(positive_cosine=.22,margin=.01,center_delta=.001,minimum_scale_margin=-.005,
    track_representatives=3,track_minimum_votes=2,source_confirmation_frames=3,maximum_sources_per_frame=3,
    score_interpretation='relative_cosine_similarity_not_calibrated_probability',thresholds='uncalibrated_pilot')


def crops_for_source(rgb,source):
    """Centered tight/context crops plus context with the central peak removed."""
    h,w=rgb.shape[:2];x,y=source['xy'];radius=source['radius'];image=Image.fromarray(rgb)
    views=[];boxes=[]
    for side in [np.clip(4*radius,40,144),np.clip(9*radius,88,288)]:
        side=int(round(side));x0=int(round(x-side/2));y0=int(round(y-side/2))
        x0=max(0,min(w-side,x0));y0=max(0,min(h-side,y0));box=(x0,y0,x0+side,y0+side)
        views.append(image.crop(box));boxes.append(box)
    masked=views[1].copy();array=np.asarray(masked);color=tuple(np.median(array.reshape(-1,3),axis=0).astype(int).tolist())
    ox,oy=x-boxes[1][0],y-boxes[1][1];r=max(3.,1.4*radius)
    ImageDraw.Draw(masked).ellipse((ox-r,oy-r,ox+r,oy+r),fill=color)
    return [*views,masked]


def decision_from_scores(tight,context,masked,positive_names,negative_names):
    average=(np.asarray(tight)+context)/2;count=len(positive_names)
    pos_index=int(np.argmax(average[:count]));neg_index=int(np.argmax(average[count:]))
    pos=float(average[pos_index]);neg=float(average[count+neg_index]);margin=pos-neg
    delta=float(context[pos_index]-masked[pos_index])
    scale_margin=min(float(np.max(row[:count])-np.max(row[count:]))for row in [tight,context])
    reasons=[]
    if pos<POLICY['positive_cosine']:reasons.append('weak_positive_similarity')
    if margin<POLICY['margin']:reasons.append('negative_context_competes')
    if delta<POLICY['center_delta']:reasons.append('center_not_responsible_for_source_score')
    if scale_margin<POLICY['minimum_scale_margin']:reasons.append('crop_scales_disagree')
    return dict(accepted=not reasons,positive_type=positive_names[pos_index],negative_type=negative_names[neg_index],
        positive_cosine=pos,negative_cosine=neg,margin=margin,center_delta=delta,minimum_scale_margin=scale_margin,
        reasons=reasons,score_is_probability=False,human_identity_verified=False)


class LocalClipSources:
    def __init__(self,ready,request):
        if len(request['gpus'])!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=request['gpus'][0]:
            raise ValueError('One scheduler-owned GPU required for local CLIP')
        root=Path(ready['path'])
        for name,ref in ready['files'].items():
            p=root/name
            if p.stat().st_size!=ref['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('Local CLIP file binding differs')
        import torch
        from transformers import CLIPConfig,CLIPModel,CLIPImageProcessor,CLIPTokenizerFast
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.manual_seed(1101);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        if torch.cuda.device_count()!=1:raise ValueError('Unexpected GPU visibility')
        self.torch=torch;self.positive_names=list(POSITIVE);self.negative_names=list(NEGATIVE)
        self.processor=CLIPImageProcessor.from_pretrained(root,local_files_only=True)
        tokenizer=CLIPTokenizerFast.from_pretrained(root,local_files_only=True)
        self.model=CLIPModel(CLIPConfig.from_pretrained(root,local_files_only=True))
        state=torch.load(root/'pytorch_model.bin',map_location='cpu',weights_only=True)
        state={k:v for k,v in state.items()if not k.endswith('.position_ids')}
        missing,unexpected=self.model.load_state_dict(state,strict=False)
        if missing or unexpected:raise ValueError('CLIP architecture/weight mismatch: '+str((missing,unexpected)))
        del state;self.model=self.model.float().to('cuda').eval()
        prompts=[];spans=[]
        for values in [*POSITIVE.values(),*NEGATIVE.values()]:
            spans.append((len(prompts),len(prompts)+len(values)));prompts.extend(values)
        tokens=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True).to('cuda')
        with torch.inference_mode():
            features=self.model.get_text_features(**tokens);features=features/features.norm(dim=-1,keepdim=True)
            labels=torch.stack([features[a:b].mean(0)for a,b in spans]);self.labels=labels/labels.norm(dim=-1,keepdim=True)
        self.queries=0;self.images=0;self.revision=ready['revision']
        self.prompt_sha256=hashlib.sha256(json.dumps([POSITIVE,NEGATIVE],sort_keys=True).encode()).hexdigest()

    def score_queries(self,queries):
        if not queries:return []
        crops=[image for rgb,source in queries for image in crops_for_source(rgb,source)]
        matrices=[];t=self.torch
        for start in range(0,len(crops),32):
            inputs=self.processor(images=crops[start:start+32],return_tensors='pt')
            with t.inference_mode():
                features=self.model.get_image_features(pixel_values=inputs['pixel_values'].to('cuda',dtype=t.float32))
                features=features/features.norm(dim=-1,keepdim=True)
                matrices.append((features@self.labels.T).cpu().numpy())
        scores=np.concatenate(matrices).reshape(len(queries),3,-1)
        if not np.isfinite(scores).all():raise ValueError('Non-finite local CLIP cosine scores')
        self.queries+=len(queries);self.images+=len(crops)
        return [decision_from_scores(a,b,c,self.positive_names,self.negative_names)for a,b,c in scores]

    def stats(self):
        return dict(model='openai/clip-vit-base-patch32',revision=self.revision,prompt_sha256=self.prompt_sha256,
            source_queries=self.queries,local_image_encodings=self.images,model_API_calls=0,images_uploaded=0,
            policy=POLICY,peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=self.torch.cuda.max_memory_reserved()/2**20)

    def close(self):
        del self.model;del self.labels;self.torch.cuda.empty_cache()


def automatic_tracks(tracker,images,classifier):
    """Whole-clip automatic selection, followed by current-frame semantic support."""
    groups=defaultdict(list)
    for i,row in enumerate(tracker.observations):
        for source in row['sources']:
            if source['confirmed'] and tracker.tracks[source['track_id']]['count']>=tracker.cfg.minimum_observations:
                groups[source['track_id']].append((i,source))
    cache={};representatives={};queries=[];keys=[]
    for key,rows in groups.items():
        if len(rows)<3:continue
        positions=sorted(set(np.linspace(0,len(rows)-1,3).round().astype(int).tolist()))
        representatives[key]=[(rows[j][0],key)for j in positions]
        for j in positions:
            i,source=rows[j];keys.append((i,key));queries.append((images[i],source))
    for key,score in zip(keys,classifier.score_queries(queries)):cache[key]=score
    accepted=set();decisions=[];ranks={}
    for key,refs in representatives.items():
        scores=[cache[ref]for ref in refs];votes=sum(s['accepted']for s in scores)
        margin=float(np.median([s['margin']for s in scores]));passed=votes>=2 and margin>=POLICY['margin']
        if passed:accepted.add(key);ranks[key]=margin
        decisions.append(dict(track_id=key,accepted=passed,positive_votes=votes,median_margin=margin,
            sample_frames=[i for i,_ in refs],samples=scores,
            decision_basis='local_CLIP_multiscale_center_evidence_and_temporal_votes'))
    remaining=[];remaining_keys=[]
    for key in accepted:
        for i,source in groups[key]:
            if (i,key)not in cache:remaining_keys.append((i,key));remaining.append((images[i],source))
    for key,score in zip(remaining_keys,classifier.score_queries(remaining)):cache[key]=score
    counters={};previous_selected=set();observations=[];union=set()
    for i,frame in enumerate(tracker.observations):
        supported=[];diagnostics=[];present=set()
        for source in frame['sources']:
            key=source['track_id'];score=cache.get((i,key))
            if score is not None:diagnostics.append(dict(score,track_id=key,xy=source['xy']))
            hold=bool(score and counters.get(key,0)>0 and score['margin']>0 and score['positive_cosine']>.21 and score['center_delta']>=0)
            if key in accepted and source['confirmed'] and score and(score['accepted']or hold):
                count=counters.get(key,0)+1;counters[key]=count;present.add(key)
                if count>=3:
                    supported.append(dict(source,fade=min(1.,(count-2)/3),automatic_source=score))
        counters={key:value for key,value in counters.items()if key in present}
        supported.sort(key=lambda s:-(ranks[s['track_id']]+(.005 if s['track_id']in previous_selected else 0)))
        active=supported[:POLICY['maximum_sources_per_frame']];previous_selected={s['track_id']for s in active};union|=previous_selected
        observations.append(dict(frame,semantic_candidates=diagnostics,active_sources=active,effect_applied=bool(active),
            status='automatic_source_supported'if active else'no_supported_automatic_source',manual_anchors_used=False,
            source_identity_semantically_verified=False))
    summary=dict(selected_track_ids=sorted(union),automatically_accepted_tracks=sorted(accepted),
        eligible_tracks=len(representatives),total_track_ids=len(tracker.tracks),active_frames=sum(r['effect_applied']for r in observations),
        total_frames=len(observations),tracking='local_LK_FB_current_peak_and_local_CLIP_support',
        maximum_simultaneous_sources=max((len(r['active_sources'])for r in observations),default=0),
        manual_anchors_used=False,online_causal_selection=False,automatic_identity_human_validated=False)
    return observations,summary,decisions

