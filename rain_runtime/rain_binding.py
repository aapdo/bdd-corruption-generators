"""Pure-JSON binding gate shared by the isolated Rain backend and CPU tests."""
def validate_render_binding(simulation,inputs,diagnostic=False):
    n=len(inputs['samples']);expected=list(range(n));qa=simulation.get('SE3_correction')
    if inputs.get('render_exposure_indices',expected)!=expected:
        raise ValueError('Backend renders a prefix: subset/reindex XML for non-prefix exposure selection')
    if qa is None:
        if diagnostic:return dict(mode='DIAGNOSTIC',uncorrected_original_XML=True,training_release_ready=False)
        raise ValueError('Strict Rain rendering requires a hash-bound correction report')
    if qa.get('corrected_XML_sha256')!=simulation.get('sha256')or not simulation.get('sha256'):
        raise ValueError('Correction report and actual rendered XML hash differ')
    if sorted(qa.get('render_exposure_indices',[]))!=expected or sorted(qa.get('corrected_indices',[]))!=expected:
        raise ValueError('Actual render exposures are not completely corrected')
    mapping={m['xml_index']:m for m in qa.get('source_to_XML_mapping',[])}
    if set(mapping)!=set(expected):raise ValueError('Missing render/source exposure mapping')
    for i,sample in enumerate(inputs['samples']):
        item=mapping[i]
        if item.get('frame_id')!=sample['image_id']:raise ValueError('Correction belongs to a different source frame')
        if 'timestamp_s'in sample and item.get('timestamp_s')!=sample['timestamp_s']:raise ValueError('Correction/source timestamp mismatch')
    if not diagnostic and qa.get('mode')!='NUMERICAL_MOTION_PROFILE_PASS':raise ValueError('Diagnostic motion is not qualified for strict rendering')
    return dict(mode=qa['mode'],corrected_exposures=n,calibrated_camera=False,training_release_ready=False)
