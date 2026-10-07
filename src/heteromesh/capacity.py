"""Conservative static sequential placement; estimates are not measured claims."""
from .manifests import ManifestError, validate_capabilities, validate_fragments
from .protocol import DTYPE_BYTES
from math import prod

class CapacityError(ValueError):
    pass

RUNTIME_FIELDS=('peak_activations_bytes','workspace_bytes','transfer_buffers_bytes',
                'runtime_overhead_bytes','safety_margin_bytes')


def _peak(estimates):
    if not estimates: return 0
    resident=sum(e['resident_weights_bytes'] for e in estimates)
    for e in estimates:
        if e['loading_peak_bytes'] < e['resident_weights_bytes']:
            raise CapacityError('loading peak below resident weights')
    running=resident+max(sum(e[k] for k in RUNTIME_FIELDS) for e in estimates)
    loading=max(resident-e['resident_weights_bytes']+e['loading_peak_bytes'] for e in estimates)
    return max(running,loading)


def validate_placement(capabilities: dict, fragments: list[dict]) -> dict:
    """All fragment weights stay resident; temporary execution is sequential.

    loading_peak_bytes includes a fragment's own weights and loading buffers.
    Loading happens sequentially while other assigned weights may be resident.
    Unified host/accelerator estimates are added, never treated as extra capacity.
    """
    try:
        validate_capabilities(capabilities);validate_fragments(fragments)
    except ManifestError as exc: raise CapacityError(str(exc)) from exc
    for fragment in fragments:
        estimates=[fragment['memory']['host']]
        if fragment['memory']['accelerator'] is not None: estimates.append(fragment['memory']['accelerator'])
        if sum(x['resident_weights_bytes'] for x in estimates)<sum(x['bytes'] for x in fragment['weights']):
            raise CapacityError('resident estimate smaller than declared weights')
        boundary_bytes=sum(prod(t['shape'])*DTYPE_BYTES[t['dtype']] for t in fragment['inputs']+fragment['outputs'])
        if sum(x['peak_activations_bytes'] for x in estimates)<boundary_bytes:
            raise CapacityError('activation estimate smaller than live input/output tensors')
        if fragment['backend']!=capabilities['backend']: raise CapacityError('backend mismatch')
        if fragment['compute_dtype'] not in capabilities['compute_dtypes']: raise CapacityError('compute dtype unsupported')
        if not set(fragment['required_ops'])<=set(capabilities['supported_ops']): raise CapacityError('operator unsupported')
        for tensor in fragment['inputs']+fragment['outputs']:
            if tensor['dtype'] not in capabilities['wire_dtypes']: raise CapacityError('wire dtype unsupported')
            if tensor['dtype']!=fragment['compute_dtype']:
                raise CapacityError('dtype conversion requires an explicit separate fragment')
    memory=capabilities['memory']
    if memory['unified']:
        estimates=[]
        for fragment in fragments:
            host=fragment['memory']['host']; gpu=fragment['memory']['accelerator']
            estimates.append({k:host[k]+(gpu[k] if gpu else 0) for k in host})
        host_peak=_peak(estimates);gpu_peak=0
    else:
        host_peak=_peak([f['memory']['host'] for f in fragments])
        gpu_peak=_peak([f['memory']['accelerator'] for f in fragments if f['memory']['accelerator'] is not None])
    for peak,budget in [(host_peak,memory['host_budget_bytes']),(gpu_peak,memory['accelerator_budget_bytes'])]:
        if peak and (budget is None or peak>budget): raise CapacityError('insufficient or unknown memory budget')
    if memory['host_budget_bytes'] is None: raise CapacityError('unknown host budget')
    return {'host_peak_bytes':host_peak,'accelerator_peak_bytes':gpu_peak}
