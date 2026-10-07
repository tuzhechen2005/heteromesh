"""Strict v1 capability and fragment manifests, with no executable fields."""
from importlib.resources import files
import json
from jsonschema import Draft202012Validator
from .protocol import ProtocolError, DTYPE_BYTES, canonical_json

class ManifestError(ValueError):
    pass


def _validate(data,kind):
    try:
        canonical_json(data)
        schema=json.loads(files('heteromesh.schemas').joinpath(kind+'.json').read_text())
        errors=list(Draft202012Validator(schema).iter_errors(data))
        if errors:
            raise ManifestError(f'{kind}: {errors[0].message}')
    except ProtocolError as exc:
        raise ManifestError(str(exc)) from exc


def validate_capabilities(data: dict) -> dict:
    _validate(data,'capabilities')
    if data['memory']['unified'] and data['memory']['accelerator_budget_bytes'] is not None:
        raise ManifestError('unified memory must not duplicate accelerator budget')
    for field in ('wire_dtypes','compute_dtypes'):
        if not set(data[field])<=set(DTYPE_BYTES):
            raise ManifestError('unknown dtype capability')
    return data


def validate_manifest(data: dict) -> dict:
    _validate(data,'manifest')
    identifiers=[x['id'] for x in data['fragments']]
    if len(identifiers)!=len(set(identifiers)): raise ManifestError('duplicate fragment ID')
    for fragment in data['fragments']:
        for field in ('inputs','outputs'):
            names=[x['name'] for x in fragment[field]]
            if len(names)!=len(set(names)): raise ManifestError('duplicate tensor name')
    return data


def validate_fragments(fragments):
    """Apply the same fragment schema outside a whole-model manifest."""
    validate_manifest({'protocol_version':1,'model_id':'validation','revision':'validation',
                       'source':'local:validation','license':'validation','profile':'validation',
                       'fragments':fragments})
