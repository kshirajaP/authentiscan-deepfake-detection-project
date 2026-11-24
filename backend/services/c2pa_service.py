"""
C2PA (Content Provenance and Authenticity) Verification Service
Location: backend/services/c2pa_service.py

Uses the official c2pa-python SDK to read and validate real C2PA
manifests embedded in uploaded media. Files without a manifest
(the common case today) correctly report 'no_credentials' rather
than a simulated result.
"""
import json

try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False
    print("⚠️ c2pa-python not installed — C2PA verification will report "
          "'unavailable' for all files. Install with: pip install c2pa-python")


class C2PAService:
    """
    Verifies C2PA content credentials using the official c2pa-python SDK.
    """

    @staticmethod
    def verify_credentials(file_path):
        """
        Read and validate any C2PA manifest embedded in file_path.

        Returns a dict describing whether credentials were found and,
        if so, whether the embedded signature/hash validation passed.
        """
        if not C2PA_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'c2pa-python SDK is not installed on this server',
                'has_credentials': False,
                'is_valid': None,
                'provenance': None
            }

        try:
            with c2pa.Reader(file_path) as reader:
                manifest_json = reader.json()
            data = json.loads(manifest_json)

        except c2pa.C2paError:
            # This is the expected, common case: most media has no C2PA manifest.
            return {
                'status': 'no_credentials',
                'message': 'No C2PA content credentials found in this file',
                'has_credentials': False,
                'is_valid': None,
                'provenance': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'C2PA verification failed: {str(e)}',
                'has_credentials': False,
                'is_valid': False,
                'provenance': None
            }

        active_label = data.get('active_manifest')
        manifest = data.get('manifests', {}).get(active_label) if active_label else None

        if not manifest:
            return {
                'status': 'no_credentials',
                'message': 'No active C2PA manifest found in this file',
                'has_credentials': False,
                'is_valid': None,
                'provenance': None
            }

        # The SDK reports overall trust/validity here directly.
        validation_state = data.get('validation_state', 'Unknown')
        is_valid = validation_state.lower() in ('valid', 'trusted')

        claim_generators = manifest.get('claim_generator_info') or [{}]
        signature_info = manifest.get('signature_info', {}) or {}

        actions = []
        for assertion in manifest.get('assertions', []):
            if assertion.get('label', '').startswith('c2pa.actions'):
                for action in assertion.get('data', {}).get('actions', []):
                    actions.append({
                        'action': action.get('action'),
                        'software_agent': (action.get('softwareAgent') or {}).get('name')
                    })

        return {
            'status': 'valid' if is_valid else 'invalid',
            'message': (
                'C2PA credentials found and cryptographically valid'
                if is_valid else
                'C2PA credentials found, but signature/certificate validation failed '
                '(see validation_state for detail)'
            ),
            'has_credentials': True,
            'is_valid': is_valid,
            'validation_state': validation_state,
            'provenance': {
                'title': manifest.get('title'),
                'generator': claim_generators[0].get('name', 'Unknown'),
                'signature_issuer': signature_info.get('issuer', 'Unknown'),
                'signature_common_name': signature_info.get('common_name'),
                'signature_algorithm': signature_info.get('alg'),
                'actions': actions
            }
        }

    @staticmethod
    def get_provenance_chain(file_path):
        """
        Extract a simplified, chronological chain of actions from the
        C2PA manifest (if present) for display in the UI.
        """
        verification = C2PAService.verify_credentials(file_path)

        if not verification['has_credentials']:
            return {
                'status': 'no_chain',
                'message': 'No provenance information available',
                'chain': []
            }

        provenance = verification['provenance']
        chain = []
        for idx, action in enumerate(provenance.get('actions', []), start=1):
            chain.append({
                'step': idx,
                'action': action.get('action'),
                'actor': action.get('software_agent') or provenance.get('generator'),
                'issuer': provenance.get('signature_issuer')
            })

        if not chain:
            # Manifest present but no explicit action assertions parsed
            chain.append({
                'step': 1,
                'action': 'created',
                'actor': provenance.get('generator'),
                'issuer': provenance.get('signature_issuer')
            })

        return {
            'status': 'success',
            'is_valid': verification['is_valid'],
            'chain': chain
        }
