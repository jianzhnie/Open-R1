from transformers.utils.import_utils import _is_package_available

# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available('e2b')


def is_e2b_available() -> bool:
    return _e2b_available
