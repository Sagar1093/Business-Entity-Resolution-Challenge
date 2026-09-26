"""Stage wrapper: S3-H address-token blocking."""
from .. import blocking_h


def run(cfg, force: bool = False):
    return blocking_h.run(cfg, force=force)
