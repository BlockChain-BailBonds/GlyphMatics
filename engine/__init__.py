"""Compatibility package for the legacy GlyphMatics tensor engine.

The canonical high-level API lives in :mod:`glyphmatics`, but historical
callers import ``engine.glyphmatics_core`` directly. Keeping this package
installable preserves that public import path.
"""

from .glyphmatics_core import *  # noqa: F401,F403
