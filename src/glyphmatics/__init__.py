__version__ = "0.1.0"
__all__ = ["GlyphMaticsEngine", "visualize_network"]


def __getattr__(name):
    """Load the legacy tensor engine and Graphviz integration only on use."""

    if name == "GlyphMaticsEngine":
        from .component import GlyphMaticsEngine

        return GlyphMaticsEngine
    if name == "visualize_network":
        from .visualization.network import visualize_network

        return visualize_network
    raise AttributeError(name)
