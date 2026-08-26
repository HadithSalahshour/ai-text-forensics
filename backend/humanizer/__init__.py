"""Local AI-writing detector and humanizer.

Detection runs entirely on-device. The rewrite engine's optional Claude pass is
the only component that ever touches the network, and it is off by default.
"""

__version__ = "0.1.0"
