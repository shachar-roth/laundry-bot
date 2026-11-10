import os

# Headless rendering by default; must be set before mujoco is first imported.
os.environ.setdefault("MUJOCO_GL", "osmesa")
