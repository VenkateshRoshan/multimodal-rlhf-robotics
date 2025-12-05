import pybullet as p
import pybullet_data
import time
import random

def load_environment(use_gui=False):
    """
    Loads a basic PyBullet environment with:
    - Plane
    - Table
    - Franka Panda robot
    
    Returns:
        dict with robot_id, table_id, plane_id
    """

    # Connect to PyBullet
    if use_gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    # URDF search path
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Physics settings
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Load table (0.62m height usually)
    table_position = [0.5, 0.0, 0.0]
    table_orientation = p.getQuaternionFromEuler([0, 0, 0])
    table_id = p.loadURDF(
        "table/table.urdf",
        table_position,
        table_orientation,
        useFixedBase=True
    )

    # Load Panda robot slightly above table
    robot_position = [0.0, 0.0, 0.63]  # 0.63m = table height + small offset
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        robot_position,
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True
    )

    return {
        "robot_id": robot_id,
        "table_id": table_id,
        "plane_id": plane_id
    }

def load_cube(position=None, size=0.04, color=None):
    """
    Load a cube object into the PyBullet scene.

    Args:
        position (list or None): [x, y, z] spawn position.
                                 If None → random position on table.
        size (float): side length of cube (default = 4cm).
        color (list or None): [r, g, b, a] RGBA color. If None → random color.
    
    Returns:
        int: cube_id (PyBullet body unique ID)
    """

    # If no position provided → random on table top region
    if position is None:
        position = [
            random.uniform(0.35, 0.65),   # X range on table
            random.uniform(-0.20, 0.20),  # Y range
            0.65                          # Z height slightly above table
        ]

    # Random visual color
    if color is None:
        color = [
            random.random(),  # R
            random.random(),  # G
            random.random(),  # B
            1.0               # A
        ]

    # Cube collision + visual shape
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=color)

    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=position
    )

    return cube_id

def load_multiple_cubes(n=3):
    """
    Load multiple cubes with random positions & colors.

    Returns:
        List of cube_ids
    """
    cubes = []
    for _ in range(n):
        cubes.append(load_cube())
    return cubes


if __name__ == "__main__":
    env = load_environment(use_gui=True)
    print("Environment loaded:", env)
    cubes = load_multiple_cubes(n=5)
    time.sleep(30)  # Keep GUI open for 30 seconds