import pybullet as p
import pybullet_data
import time


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


if __name__ == "__main__":
    env = load_environment(use_gui=True)
    print("Environment loaded:", env)
    time.sleep(30)  # Keep GUI open for 30 seconds
