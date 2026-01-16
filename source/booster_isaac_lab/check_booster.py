import argparse
from isaaclab.app import AppLauncher

# 1. Setup the App Launcher
parser = argparse.ArgumentParser(description="Verify Booster Robot Configuration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Import Isaac Lab modules
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# 3. Import YOUR Booster Config 
# Note: Ensure your PYTHONPATH includes the directory where booster_isaac_lab resides
try:
    from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG
except ImportError as e:
    print(f"\n[ERROR]: Could not import BOOSTER_T1_CFG. Error: {e}")
    print("Ensure you run this from the 'source' directory or set your PYTHONPATH.\n")
    simulation_app.close()
    exit()

def design_scene():
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )

    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))



def main():
    # """Main function to run the simulation."""
    
    # # 4. Define the Scene Configuration
    # scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    # # Corrected Ground Plane setup
    # scene_cfg.ground = AssetBaseCfg(
    #     prim_path="/World/defaultGround",
    #     spawn=sim_utils.GroundPlaneCfg()
    # )
    
    # # Setup Robot: We modify the prim_path of your existing config
    # scene_cfg.robot = BOOSTER_T1_CFG
    # scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    
    # # Add a light so we can see the robot
    # scene_cfg.light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # )

    # # 5. Initialize the Scene
    # scene = InteractiveScene(scene_cfg)
    
    # # 6. Start Simulation
    # sim_utils.play()
    # print("[INFO]: Simulation started. The robot should spawn at the center.")

    # # 7. Simulation Loop
    # while simulation_app.is_running():
    #     # Step the physics and update the scene
    #     scene.write_data_to_sim()
    #     sim_utils.step()
    #     scene.update(dt=0.01)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    #set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    #design scene
    design_scene()
    
    #play simulator
    sim.reset()

    print("[INFO]: setup complete......")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()