import argparse
from isaaclab.app import AppLauncher

# 1. Setup the App Launcher
parser = argparse.ArgumentParser(description="Verify Booster Robot Configuration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# spawn_t1_simple.py
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
import torch

# Import your config
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG


def main():
    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    #set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Add ground
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    # spawn distant light

    cfg_light_distant = sim_utils.DistantLightCfg(

        intensity=3000.0,

        color=(0.75, 0.75, 0.75),

    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))


    # 1. Get the height from your config
    spawn_pos = BOOSTER_T1_CFG.init_state.pos  # This is (0.0, 0.0, 0.70)

    # 2. Tell the spawner specifically where to put the root prim
    BOOSTER_T1_CFG.spawn.func(
        "/World/T1", 
        BOOSTER_T1_CFG.spawn, 
        translation=spawn_pos  # <--- Add this!
    )
    # Run
    #play simulator
    sim.reset()

    print("[INFO]: setup complete......")

    while simulation_app.is_running():
        sim.step()
    
if __name__ == "__main__":
    main()
    simulation_app.close()