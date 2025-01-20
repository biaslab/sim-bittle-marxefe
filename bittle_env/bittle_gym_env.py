"""This file implements the gym environment of Bittle.

NOTE: This is based on the minitaur code. Some changes:
  * follow the gym env customs by resetting the environment once and immediately get an observation upon reset to determine an action instead of having to reset twice
  * putting single variables into tuples with (var) is removed as I don't know why it was necessary.
  * a bunch of weird mistakes (gravity = -10) and spelling errors...
"""
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from .bittle_robot import Bittle
import gymnasium.spaces
import gymnasium as gym
from pybullet_utils import bullet_client as bc
import math
import numpy as np
import pybullet
import pybullet_data
import logging
import time
import pdb
from PIL import Image

NUM_SUBSTEPS = 5
RENDER_HEIGHT = 720
RENDER_WIDTH = 960

# actions config for a position controller where actions represent action offsets relative to the initial joint positions
ACTIONS_CONFIG_RELATIVE = {
  'bounds' : {
    'type' : 'relative',
    'range' : {
      'min' : -0.4,
      'max' : 0.4
    }
  }
}

ACTIONS_CONFIG_DEFAULT = ACTIONS_CONFIG_RELATIVE

# O_smith2022walking: similar or equivalent to smith2022walking
OBSERVATIONS_CONFIG_DEFAULT = {
  'JA' : True, # Joint / motor angles
  'JV' : True, # Joint / motor velocities,
  'JT' : False, # Joint / motor torques,
  #'BO' : { 'type' : 'quaternion', 'active' : [1,1,1,1] },
  'BO' : { 'type' : 'euler', 'active' : [1,1,0] },
  'BAV' : { 'active' : [1,1,1] },
  'BLV' : { 'active' : [1,1,1] },
  'FC' : True,
  'Aprev' : True
}


class BittleBulletEnv(gymnasium.Env):
  """The gym environment for the Bittle.

  It simulates the locomotion of a Bittle, a quadruped robot.
  The state space include the angles, velocities and torques for all the motors and the action space is the desired motor angle for each motor.
  The reward function is based on how far the Bittle walks in 1000 steps and penalizes the energy expenditure.

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50, 'render_modes' : ['human', 'rgb_array'], 'render_fps' : 50}

  def __init__(
      self,
      render_mode = None,
      save_every = 0,
      save_images_folder = 'images/',
      urdf_root = 'bittle_env/urdfs/', # pybullet_data.getDataPath(),
      action_repeat = 1,
      distance_weight = 1.0,
      distance_limit = float("inf"),
      observation_noise_stdev = 0.0,
      self_collision_enabled = True,
      hard_reset = False,
      on_rack = False,
      render = False,
      max_force = 0.55,
      control_mode = 'PD',
      actions_config = ACTIONS_CONFIG_DEFAULT,
      observations_config = OBSERVATIONS_CONFIG_DEFAULT
    ):
    """Initialize the Bittle gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      action_repeat: The number of simulation steps before actions are applied.
      distance_weight: The weight of the distance term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the Bittle back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the Bittle on rack. This is only used to debug
        the walking gait. In this mode, the Bittle's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
    """
    logging.debug(observations_config)
    #print(f'RUNNING WITH {action_repeat} and {max_force} and {urdf_root}')

    self._enable_debug_prints = False
    self._time_step = 0.01
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 100
    self._urdf_root = urdf_root
    if self._enable_debug_prints: print("urdf_root=" + self._urdf_root)
    self._self_collision_enabled = self_collision_enabled
    self._env_step_counter = 0
    self._control_mode = control_mode
    self._actions_config = actions_config

    self.render_mode = render_mode
    self.should_render = render_mode == "human"
    #print(f'WHAT ABOUT RENDERING? self.render_mode: {self.render_mode}, render: {render} or {self.should_render}')

    self._reward_weight_distance = distance_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._on_rack = on_rack
    self._last_frame_time = 0.0
    self._max_force = max_force
    self._num_motors = 8
    self._motor_angle_observation_index = 0
    self._motor_velocity_observation_index = self._motor_angle_observation_index + self._num_motors
    self._motor_torque_observation_index = self._motor_velocity_observation_index + self._num_motors
    self._observations_config = observations_config

    if self.should_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI, options="--width=1920 --height=1060")
    else:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

    self._first_initialization = True
    self._hard_reset = True
    self.reset()
    self._hard_reset = hard_reset  # This assignment need to be after reset() - first time initialization

    (action_high, action_low) = self.bittle.GetActionBounds()
    self.action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    (observation_high, observation_low) = self.bittle.GetObservationBounds()
    self.observation_space = gymnasium.spaces.Box(observation_low, observation_high, dtype=np.float32)

    #pdb.set_trace()

    if self._enable_debug_prints or True:
      logging.debug(f'A: {self.action_space}')
      logging.debug(f'O: {self.observation_space}')

    # allow saving images every n steps
    self._save_every = save_every
    self._enable_save_images = False
    self._save_images_folder = save_images_folder
    override_image = False

    if not isinstance(save_every, int):
      if self._enable_debug_prints: logging.error(f'Not saving images, save_every must be of type int, not {type(save_every)}')
      override_image = True # ensure that the image saving is disabled

    # create image folder
    if not isinstance(save_images_folder, str) or save_images_folder == '': 
      self._save_images_folder = 'images/'
      logging.warning(f'No save/correct image folder provided, using default: {self._save_images_folder}')
    elif save_images_folder[-1] != '/': 
      self._save_images_folder += '/'
      logging.warning(f'Image folder should end with /, appending / to {self._save_images_folder}')
    
    # set up image saving
    if save_every > 0 and self.should_render and not override_image:
      os.makedirs(self._save_images_folder, exist_ok=True) # create folder if not exists
      if self._enable_debug_prints: logging.debug(f'Saving images every {save_every} steps')
      self._save_every = save_every
      self._enable_save_images = True
    else:
      logging.debug(f'Not saving images, save_every is {save_every} <= 0')

  def configure(self, args):
    self._args = args

  # got an unexpected keyword argument 'seed'
  # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
  def reset(self, seed=None, options=None):
    if self._first_initialization or self._hard_reset:
      logging.debug('Creating TERRAIN ETC')
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      plane = self._pybullet_client.loadURDF(f'{pybullet_data.getDataPath()}/plane.urdf')
      self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
      self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -9.81) # -10, really?
      self.bittle = Bittle(
        pybullet_client=self._pybullet_client,
        urdf_root=self._urdf_root,
        time_step=self._time_step,
        self_collision_enabled=self._self_collision_enabled,
        on_rack=self._on_rack,
        max_force = self._max_force,
        action_repeat = self._action_repeat,
        control_mode = self._control_mode,
        actions_config = self._actions_config,
        observations_config = self._observations_config
      )
      self._first_initialization = False
    else:
      self.bittle.Reset(reload_urdf=False)

    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]

    self._update_cam(init=True)
    return self._get_observation(), {}
    #return self._noisy_observation(), {}

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    #assert action in self.action_space, f'{action} not in A'
    if self.should_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._action_repeat * self._time_step - time_spent
      if time_to_sleep > 0: time.sleep(time_to_sleep)

      self._update_cam()
    #action = self._transform_action_to_motor_command(action)
    action = self.bittle.Step(action)

    self._env_step_counter += 1
    reward = self._reward()
    terminated = self._termination()
    # XXX: adding truncated as a 5th return value is a problem introduced by gym version 0.26.0 where TimeLimit requires the 4th element to be truncated. to test manually, you have to unwrap the timelimit and add your own time limit checker
    truncated = False
    observation = self._get_observation()
    #assert observation in self.observation_space, f'{observation} not in O: \n {self.observation_space}'

    # save images every n steps
    if self._enable_save_images and self._env_step_counter % self._save_every == 0:
      self._get_frame(f"{self._env_step_counter}".zfill(8) + ".jpg") # left pad with zeros

    return observation, reward, terminated, truncated, {}
    #return np.array(self._noisy_observation()), reward, terminated, truncated, {}

  def _update_cam(self, init=False):
    cam_dist, cam_yaw, cam_pitch, cam_pos = self._get_cam_values(init=init)
    if self._enable_debug_prints: print(f'Camera: {cam_dist}, {cam_yaw}, {cam_pitch}, {cam_pos}')
    self._pybullet_client.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

  def set_cam_values(self, cam_yaw = None, cam_pitch = None, cam_dist = None):
    cam_info = self._pybullet_client.getDebugVisualizerCamera()
    if self._enable_debug_prints: print(f'Camera Info: {cam_info}')
    cam_yaw = cam_info[8] if cam_yaw is None else cam_yaw
    cam_pitch = cam_info[9] if cam_pitch is None else cam_pitch
    cam_dist = cam_info[10] if cam_dist is None else cam_dist
    cam_pos = self.bittle.GetBasePosition()
    self._pybullet_client.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

  def _get_cam_values(self, init=False):
    if init:
      cam_dist = 5.0
      cam_dist = 3.0
      cam_yaw = 90.0 # Looking left to right (see battery direction) - has some pixelation issues in the mesh
      cam_yaw = -90.0 # Looking right to left (see battery direction) - 
      cam_pitch = -30.0 # -30
    else:
      #     0,      1,          2,          3,        4,          5,        6,        7,       8,         9,       10,         11
      #   int,    int,  float(16),  float(16), float(3),   float(3), float(3), float(3),   float,     float,    float,   float(3)
      # width, height, viewmatrix, projmatrix, cameraUp, camForward,  horizon, vertical, cam_yaw, cam_pitch, cam_dist, cam_target
      cam_info = self._pybullet_client.getDebugVisualizerCamera()
      if self._enable_debug_prints: print(f'Camera Info: {cam_info}')
      cam_yaw = cam_info[8]
      cam_pitch = cam_info[9]
      cam_dist = cam_info[10]
    cam_pos = self.bittle.GetBasePosition()
    return cam_dist, cam_yaw, cam_pitch, cam_pos
  
  def _get_frame(self, filename:str)->None:
    """
    Get the current frame of the simulation as an image.
    
    Args:
    - filename (str): The name of the file to save the image as.
    
    Returns:
    - None"""
    width = 1920
    height = 1080
    image = pybullet.getCameraImage(width, height, shadow=True, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)[2] # Only RGBA
    image = np.reshape(image, (height, width, 4))
    image = Image.fromarray(image, 'RGBA').convert('RGB')

    # Save the image
    image.save(self._save_images_folder + filename, "JPEG")

  def render(self, force=False):
    if not force and self.render_mode == 'human': return np.array([])

    # _pybullet_client.getDebugVisualizerCamera will return zeros, camera does not exist yet?
    #cam_info = self._pybullet_client.getDebugVisualizerCamera()
    #if self._enable_debug_prints: print(f'Camera Info: {cam_info}') # everything is 0 for some reason
    #view_matrix, proj_matrix = cam_info[2:4]

    cam_dist, cam_yaw, cam_pitch, cam_pos = self._get_cam_values(init=True)
    if self._enable_debug_prints: print(f'Camera: {cam_dist}, {cam_yaw}, {cam_pitch}, {cam_pos}')
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
      cameraTargetPosition=cam_pos,
      distance=cam_dist,
      yaw=cam_yaw,
      pitch=cam_pitch,
      roll=0,
      upAxisIndex=2
    )
    if self._enable_debug_prints: print(f'view_matrix: {view_matrix}')
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
      fov=60,
      aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
      nearVal=0.1,
      farVal=100.0
    )
    if self._enable_debug_prints: print(f'proj_matrix: {proj_matrix}')
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
      width=RENDER_WIDTH,
      height=RENDER_HEIGHT,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    )
    #rgb_array = np.array(px)
    rgb_array = np.reshape(np.array(px), (RENDER_HEIGHT, RENDER_WIDTH, -1))
    print(rgb_array.shape)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    position = self.bittle.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    termination_fallen = self.bittle.IsFallen()
    termination_distance = distance > self._distance_limit
    if self._enable_debug_prints:
      if termination_fallen or termination_distance:
        print(f'DONE! REASON: fallen {termination_fallen} or distance {termination_distance}')
    return termination_fallen or termination_distance

  # NOTE: the bittle is oriented towards the y-Axis, not towards the x-Axis like the minitaur. Thus, we have to switch forward and drift reward!
  def _reward(self):
    current_base_position = self.bittle.GetBasePosition()
    reward_forward = current_base_position[1] - self._last_base_position[1]
    self._last_base_position = current_base_position
    reward = 0
    reward += self._reward_weight_distance * reward_forward
    return reward

  def _get_observation(self):
    self._observation = self.bittle.GetObservation()
    return np.asarray(self._observation, dtype=np.float32)

  def _noisy_observation(self, observation):
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(scale=self._observation_noise_stdev, size=observation.shape) * self.bittle.GetObservationUpperBound())
    return observation
