"""This file implements the functionalities of a Bittle using pybullet.

Originally adapted for Bittle from pybullet_envs/bullet/minitaur.py, it now also includes many functionalities from minitaur/robots/minitaur.py

To see what API calls are possible with the pybullet_client, see bulletphysics/bullet3/examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp.

"""
"""
observations_config:

joint angles, joint velocites(roll, pitch, yaw), joint torques, base orientation(rpy), base angular velocity(rpy), base linear velocity(xyz), binary foot contacts, previous action (not current action), paper
JA, JV, JT, BO(rpy), BAV(rpy), BLV, FC, Aprev, other, paper
----------------------------------------
 -,  -,  -,     ---,      ---, ---,  -,  -,     -, -
 1,  1,  0,     110,      111  111,  1,  1,     0, bittle_env [joints: 8] [bittle]

 1,  1,  0,     110,      111, 111,  1,  1,     0, smith2022walking [joints: 12] [A1] [PD targets] [restricted action space]
 1,  0,  0,   ? 111,    ? 111, 000,  0,  0,     0, wu2023daydreamer [joints: 12] [A1] [PD targets] [Butterworth filter]
 1,  0,  0,     110,      110, 000,  0,  1,     0, ha2020learning (history of last 6 time steps) [joints: 8] [minitaur] [PD targets]
 1,  0,  0,     111,      000, 111,  0,  0,     1, yang2020data [joints: 8] [minitaur] 8+3+1 = 14 + 4 TG state (phase of TG)! [other=TG state: phase]
 1,  0,  0,     110,      110, 000,  0,  0,     0, haarnoja2018learning [joints: 8] [minitaur] [PD targets]
 1,  0,  0,     110,      110, 000,  0,  0,     0, tan2018simtoreal [joints: 8] [minitaur] [PD targets]
 1,  1,  1,    1111,      000, 000,  0,  0,     0, singla2019realizing [joints: 8] [Stoch] [BO in quaternions] 8+8+8+4
 1,  1,  0,     000,      111, 111,  0,  1,     1, hwangbo2019learning [joints: 12] [anymal B] [other = body twist, command, sparsely sampled joint state history, height, BO indirectly via 2D IMU gravity vector]
 1,  1,  0,     111,      111, 111,  0,  0,     1, gangapurwala2020guided [joints: 12] [anymal B] [109-D env state space] [other = height, policy output/desired joints, desired base velocty]
 1,  1,  0,     111,      111, 111,  0,  0,     1, lee2020learning [joints: 12?] [anymal C] [other = command vector, base twist, TG phase, base frequency] [PD targets] [difference O for teacher/student] [see Table S4] [total = 121 (student) and 121+71=192 (teacher)]
 0,  0,  0,     000,      000, 000,  0,  0,     1, gangapurwala2021real [total = 546 = 486 + 60 (for 80x80 maps) WTF] [other=generalized coordinates, ] [PD targets] [joints: 12] [anymal B]
"""


"""
 controlModes

PyBullet Quickstart Guide p.25

TORQUE_CONTROL == pybullet.c:CONTROL_MODE_TORQUE
  torque
POSITION_CONTROL == pybullet.c:CONTROL_MODE_POSITION_VELOCITY_PD
  positionGain
  velocityGain
VELOCITY_CONTROL == pybullet.c:CONTROL_MODE_VELOCITY
  targetVelocity
PD_CONTROL == pybullet.c:CONTROL_MODE_PD
STABLE_PD_CONTROL == pybullet.c:CONTROL_MODE_STABLE_PD
"""

from .bittle_motor import MotorModel
import copy
import math
import numpy as np
import logging
import os
import pdb
import re
import time

# TODO: Footfall pattern of gaits from feet/knee link (maybe add a paw link) contacts

# actions config for a position controller where actions represent the abolute joint positions
ACTIONS_CONFIG_ABSOLUTE = {
  'bounds' : {
    'type' : 'absolute',
    'range' : {
      'min' : -math.pi,
      'max' : math.pi
    }
  }
}
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

class Bittle(object):
  def __init__(
    self,
    pybullet_client,
    urdf_root = os.path.join(os.path.dirname(__file__), "../data"),
    on_rack = False,
    action_repeat = 1,
    time_step = 0.01,
    self_collision_enabled = True,
    motor_offset = None,
    max_force = 0.55, # NOTE: 0.05 is too low of a torque, 0.55 is quite OK - need to figure out reasonable max force
    #fallen_threshold_orientation = 0.60, # minitaur: 0.85
    fallen_threshold_orientation = 0.80, # minitaur: 0.85
    fallen_threshold_height = 0.50, # minitaur: 0.13
    control_mode = 'PD',
    actions_config = ACTIONS_CONFIG_DEFAULT,
    observations_config = OBSERVATIONS_CONFIG_DEFAULT,
  ):
    """Constructs a Bittle and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
      on_rack: Whether to place the bittle on rack. This is only used to debug
        the walking gait. In this mode, the bittle's base is hanged midair so
        that its walking gait is clearer to visualize.
    """
    self._enable_debug_prints = False
    self.num_motors = 8
    self.num_legs = int(self.num_motors / 2)
    self._fallen_threshold_orientation = fallen_threshold_orientation
    self._fallen_threshold_height = fallen_threshold_height
    self._action_repeat = action_repeat
    self._pybullet_client = pybullet_client
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled

    self._orientation_representations = ['quaternion', 'euler']
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_action = np.zeros(self.num_motors)
    self._motor_offset = np.zeros(self.num_motors) if motor_offset is None else motor_offset
    # Left side (first 4 values) is mirrored to right side
    self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
    # determines leg ID ordering as above: LF LB RF RB
    self._motor_order = ["left-front", "left-back", "right-front", "right-back"]
    # all motors are joints of type revolute/continuous. ordering: left to right, front to back, "L" to "R" -- possibly shoulder to knee?
    self._motor_names = [
        "left-front-shoulder-joint", "left-front-knee-joint",
        "left-back-shoulder-joint", "left-back-knee-joint",
        "right-front-shoulder-joint", "right-front-knee-joint",
        "right-back-shoulder-joint", "right-back-knee-joint"
    ]
    self._name_patterns = {
      'joint' : {
        'knee' : re.compile(r'.*-knee-joint'),
        'shoulder' : re.compile(r'.*-shoulder-joint')
      },
      'link' : {
        'knee' : re.compile(r'.*-knee-link'),
        'shoulder' : re.compile(r'.*-shoulder-link')
      }
    }
    # [LFS, LFK, LBS, LBK, RFS, RFK, RBS, RBK]
    # [  4,   5,   2,   3,  10,  11,   8,   9] is the motor list currently
    # shoulder: 4, 2, 10, 8
    # knee: 5, 3, 11, 9
    self._observed_torque_limit = 5.7
    self._motor_voltage = 16.0
    self._motor_viscous_damping = 0.0
    self._motor_torque_constant = 0.0954
    self._motor_speed_limit = self._motor_voltage / (self._motor_viscous_damping + self._motor_torque_constant)
    # max values for torque control:  2.0
    # min values for torque control: -2.0
    # max values for PD control:
    self._quarter_pi = math.pi / 4

    control_modes = {
      'PD' : self._pybullet_client.PD_CONTROL,
      'position': self._pybullet_client.POSITION_CONTROL,
      'torque' : self._pybullet_client.TORQUE_CONTROL,
      'velocity' : self._pybullet_client.VELOCITY_CONTROL
    }
    assert control_mode in control_modes, f'Cannot find control mode {control_mode}. Options: {control_modes.keys()}'
    self._control_mode = control_modes[control_mode]
    logging.debug(f'chosen control mode: {control_mode}')

    self._actions_config = actions_config
    self._max_force = max_force
    # action bounds (symmetric) and thus action space depends on the choice of control
    if self._control_mode == self._pybullet_client.TORQUE_CONTROL:
      self._action_bound = self._max_force
    elif self._control_mode == self._pybullet_client.POSITION_CONTROL or self._control_mode == self._pybullet_client.PD_CONTROL:
      self._action_bound = self._actions_config['bounds']['range']['max'] # math.pi
    elif self._control_mode == self._pybullet_client.VELOCITY_CONTROL:
      self._action_bound = 5.0
    else: raise NotImplementedError(f'Unrecognized control mode: {self._control_mode}')
    logging.debug(f'action_bound: {self._action_bound}')
    self._observation_eps = 0.0 # was 0.01 but dont know why this exists
    self._on_rack = on_rack

    self._observations_config = observations_config
    if self._enable_debug_prints: logging.debug(f'Observations: {self._observations_config}')
    self.time_step = time_step
    STANDING = [0.0, 1.48, 0.0, 1.48, 0.0, -1.22, 0.0, -1.22]
    STANDING = [self._quarter_pi, 0, self._quarter_pi, 0., -self._quarter_pi, 0., -self._quarter_pi, 0]
    INIT_POSITION_KNEELING = [0, 0, 0.65]
    KNEELING = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    INIT_POSITION_STANDING = [0, 0, 1.00] # no feet have contact
    INIT_POSITION_STANDING = [0, 0, 0.775] # all feet have contact
    INIT_POSITION_ONRACK = [0, 0, 2.00] # Hangs in the air
    #INIT_POSITION_ONRACK = [0, 0, 1.05] # Only 3 and 5 have contact == left side
    #INIT_POSITION_ONRACK = [0, 0, 0.80] # All 4 legs have contact
    #INIT_POSITION_ONRACK = [0, 0, 1.00] # All 4 legs have contact: 3 (LF) 5 (LB), 9 (RF) 11 (RB)
    # INIT_ORIENTATION are quaternions. can be exchanged to angles with orn = p.getQuaternionFromEuler([0, 0, 0])
    self._init_position_onrack = [0, 0, 2.00]
    self._desired_position = STANDING
    if self._desired_position == STANDING:
      self._init_position = INIT_POSITION_STANDING
    elif self._desired_position == KNEELING:
      self._init_position = INIT_POSITION_KNEELING
    if self._on_rack:
      self._init_position = INIT_POSITION_ONRACK

    self._init_orientation = [0, 0, 0, 1]
    _, self._init_orientation_inv = self._pybullet_client.invertTransform(
      position=[0,0,0],
      orientation = self._GetDefaultInitOrientation()
    )

    self.Reset(reload_urdf = True)
    self.ReceiveObservation()


  def _GetDefaultInitOrientation(self):
    # TODO: reset at current position stuff
    return self._init_orientation

  def Reset(self, reload_urdf=False):
    if self._enable_debug_prints: print(f'Resetting Bittle. reload_urdf: {reload_urdf}')
    if reload_urdf:
      urdf_path = f'{self._urdf_root}/bittle/bittle.urdf'
      if self._enable_debug_prints: print(f'LOADING FROM {self._urdf_root}')
      flags = 0
      # TODO: use inertia values from URDF
      # b3Printf: Bad inertia tensor properties, setting inertia to zero for link <all motors> from bulletphysics/bullet3/examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp
      #flags |= self._pybullet_client.URDF_USE_INERTIA_FROM_FILE # there are a lot of concrete inertial values in the URDF file, so we use this flag, see pybullet_quickstartguide.pdf
      if self._self_collision_enabled: flags |= self._pybullet_client.URDF_USE_SELF_COLLISION
      self._body_ID = self._pybullet_client.loadURDF(urdf_path, basePosition = self._init_position, baseOrientation = self._init_orientation, flags=flags)
      self._BuildMotorIDListFromURDF()
      if self._on_rack:
        self._pybullet_client.createConstraint(self._body_ID, -1, -1, -1, self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self._init_position_onrack)

    self._pybullet_client.resetBasePositionAndOrientation(self._body_ID, self._init_position, self._init_orientation)
    self._pybullet_client.resetBaseVelocity(self._body_ID, [0, 0, 0], [0, 0, 0])
    self.ResetPose()

    self._overheat_counter = np.zeros(self.num_motors)
    self._step_counter = 0
    self._state_action_counter = 0
    self._last_action = np.zeros(self.num_motors)

    # to update the simulation
    #time.sleep(0.1)
    self._pybullet_client.stepSimulation()
    feet_ground_contact = self.GetFootContacts()
    all_feet_ground_contact = all(feet_ground_contact)
    if not self._on_rack:
      assert all_feet_ground_contact, f'Reset: Not all feet have contact on reset: f{feet_ground_contact}'


  def ReceiveObservation(self):
    """Receive the observation from sensors.

    This function is called once per step.
    The observations are only updated when this function is called.
    """
    self._joint_states = self._pybullet_client.getJointStates(self._body_ID, self._motor_IDs)
    # base orientation is returned as quaternions
    self._base_position, self._base_orientation = (self._pybullet_client.getBasePositionAndOrientation(self._body_ID))
    # TODO: _base_orientation is global orientation - to get the relative, see robots/minitaur.py
    _, self._base_orientation = self._pybullet_client.multiplyTransforms(
      positionA = [0,0,0],
      orientationA = self._base_orientation,
      positionB = [0,0,0],
      orientationB=self._GetDefaultInitOrientation()
    )
    (self._base_velocity_linear, self._base_velocity_angular) = self._pybullet_client.getBaseVelocity(self._body_ID)

  def _BuildMotorIDListFromURDF(self):
    """
    Gets all joint names from the URDF and gives them an ID, by order of appearance
    """
    self._joint_name_to_id = {}
    self._link_IDs = {
      'knee' : [],
      'shoulder' : []
    }
    num_joints = self._pybullet_client.getNumJoints(self._body_ID)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self._body_ID, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_ID = joint_info[0]
      self._joint_name_to_id[joint_name] = joint_ID
      joint_pattern = None
      for joint_pattern_type in self._name_patterns.keys():
        for pattern_type, pattern in self._name_patterns[joint_pattern_type].items():
          if not pattern.match(joint_name): continue
          joint_pattern = pattern
          break
        if joint_pattern is not None: break
      if joint_pattern is None:
        logging.debug(f'Cannot find a pattern for {joint_name}')
        continue
      logging.debug(f'Found {joint_pattern} of type {joint_pattern_type} for {joint_name} with ID {joint_ID}')
      if joint_pattern_type != 'joint': continue # We do not care about the links, yet. Each joint is associated with the link
      self._link_IDs[pattern_type].append(joint_ID)
    self._motor_IDs = [self._joint_name_to_id[motor_name] for motor_name in self._motor_names]
    if self._enable_debug_prints: print(f'_BuildMotorIdList: {self._motor_IDs}')
    logging.debug(f'_BuildMotorIdList: self._motor_IDs: {self._motor_IDs}')

  def ResetPose(self):
    """Reset the pose of the Bittle.
    """
    if self._enable_debug_prints: print(f'Resetting pose.')

    # Reset pose for each leg
    for i, motor_ID in enumerate(self._motor_IDs):
      self._pybullet_client.resetJointState(
        self._body_ID,
        motor_ID,
        self._desired_position[i]
      )

    # Disable the default velocity motor in pybullet
    self._disable_default_motor()


  def _disable_default_motor(self):
    """
    Disable the default velocity motor.

    Info from PyBullet Quickstart Guide p.24, setJointMotorControl2/Array:
      Important Note: by default, each revolute joint and prismatic joint is motorized using a velocity motor.
      You can disable those default motor by using a maximum force of 0.
      This will let you perform torque control.
    """
    self._pybullet_client.setJointMotorControlArray(
      bodyIndex = self._body_ID,
      jointIndices = self._motor_IDs,
      controlMode = self._pybullet_client.VELOCITY_CONTROL,
      targetVelocities = [0.0] * len(self._motor_IDs),
      forces = [0.0]* len(self._motor_IDs)
    )

  def GetBasePosition(self):
    """Get the position of Bittle's base.

    Returns:
      The position of Bittle's base.
    """
    return self._base_position

  def GetContacts(self, motor_IDs):
    contact_points = map(self.GetGroundContactPoints, motor_IDs)
    contact_points = map(bool, contact_points)
    return np.asarray(list(contact_points))

  def GetFootContacts(self):
    """
    Returns:
      Boolean value for every foot of the Bittle in contact with something else than its own body.
      NOTE: The URDF does not include yet a point/link for the foot itself, so the knee link is a proxy for that.
    """
    return self.GetContacts(self._link_IDs['knee'])

  def GetFeetCount(self):
    return len(self._link_IDs['knee'])

  def GetShoulderContacts(self):
    return self.GetContacts(self._link_IDs['shoulder'])

  def HasFootContact(self):
    return any(self.GetFootContacts())

  def HasShoulderContact(self):
    return any(self.GetShoulderContacts())

  def GetGroundContactPoints(self, motor_ID):
    """Get foot contact with the ground
    Returns:
      List of Booleans representing ith leg being in contact with the ground.
    """
    return self._pybullet_client.getContactPoints(bodyA=0, bodyB=self._body_ID, linkIndexA=-1, linkIndexB=motor_ID)

  def GetContactPoints(self, motor_ID):
    return self._pybullet_client.getContactPoints(bodyA=self._body_ID, linkIndexA=motor_ID)

  def GetBaseOrientation(self, representation):
    """Get the orientation of Bittle's base in a given representation.

    Returns:
      The orientation of Bittle's base.
    """
    if representation not in self._orientation_representations: raise NotImplementedError(f'Unrecognized orientation representation {representation}')
    orientation = self._base_orientation if representation == 'quaternion' else self._pybullet_client.getEulerFromQuaternion(self._base_orientation)
    return np.asarray(orientation)

  def GetBaseLinearVelocity(self):
    return np.asarray(self._base_velocity_linear)

  def GetBaseAngularVelocity(self):
    return np.asarray(self._base_velocity_angular)

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def GetActionBounds(self):
    # TODO: Depending on motor command type (direct torque control or position / PD control, we (should) have different action spaces)
    return self.GetActionUpperBound(), self.GetActionLowerBound()

  def GetActionUpperBound(self):
    return np.array([self._action_bound] * self.GetActionDimension())

  def GetActionLowerBound(self):
    return -self.GetActionUpperBound()

  def GetObservationBounds(self):
    return self.GetObservationUpperBound(), self.GetObservationLowerBound()

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation.
      See GetObservation() for the details of each element of an observation.
    """
    upper_bound = np.array([0.0] * self.GetObservationDimension())

    idx = 0
    # Joint angle
    if self._observations_config['JA']:
      upper_bound[idx:idx+self.num_motors] = math.pi
      idx += self.num_motors
    # Joint velocity
    if self._observations_config['JV']:
      upper_bound[idx:idx + self.num_motors] = (self._motor_speed_limit)
      idx += self.num_motors
    # Joint torque
    if self._observations_config['JT']:
      upper_bound[idx:idx + self.num_motors] = (self._observed_torque_limit)
      idx += self.num_motors
    if self._observations_config['BO']:
      if self._observations_config['BO']['type'] not in self._orientation_representations:
        raise Exception("Wrong BO type in observations_config")
      for _ in self._observations_config['BO']['active']:
        if _ != 1: continue
        # if self._observations_config['BO']['type'] == 'euler' / quaternion
        upper_bound[idx:idx + 1] = math.pi
        idx += 1
    for vel_type in ['BLV', 'BAV']:
      if not self._observations_config[vel_type]: continue
      for _ in self._observations_config[vel_type]['active']:
        upper_bound[idx:idx + 1] = math.pi
        idx += 1
    if self._observations_config['FC']:
      n_feet = self.GetFeetCount()
      upper_bound[idx:idx + n_feet] = 1.0 # TODO: boolean to range mapping correct?
      idx += n_feet
    if self._observations_config['Aprev']:
      upper_bound[idx:idx+self.num_motors] = math.pi
      idx += self.num_motors
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

  def GetObservationDimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  def GetObservation(self):
    """Get the observations of Bittle.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list.
        observation[0:8] are motor angles,
        observation[8:16] are motor velocities
        observation[16:24] are motor torques
        observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []
    if self._observations_config['JA']: observation.extend(self.GetTrueMotorAngles().tolist())
    if self._observations_config['JV']: observation.extend(self.GetTrueMotorVelocities().tolist())
    if self._observations_config['JT']: observation.extend(self.GetTrueMotorTorques().tolist())
    if self._observations_config['BO']:
      if self._observations_config['BO']['type'] not in self._orientation_representations:
        raise Exception("Wrong BO type in observations_config")
      orientation = []
      for i, _ in enumerate(self.GetBaseOrientation(self._observations_config['BO']['type'])):
        if self._observations_config['BO']['active'][i] != 1: continue
        orientation.append(_)
      observation.extend(orientation)
    for vel_type in ['BLV', 'BAV']:
      if not self._observations_config[vel_type]: continue
      velocity = self.GetBaseLinearVelocity() if vel_type == 'BLV' else self.GetBaseAngularVelocity()
      tmp = []
      for i, _ in enumerate(velocity):
        if self._observations_config[vel_type]['active'][i] != 1: continue
        tmp.append(_)
      observation.extend(tmp)
    if self._observations_config['FC']: observation.extend(self.GetFootContacts().tolist())
    if self._observations_config['Aprev']: observation.extend(self._last_action)
    return observation

  def GetObservationIndices(self):
    idx = 0
    indices = {}
    # Joint angle, joint velocity, joint torque
    for obs_type in ['JA', 'JV', 'JT']:
      if not self._observations_config[obs_type]: continue
      indices[obs_type] = [idx, idx+self.num_motors]
      idx += self.num_motors
    for obs_type in ['BO', 'BLV', 'BAV']:
      if not self._observations_config[obs_type]: continue
      indices[obs_type] = [idx, idx + sum(self._observations_config[obs_type]['active'])]
      idx += sum(self._observations_config[obs_type]['active'])
    if self._observations_config['FC']:
      n_feet = self.GetFeetCount()
      indices['FC'] = [idx, idx + n_feet]
      idx += n_feet
    if self._observations_config['Aprev']:
      indices['Aprev'] = [idx, idx+self.num_motors]
      idx += self.num_motors
    return indices


  def Step(self, action):
    for t in range(self._action_repeat):
      self._StepInternal(action)
      self._step_counter += 1
    self._last_action = action

  def _StepInternal(self, action):
    self.ApplyAction(action)
    self._pybullet_client.stepSimulation()
    self.ReceiveObservation()
    self._state_action_counter += 1


  def ApplyAction(self, action):
    """Set the desired motor angles to the motors of the Bittle.
    Args:
      action: depending on control_mode actions are:
        * TORQUE_CONTROL:   direct torques
        * POSITION_CONTORL: joint positions
        * VELOCITY_CONTROL: joint velocities
        * PD_CONTROL:       joint positions
    """
    if self._enable_debug_prints: print(f'ApplyAction({action}')
    #print(f'ApplyAction({action}')
    self._applied_action = np.multiply(action, self._motor_direction)

    if self._control_mode == self._pybullet_client.TORQUE_CONTROL:
      # torque control: actions represent direct motor commands as torques
      if self._enable_debug_prints: print(f'{self._applied_action} self._applied_action')
      #print(f'torque: {self._applied_action}')
      self._pybullet_client.setJointMotorControlArray(
        bodyIndex = self._body_ID,
        jointIndices = self._motor_IDs,
        controlMode = self._control_mode,
        forces = self._applied_action
      )
    elif self._control_mode == self._pybullet_client.POSITION_CONTROL:
      # position control: actions represent joint positions
      kps = [20.0] * len(self._motor_IDs)
      kds = [0.5] * len(self._motor_IDs)
      max_forces = [self._max_force] * len(self._motor_IDs)
      self._pybullet_client.setJointMotorControlArray(
        bodyIndex = self._body_ID,
        jointIndices = self._motor_IDs,
        controlMode = self._control_mode,
        targetPositions = self._applied_action,
        positionGains = kps,
        velocityGains = kds,
        forces = max_forces
      )
    elif self._control_mode == self._pybullet_client.VELOCITY_CONTROL:
      max_forces = [self._max_force] * len(self._motor_IDs)
      self._pybullet_client.setJointMotorControlArray(
        bodyIndex = self._body_ID,
        jointIndices = self._motor_IDs,
        controlMode = self._control_mode,
        targetVelocities = self._applied_action,
        forces = max_forces
      )
    elif self._control_mode == self._pybullet_client.PD_CONTROL:
      kps = [20.0] * len(self._motor_IDs)
      kds = [0.5] * len(self._motor_IDs)
      max_forces = [self._max_force] * len(self._motor_IDs)
      target_velocities = [0.0] * len(self._motor_IDs)
      if self._actions_config['bounds']['type'] == 'relative':
        self._applied_action += self._desired_position
      self._pybullet_client.setJointMotorControlArray(
        bodyIndex = self._body_ID,
        jointIndices = self._motor_IDs,
        controlMode = self._control_mode,
        targetPositions = self._applied_action,
        targetVelocities = target_velocities,
        positionGains = kps,
        velocityGains = kds,
        forces = max_forces
      )
    else: raise NotImplementedError(f'ApplyAction: unrecognizable control mode {self._control_mode}')

  def GetTrueMotorAngles(self):
    """Get the eight motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = np.asarray([_[0] for _ in self._joint_states])
    motor_angles = np.multiply(motor_angles - self._motor_offset, self._motor_direction)
    return motor_angles

  def GetTrueMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = np.asarray([_[1] for _ in self._joint_states])
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetTrueMotorTorques(self):
    """Get the amount of torques the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    return self._observed_motor_torques

  def IsFallen(self):
    """Decide whether the Bittle has fallen.

    Returns:
      Boolean value that indicates whether the Bittle has fallen.
    """
    # reason 1: the up directions between the base and the world is larger than some threshold
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(self._base_orientation)
    local_up = rot_mat[6:]
    magic_value = np.dot(np.asarray([0, 0, 1]), np.asarray(local_up))
    fallen_reason_1 = magic_value < self._fallen_threshold_orientation
    #print(f'fallen_reason_1: {magic_value} vs {self._fallen_threshold_orientation}')
    #print(f'magic value: {magic_value}')

    # reason 2: the base is too low on the ground than some threshold
    fallen_reason_2 = self._base_position[2] < self._fallen_threshold_height # 0.5 # sitting down is 0.66
    #print(f'fallen_reason_2: {self._base_position[2]} vs {self._fallen_threshold_height}')

    # reason 3: the shoulders have contact with the ground
    has_shoulder_contact = self.HasShoulderContact()
    #print(f'fallen_reason_3: {has_shoulder_contact}')

    is_fallen = False
    is_fallen |= fallen_reason_1
    is_fallen |= fallen_reason_2
    is_fallen |= has_shoulder_contact

    #print(f'FALLEN: {is_fallen} = {fallen_reason_1} {fallen_reason_2} {has_shoulder_contact}')
    return is_fallen
