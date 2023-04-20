import time
import glob
from typing import Union, Tuple
import os
from collections import deque
import numpy as np
from numpy.typing import ArrayLike, NDArray
#import scipy.spatial.transform

import pybullet as pb
import pybullet_data


class Object:
    def __init__(self, position: ArrayLike, quat: ArrayLike) -> None:
        '''Instantiates random household object in simulator at specified
        position and rotation quaternion.  Color is randomly assigned to a valid
        RGB value.
        '''
        urdf_paths = glob.glob('assets/random_household_object/*/*.urdf')
        urdf_path = urdf_paths[np.random.choice(np.arange(len(urdf_paths)))]
        self.id = pb.loadURDF(urdf_path, position, quat, globalScaling=0.6)

        # set color randomly
        r, g, b = np.random.uniform(0.0, 1, size=3)**0.8
        pb.changeVisualShape(self.id, -1, -1,
                             rgbaColor=(r, g, b, 1))

    def get_position(self) -> NDArray:
        '''Returns (x, y, z) position of object. Exact object center is defined
        in the respective URDF
        '''
        return np.array(pb.getBasePositionAndOrientation(self.id)[0])

    def remove(self) -> None:
        '''Remove object from simulator'''
        pb.removeBody(self.id)


class PandaArm():
    def __init__(self) -> None:
        '''Panda Robot Arm in pybullet simulator'''
        super().__init__()
        self.home_positions = [
            -0.60, -0.14, 1.19, -2.40, 0.11, 2.28, -1, 0.0, 0, 0, 0, 0, 0
        ]
        self.home_positions_joint = self.home_positions[:7]
        self.gripper_joint_limit = [0, 0.04]
        self.max_force = 240
        self.position_gain = 0.05
        self.end_effector_index = 11

        self.num_dofs = 7
        self.ll = [-7]*self.num_dofs
        self.ul = [7]*self.num_dofs
        self.jr = [7]*self.num_dofs

        urdf_filepath = os.path.join('assets', 'panda.urdf')
        self.id = pb.loadURDF(urdf_filepath, useFixedBase=True)
        pb.resetBasePositionAndOrientation(self.id, [-0.1,0,0], [0,0,0,1])

        self.gripper_closed = False
        self.num_joints = pb.getNumJoints(self.id)
        [pb.resetJointState(self.id, idx, self.home_positions[idx])
         for idx in range(self.num_joints)]
        pb.enableJointForceTorqueSensor(self.id, 8)
        c = pb.createConstraint(self.id,
                                9,
                                self.id,
                                10,
                                jointType=pb.JOINT_GEAR,
                                jointAxis=[1, 0, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0])
        pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(pb.getNumJoints(self.id)):
            pb.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

        self.openGripper()

        self.arm_joint_names = list()
        self.arm_joint_indices = list()
        for i in range (self.num_joints):
            joint_info = pb.getJointInfo(self.id, i)
            if i in range(self.num_dofs):
                self.arm_joint_names.append(str(joint_info[1]))
                self.arm_joint_indices.append(i)

    def reset(self):
        '''Move to home position, slightly above and to the left of the workspace'''
        self.gripper_closed = False
        [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
        self.moveToJ(self.home_positions_joint[:self.num_dofs])
        self.openGripper()

    def closeGripper(self, max_it=100) -> bool:
        '''Closes gripper and returns whether gripper was closed succesfully'''
        p1, p2 = self._getGripperJointPosition()
        target = self.gripper_joint_limit[0]
        self._sendGripperCommand(target, target, force=10)
        self.gripper_closed = True
        it = 0
        while abs(target-p1) + abs(target-p2) > 0.001:
            pb.stepSimulation()
            it += 1
            p1_, p2_ = self._getGripperJointPosition()
            if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
                return False
            p1 = p1_
            p2 = p2_
        return True

    def openGripper(self):
        '''opens gripper'''
        p1, p2 = self._getGripperJointPosition()
        target = self.gripper_joint_limit[1]
        self._sendGripperCommand(target, target)
        self.gripper_closed = False
        it = 0
        while abs(target-p1) + abs(target-p2) > 0.001:
            pb.stepSimulation()
            it += 1
            if it > 100:
                return False
            p1_, p2_ = self._getGripperJointPosition()
            if p1 >= p1_ and p2 >= p2_:
                return False
            p1 = p1_
            p2 = p2_
        return True

    def _calculateIK(self, pos, rot):
        return pb.calculateInverseKinematics(
            self.id, self.end_effector_index, pos, rot, self.ll, self.ul, self.jr,
        )[:self.num_dofs]

    def _getGripperJointPosition(self):
        p1 = pb.getJointState(self.id, 9)[0]
        p2 = pb.getJointState(self.id, 10)[0]
        return p1, p2

    def moveTo(self, pos, rot, dynamic=True, pos_th=1e-3, rot_th=1e-3):
        '''
        Move the end effector to the desired cartesian pose.
        Args:
          pos (numpy.array): Desired end-effector position.
          rot (numpy.array): Desired end-effector orientation.
          dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
          pos_th (float): Positional threshold for ending the movement. Defaults to 1e-3.
          rot_th (float): Rotational threshold for ending the movement. Defaults to 1e-3.
        '''

        close_enough = False
        outer_it = 0
        max_outer_it = 10
        max_inner_it = 100

        while not close_enough and outer_it < max_outer_it:
            ik_solve = self._calculateIK(pos, rot)
            self.moveToJ(ik_solve, dynamic, max_inner_it)

            ls = pb.getLinkState(self.id, self.end_effector_index)
            new_pos = list(ls[4])
            new_rot = list(ls[5])
            close_enough = np.allclose(np.array(new_pos), pos, atol=pos_th) and \
                           np.allclose(np.array(new_rot), rot, atol=rot_th)
            outer_it += 1

        return close_enough

    def moveToJ(self, target_pose, dynamic=True, max_it=1000):
        '''
        Move the desired joint positions

        Args:
        joint_pose (numpy.array): Joint positions for each joint in the manipulator.
        dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
        max_it (int): Maximum number of iterations the movement can take. Defaults to 10000.
        '''
        if dynamic:
            self._sendPositionCommand(target_pose)
            past_joint_pos = deque(maxlen=5)
            joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
            joint_pos = list(zip(*joint_state))[0]
            n_it = 0
            while not np.allclose(joint_pos, target_pose, atol=1e-3) and n_it < max_it:
                pb.stepSimulation()
                n_it += 1
                # Check to see if the arm can't move any close to the desired joint position
                if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
                    break
                past_joint_pos.append(joint_pos)
                joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
                joint_pos = list(zip(*joint_state))[0]
        else:
            self._setJointPoses(target_pose)

    def _setJointPoses(self, q_poses):
        '''
        Set the joints to the given positions.
        Args:
        q_poses (numpy.array): The joint positions.
        '''
        for i in range(len(q_poses)):
            motor = self.arm_joint_indices[i]
            pb.resetJointState(self.id, motor, q_poses[i])

        self._sendPositionCommand(q_poses)


    def _sendPositionCommand(self, commands):
        ''''''
        num_motors = len(self.arm_joint_indices)
        pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                               targetVelocities=[0.]*num_motors,
                               forces=[self.max_force]*num_motors,
                               positionGains=[self.position_gain]*num_motors,
                               velocityGains=[1.0]*num_motors)

    def _sendGripperCommand(self, target_pos1, target_pos2, force=10):
        pb.setJointMotorControlArray(self.id,
                                     [9, 10],
                                     pb.POSITION_CONTROL,
                                     [target_pos1, target_pos2],
                                     forces=[force, force])


class TopDownCamera:
    def __init__(self, workspace: np.ndarray, img_size: int) -> None:
        '''Camera that is mounted 0.25 meters above workspace to capture
        top-down image

        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))

        Attributes
        ----------
        img_size : int
            height, width of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_size = img_size

        cam_height = 0.25
        workspace_width = workspace[1,0] - workspace[0,0]
        fov = 2 * np.degrees(np.arctan2(workspace_width/2, cam_height))

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, cam_height)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(-1,0,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=fov,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)

    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image

        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_size,
                                 height=self.img_size,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]
        rgba = np.reshape(rgba, (self.img_size, self.img_size, 4)).astype(np.uint8)
        rgba = np.reshape(rgba, (self.img_size, self.img_size, 4))
        return rgba[..., :3]


class Simulator:
    def __init__(self,
                 num_objects: int=1,
                 img_size: int=128,
                 grasp_height: float=0.01,
                 obj_rotation_range: Tuple=(-np.pi, np.pi),
                 render: bool=True,
                ):
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        if render:
            pb.resetDebugVisualizerCamera(1.2, 64, -35, (0,0,0))
            pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        pb.setGravity(0, 0, -9.8)
        pb.setPhysicsEngineParameter(
            numSubSteps=0,
            numSolverIterations=100,
            solverResidualThreshold=1e-7,
            constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI,
        )

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pb.loadURDF("plane.urdf", (0, -0.5, 0))
        # makes collisions with plane more stable
        pb.changeDynamics(self.plane, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        self.robot = PandaArm()
        self.grasp_height = grasp_height

        self.workspace = np.array(((0.2, -0.15), # ((min_x, min_y)
                                   (0.5, 0.15))) #  (max_x, max_y))
        if render:
            self._draw_workspace()

        self.img_size = img_size
        self.camera = TopDownCamera(self.workspace, img_size)

        self.num_objects = num_objects
        self.obj_rotation_range = obj_rotation_range
        self.objects = []

    def reset(self):
        self.robot.reset()

        # remove old objects
        [o.remove() for o in self.objects]
        self.objects = []

        # add some padding to
        xy_min = self.workspace[0] + 0.02
        xy_max = self.workspace[1] - 0.02
        xys = np.random.uniform(xy_min, xy_max, size=(self.num_objects, 2))
        thetas = np.random.uniform(*self.obj_rotation_range, size=self.num_objects)
        for i in range(self.num_objects):
            position = np.array((*xys[i], 0.05))
            quat = pb.getQuaternionFromEuler((0, 0, thetas[i]))
            self.objects.append(Object(position, quat))
            [pb.stepSimulation() for _ in range(20)]

        # let all objects fall to ground
        [pb.stepSimulation() for _ in range(100)]

        # friction makes more realistic grasping, but can cause objects to 'stick'
        # to each other when falling
        [pb.changeDynamics(o.id, -1, lateralFriction=0.6) for o in self.objects]

    def render_image(self) -> np.ndarray:
        return self.camera.get_rgb_image()

    def execute_grasp(self, x: float, y: float, theta: float) -> bool:
        # teleport to prepick
        prepick_pos = (x, y, self.grasp_height+0.05)
        quat = pb.getQuaternionFromEuler((0, 0, theta))
        self.robot.moveTo(prepick_pos, quat, dynamic=False)

        # move to pick
        pick_pos = (x, y , self.grasp_height)
        self.robot.moveTo(pick_pos, quat, dynamic=True)

        # close gripper and early terminate
        fully_closed = self.robot.closeGripper()
        if fully_closed:
            # failed grasp because gripper has nothing in it
            return False

        # lift hand
        lift_pos = (x, y, 0.15)
        self.robot.moveTo(lift_pos, quat, dynamic=True)

        # check if any object is above ground plane
        [pb.stepSimulation() for _ in range(50)]
        if any(o.get_position()[2] > 0.05 for o in self.objects):
            return True

        return False

    def _convert_from_pixel(self, pxy: np.ndarray) -> np.ndarray:
        xy_norm = pxy.astype(float) / self.img_size

        xy = xy_norm * np.subtract(*self.workspace[::-1]) + self.workspace[0]
        return xy

    def _convert_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy_norm = np.subtract(xy, self.workspace[0]) \
                    / np.subtract(*self.workspace[::-1])
        xy_norm = np.clip(xy_norm, 0, 1)

        #xy axis are flipped from world to image space
        pxy = self.img_size * xy_norm
        return pxy.astype(int)

    def _draw_workspace(self) -> None:
        '''This is just for visualization purposes, to help you with the object
        resetting.  Must be in GUI mode, otherwise error occurs

        Note
        ----
        Pybullet debug lines dont show up in camera images
        '''
        corner_ids = ((0,0), (0,1), (1,1), (1,0), (0,0))
        for i in range(4):
            start = (*self.workspace[corner_ids[i],[0,1]], 0.01)
            end = (*self.workspace[corner_ids[i+1],[0,1]], 0.01)
            pb.addUserDebugLine(start, end, (0,0,0), 5)

if __name__ == "__main__":
    sim = Simulator()

    for _ in range(50):
        sim.reset()
        img = sim.render_image()

        # attempt grasp at occupied pixel in image to increase likelihood of success
        occ_mask = np.argwhere((img < 230).any(axis=2))
        pxy = occ_mask[np.random.randint(len(occ_mask))]

        x, y = sim._convert_from_pixel(pxy)
        if np.random.random() < 0.5:
            theta = 0
        else:
            theta = np.pi/2

        label = sim.execute_grasp(x, y, theta)
        time.sleep(0.1)
        print('SUCCESS' if label else 'FAILURE')
