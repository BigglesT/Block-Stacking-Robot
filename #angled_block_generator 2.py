#angled block generator
import time
import glob
from typing import Union, Tuple
import os
from collections import deque
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.spatial.transform
import pybullet as pb
import pybullet_data

from PIL import Image

#name = "z"
angle = np.pi*0 #-np.pi/8 #0
count = 20

class Object:
    def __init__(self, position: ArrayLike, quat: ArrayLike) -> None:
        '''Instantiates random household object in simulator at specified
        position and rotation quaternion.  Color is randomly assigned to a valid
        RGB value.
        '''
        urdf_paths = glob.glob('assets/block/*/*.urdf')
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

        cam_height = 1
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

        rgba = np.reshape(rgba, (self.img_size, self.img_size, 4))
        return rgba[..., :3]

class Simulator:
    def __init__(self,
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

if __name__ == "__main__":
    #prepare simulator
    sim = Simulator()
    workspace = np.array(((0.18, -0.17), # ((min_x, min_y)
        (0.5, 0.17))) #  (max_x, max_y))
    
    #generate random positions and thetas
    num_objects = 1
    
    #generate 1 random objects in their own sub-areas (to avoid overlap)
    #xys_spawn = np.random.uniform(xy_min, xy_max, size=(num_objects, 2))
    xys_spawn = np.array([ 0.34, 0 ])#middle
    
    #spawn theta
    theta_spawn = angle

    #prepare camera
    camera = TopDownCamera(workspace, 512)

    #spawn blocks
    for i in range(count):
        position = np.array((*xys_spawn, 0.05))
        quat = pb.getQuaternionFromEuler((0, 0, theta_spawn))
        obj = Object(position, quat)
        [pb.stepSimulation() for _ in range(20)]
        img = camera.get_rgb_image()
        image = Image.fromarray(img, 'RGB')
        image.save("imgz"+str(i+1)+".png")
        obj.remove()
