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

from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import time

import cv2

mode = "haar" #"harris" "haar"

#i put a bunch of .sleep() functions when pybullet step is called
time_step = 0.00

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
            time.sleep(time_step)
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
            time.sleep(time_step)
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
            time_step_count = 0
            while not np.allclose(joint_pos, target_pose, atol=1e-3) and n_it < max_it:
                pb.stepSimulation()
                if time_step_count<100:
                    time.sleep(time_step)
                    time_step_count += 1
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

def analysis_dummy(img: np.ndarray):
    #the stack is at .5,.15
    #the blocks should be located away from the stack
    #blocks are between [.2, -.15] and [.4, .15]
    x = [.25, .25, .4]
    y = [-.1, .1,-.1]
    theta = [np.arctan(y[0]/x[0]), np.arctan(y[1]/x[1]),
        np.arctan(y[2]/x[2])]
    return x,y, theta

def analysis_harris(img: np.ndarray):
    #ONLY FOR TESTING
    # SAVE CAMERA OUTPUT AS PNG
    # LOAD PNG FOR LATER
    '''
    #https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    image = Image.fromarray(img, 'RGB')
    image.show()
    image.save("img.png")
    
    #load the png
    img_png = Image.open("img.png")
    img_png.show()
    img = np.array(img_png)
    '''
    img = img.astype('long')

    #define values
    img_size = img.shape[1]
    workspace = np.array(((0.18, -0.17), # ((min_x, min_y)
    (0.5, 0.17))) #  (max_x, max_y))

    #greyscale
    img_grey = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
    img_grey = -(img_grey-255)
    img_grey = gaussian_filter(img_grey, sigma=.5)
    img_grey = img_grey.astype('uint8')

    #pad img  
    img_pad = np.append(np.zeros(size:=(1,img_size)),img_grey,axis=0)
    img_pad = np.append(img_pad,np.zeros(size:=(1,img_size)),axis=0)
    img_pad = np.append(np.zeros(size:=(img_size+2,1)),img_pad,axis=1)
    img_pad = np.append(img_pad,np.zeros(size:=(img_size+2,1)),axis=1)
    img_pad = img_pad.astype('uint8')

    '''
    imagepad = Image.fromarray(img_pad, 'L')
    imagepad.show()
    imagepad.save("present/img_pad.png")
    '''

    #create arrays comparing pixels to neighbors
    Ix = np.zeros(size:=(img_size,img_size))#Ix
    Iy = np.zeros(size:=(img_size,img_size))#Iy
    sobel_X = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_Y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    for x in range(1,img_size+1):
        for y in range(1,img_size+1):
            sub_grid = img_pad[[x-1,x,x+1],:][:,[y-1,y,y+1]]
            Ix[x-1,y-1] = np.sum(np.multiply(sobel_X,sub_grid))
            Iy[x-1,y-1] = np.sum(np.multiply(sobel_Y,sub_grid))
    Ix = Ix.astype('uint8')
    Iy = Iy.astype('uint8')

    '''
    imageIx = Image.fromarray(Ix, 'L')    
    imageIy = Image.fromarray(Iy, 'L')
    imageIx.show()
    imageIy.show()
    imageIx.save("present/imageIx.png")
    imageIy.save("present/imageIy.png")
    '''
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    '''
    imageIxx = Image.fromarray(Ixx, 'L')
    imageIyy = Image.fromarray(Iyy, 'L')
    imageIxy = Image.fromarray(Ixy, 'L')
    imageIxx.show()
    imageIyy.show()
    imageIxy.show()
    imageIxx.save("present/imageIxx.png")
    imageIyy.save("present/imageIyy.png")
    imageIxy.save("present/imageIxy.png")
    '''

    #use the structural tensor M to get the harris value for each point
    hrs_vals = np.zeros(size:=(img_size,img_size))#harris values
    hrs_coords = [[]]
    k = 0.04
    for x in range(img_size):
        for y in range(img_size):
            M = np.array([[Ixx[x,y],Ixy[x,y]],[Ixy[x,y],Iyy[x,y]]])
            R = np.linalg.det(M)-k*np.square(np.trace(M))
            if R > 8000:
                hrs_vals[x,y]=R
                hrs_coords.append([x,y])
    del hrs_coords[0]

    hrs_vals = hrs_vals.astype('uint8')

    imageHar = Image.fromarray(hrs_vals, 'L')
    imageHar.show()
    #imageHar.save("present/imageHar.png")

    '''
    side of square is 71 pixels
    diagonal of square is ~101 pixels
    radius of circle around square is ~50 pixels
    '''

    #cycle through all the collected points
    #for each point, fit a square between it and all other points
    #count how many other points fit close to the lines of the square
    scores = [[]]
    score_index = 0
    for b in range(len(hrs_coords)):
        for t in range(len(hrs_coords)):
            score_index +=1
            p1 = np.array(hrs_coords[b])#corner 1
            p3 = np.array(hrs_coords[t])#corner 3

            dist_diag = np.linalg.norm(p1-p3)

            #only check points within expected range
            if dist_diag>90 and dist_diag < 110:
                c = ((p1+p3)/2).astype(int)
                p2 = np.array([int((p1[0]+p3[0]+p1[1]-p3[1])/2),
                    int((-p1[0]+p3[0]+p1[1]+p3[1])/2)])
                p4 = np.array([int((p1[0]+p3[0]-p1[1]+p3[1])/2),
                    int((p1[0]-p3[0]+p1[1]+p3[1])/2)])
                
                #now from points p1 and p3 we have 4 lines to make a square
                #find distance between all points and the square lines
                d1_s = []
                d2_s = []
                d3_s = []
                d4_s = []
                score = 0
                start2 = time.time()
                for p in hrs_coords:
                    #only check points within expected range
                    dist_radial = np.linalg.norm(c-p)
                    if dist_radial > 40 and dist_radial < 60:
                        #line p1 p2
                        L1 = p1
                        L2 = p2
                        num = np.linalg.norm(np.cross(L2-L1, L1-p))
                        den = np.linalg.norm(L2-L1)
                        if den != 0:
                            d1 = num/den
                        else:
                            d1 = np.linalg.norm(L1-p)
                        d1_s.append(int(d1))
                        #line p2 p3
                        L1 = p2
                        L2 = p3
                        num = np.linalg.norm(np.cross(L2-L1, L1-p))
                        den = np.linalg.norm(L2-L1)
                        if den != 0:
                            d2 = num/den
                        else:
                            d2 = np.linalg.norm(L1-p)
                        d2_s.append(int(d2))
                        #line p3 p4
                        L1 = p3
                        L2 = p4
                        num = np.linalg.norm(np.cross(L2-L1, L1-p))
                        den = np.linalg.norm(L2-L1)
                        if den != 0:
                            d3 = num/den
                        else:
                            d3 = np.linalg.norm(L1-p)
                        d3_s.append(int(d3))
                        #line p4 p1
                        L1 = p4
                        L2 = p1
                        num = np.linalg.norm(np.cross(L2-L1, L1-p))
                        den = np.linalg.norm(L2-L1)
                        if den != 0:
                            d4 = num/den
                        else:
                            d4 = np.linalg.norm(L1-p)
                        d4_s.append(int(d4))
                        #add to score if distance is below a certain value
                        co = 6 #cutoff
                        if d1 < co or d2 < co or d2 < co or d2 < co:
                            score += 1

                scores.append([b,t,c[0],c[1],score])

    del scores[0]

    #scores
    #col 0 = index of point 1 in har_pts
    #col 1 = index of point 2 in har_pts
    #col 2 = x location of center
    #col 3 = y location of center
    #col 4 = score of configuration
    score_array = np.array(scores)

    x = []
    y = []
    theta = []

    for i in range(3):
        #find max score
        max_index = np.argmax(score_array[:,4],axis=0)
        max_score = score_array[max_index]

        np.subtract(*workspace[::-1]) + workspace[0]

        #get corresponding center
        px = max_score[2]
        py = max_score[3]
        x_scale = workspace[1,0]-workspace[0,0]
        y_scale = workspace[1,1]-workspace[0,1]
        x.append(  (px/img_size) * x_scale + workspace[0,0])
        y.append(  (py/img_size) * y_scale + workspace[0,1])

        #get rotation
        p1 = np.array(hrs_coords[max_score[0]])#corner 1
        p3 = np.array(hrs_coords[max_score[1]])#corner 3
        p2 = np.array([int((p1[0]+p3[0]+p1[1]-p3[1])/2),
            int((-p1[0]+p3[0]+p1[1]+p3[1])/2)])
        p4 = np.array([int((p1[0]+p3[0]-p1[1]+p3[1])/2),
            int((p1[0]-p3[0]+p1[1]+p3[1])/2)])
        
        if p1[0] == p2[0] or p1[1] == p2[1]:#if flat
            th = 0
        else:
            all_points = np.array([p1,p2,p3,p4])
            top_index = all_points[:,1].argmax()
            top_point = all_points[top_index]
            bottom_index = all_points[:,1].argmin()
            bottom_point = all_points[bottom_index]
            th = np.arctan( (top_point[1]-bottom_point[1]) / (top_point[0]-bottom_point[0]) )

        #theta is the angle of the diagonal, change it to square angle
        #simplify to -45 45 deg range
        if th > 0:
            th = (th - np.pi/4)
        elif th < 0:
            th = (th + np.pi/4)

        theta.append(th)
        
        #delete points next to max score point
        c_max = np.array([max_score[2],max_score[3]])
        keep_list = []
        for s in range(len(score_array)-1,-1,-1): #count backwards to delete
            c_test = np.array([score_array[s,2],score_array[s,3]])
            d_test = np.linalg.norm(c_max-c_test)
            if d_test > 20:
                keep_list.append(s)
        score_array = score_array[keep_list]

        
        #generate image for presentation
        im = Image.new('RGB', (img_size, img_size), (128, 128, 128))
        im.paste(imageHar)
        draw = ImageDraw.Draw(im)
        draw.polygon(((p1[1],p1[0]), (p2[1],p2[0]),
            (p3[1],p3[0]), (p4[1],p4[0])), outline=(255, 255, 0))
        draw.point(((c_max[1],c_max[0])),fill = (255, 255, 0))
        im.show()
        

    return x,y, theta

def analysis_Haar(img: np.ndarray):
    x_meters = []
    y_meters = []

    # Load Haar cascade classifier for block detection
    cascade = cv2.CascadeClassifier('classifier/cascade.xml')

    print(cascade.empty())
    
    img = np.array(img)
    img_size = img.shape[1]

    #define values
    workspace = np.array(((0.18, -0.17), # ((min_x, min_y)
    (0.5, 0.17))) #  (max_x, max_y))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect blocks using Haar cascade
    blocks = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), maxSize=(110, 110))

    for (x, y, w, h) in blocks:
        # Calculate x, y position of center of block
        center_x = int(y + h/2)
        center_y = int(x + w/2)
        #print(center_x,",", center_y)
        
        # Convert position to meters
        x_scale = workspace[1,0]-workspace[0,0]
        y_scale = workspace[1,1]-workspace[0,1]
        x_meter = (center_x/img_size) * x_scale + workspace[0,0]
        y_meter = (center_y/img_size) * y_scale + workspace[0,1]
        x_meters.append(x_meter)
        y_meters.append(y_meter)

        #draw image
        dist = 35
        c = np.array([center_x,center_y])#center
        p1 = np.array([center_x-dist,center_y+dist])
        p2 = np.array([center_x+dist,center_y+dist])
        p3 = np.array([center_x+dist,center_y-dist])
        p4 = np.array([center_x-dist,center_y-dist])

        #generate image for presentation
        background = Image.fromarray(gray, 'L')
        im = Image.new('RGB', (img_size, img_size), (128, 128, 128))
        im.paste(background)
        draw = ImageDraw.Draw(im)
        draw.polygon(((p1[1],p1[0]), (p2[1],p2[0]),
            (p3[1],p3[0]), (p4[1],p4[0])), outline=(255, 0, 0))
        draw.point(((c[1],c[0])),fill = (255, 0, 0))
        im.show()
    
    print(x_meters)
    print(y_meters)

    theta = [0,0,0]

    return x_meters, y_meters, theta

if __name__ == "__main__":
    #prepare simulator
    sim = Simulator()
    workspace = np.array(((0.18, -0.17), # ((min_x, min_y)
        (0.5, 0.17))) #  (max_x, max_y))
    
    #generate random positions and thetas
    num_objects = 3
    xy_min = workspace[0] + 0.07
    xy_max = workspace[1] - 0.07
    
    #generate 3 random objects in their own sub-areas (to avoid overlap)
    #xys_spawn = np.random.uniform(xy_min, xy_max, size=(num_objects, 2))
    xys_spawn1 = np.random.uniform([ 0.25, -0.1 ], [ 0.27, -0.05 ], size=(1, 2))
    xys_spawn2 = np.random.uniform([ 0.25, 0.1 ], [ 0.27, 0.05 ], size=(1, 2))
    xys_spawn3 = np.random.uniform([ 0.4, 0.1 ], [ 0.38, 0 ], size=(1, 2))
    xys_spawn = np.array([xys_spawn1[0],xys_spawn2[0],xys_spawn3[0]])
    
    #random theta with 90 deg range because square is same at -45 and 45
    theta_min = -np.pi/4
    theta_max = np.pi/4
    theta_spawn = np.random.uniform(theta_min, theta_max, size=(num_objects,1))

    """SPAWN BLOCKS AT TEST LOCATIONS"""
    """
    #COMMENT OUT WHEN NOT TESTING
    x = [.25, .25, .4]
    y = [-.1, .1,-.1]
    xys_spawn = np.array([[x[0],y[0]],[x[1],y[1]],[x[2],y[2]]])
    theta_spawn = [np.arctan(y[0]/x[0]), np.arctan(y[1]/x[1]),
        np.arctan(y[2]/x[2])]
    ###################################
    """

    #spawn blocks
    objects = []
    for i in range(num_objects):
        position = np.array((*xys_spawn[i], 0.05))
        quat = pb.getQuaternionFromEuler((0, 0, theta_spawn[i]))
        objects.append(Object(position, quat))
        [pb.stepSimulation() for _ in range(20)]

    #prepare camera
    camera = TopDownCamera(workspace, 512)

    #get image from camera, extract coordinates
    img = camera.get_rgb_image()
    start_time = time.time()

    if mode == "dummy":
        x,y,theta = analysis_dummy(img)
        #give the robot the block coordinates
        x = [xys_spawn[0][0],xys_spawn[1][0],xys_spawn[2][0]]
        y = [xys_spawn[0][1],xys_spawn[1][1],xys_spawn[2][1]]
        theta = theta_spawn
    elif mode == "harris":
        x,y,theta = analysis_harris(img)
    elif mode == "haar":
        x,y,theta = analysis_Haar(img)
    
    print(np.multiply(theta,180/np.pi))

    stop_time = time.time()
    print(stop_time-start_time)

    #prepare robot
    robot = PandaArm()
    grasp_height = 0.01

    #stacking loop
    for i in range(3):        
        # move to prepick
        prepick_pos = (x[i], y[i], grasp_height+0.05)
        quat = pb.getQuaternionFromEuler((0, 0, theta[i]))
        robot.moveTo(prepick_pos, quat, dynamic=True)

        # move to pick
        pick_pos = (x[i], y[i] , grasp_height)
        robot.moveTo(pick_pos, quat, dynamic=True)

        # close gripper
        fully_closed = robot.closeGripper()

        # lift hand
        lift_pos = (x[i], y[i], 0.15)
        robot.moveTo(lift_pos, quat, dynamic=True)

        #move to pre-stack        
        stack_pos = (.5,.2,grasp_height+(i+1)*0.05)
        quat = pb.getQuaternionFromEuler((0, 0, 0))
        robot.moveTo(stack_pos, quat, dynamic=True)

        #move to stack
        stack_pos = (.5,.2,grasp_height+(i)*0.05)
        robot.moveTo(stack_pos, quat, dynamic=True)

        #open gripper
        robot.openGripper()

        # lift hand
        stack_lift_pos = (.5,.2,grasp_height+(i)*0.05+.1)
        robot.moveTo(stack_lift_pos, quat, dynamic=True)

        time.sleep(0.1)

        #go to home position
        robot.reset()

        time.sleep(0.1)

#changes:
    #added harris method
    #changed block theta range

#sources:

#harris method
    #https://www.baeldung.com/cs/harris-corner-detection
    #lecture 13 ecce 5554 prof singh

    #geometry
    #https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    #https://www.quora.com/Given-two-diagonally-opposite-points-of-a-square-how-can-I-find-out-the-other-two-points-in-terms-of-the-coordinates-of-the-known-points

    #using images
    #https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image

    #drawing in python
    #https://note.nkmk.me/en/python-pillow-imagedraw/