#!/usr/bin/env python

import roslib
import sys
import rospy
import numpy as np

from baxter_interface import Navigator
from baxter_interface import Limb
from baxter_interface import Gripper
from baxter_interface.camera import CameraController

from baxter_pykdl import baxter_kinematics
from cv_bridge import CvBridge
import cv2

from ar_track_alvar_msgs.msg import AlvarMarkers
from sensor_msgs.msg import Image
import std_srvs.srv
from baxter_core_msgs.srv import (
    SolvePositionIK, SolvePositionIKRequest)
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point
)

class BaxterArm:
	
	def __init__(self, arm, robot):
		self.robot = robot
		self.arm = arm
		self.navigator = Navigator(arm)
		self.limb = Limb(arm)
		self.gripper = Gripper(arm)
		self.gripper.calibrate()
		self.joint_names = self.limb.joint_names()
		self.kin = baxter_kinematics(arm)
		print("")

		self.acq_done = False

		self.is_object_viewed = True
		self.is_on_object = False
		self.gain = 1					# Gain for the command law
		self.seuil_min = 0.0001				# Minimal celerety for a joint
		self.seuil_max = 0.3				# Maximal celerety for a joint
		self.pose_tolerance = 0.015			# Pose tolerance between s and s*

		self.base_frame = "/base"
		self.topic_camera = "/cameras/" + arm + "_hand_camera/image"
		self.topic_alvar_camera = "/ar_pose_marker_" + arm
		self.topic_alvar_base = "/ar_pose_marker_" + arm + "_to_base"
		self.image_acquired_path = "../Stage_Baxter/ros_ws/src/" + \
			"pick_and_place/src/VSMonoAR/acquired_image.jpeg"
		self.ar_corners_path = "../Stage_Baxter/ros_ws/src/" + \
			"pick_and_place/src/VSMonoAR/ar_corners.txt"
		self.ar_size = 0.051				# size of one of the ar-tag's side in 										meter
		self.text_acquisition_done = "Here is the acquisition wich has been done"

		self.width        = 1280                        # Camera width resolution
		self.height       = 800				# Camera height resolution
		self.focale = 406.108727421			# Camera focale

		self.resetCameras()
		self.openCamera(self.width, self.height)

	#########################
	# 	cameras		#
	#########################

	def resetCameras(self):
		"""
			Reset all cameras (incase cameras fail to be recognised on boot)
		"""
		reset_srv = rospy.ServiceProxy("cameras/reset", std_srvs.srv.Empty)
		rospy.wait_for_service("cameras/reset", timeout=10)
		reset_srv()

	def openCamera(self, width, height):
		"""
			Open a camera with given resolution and set camera parameters

			:param width: The width resolution of the camera
			:param height: The height resolution of the camera
		"""
		print("... opening " + self.arm + " camera ...")
		cam = CameraController(self.arm + "_hand_camera")

		# set camera parameters
		cam.resolution          = (int(width), int(height))

		cam.exposure            = -1             # range, 0-100 auto = -1
		cam.gain                = -1             # range, 0-79 auto = -1
		cam.white_balance_blue  = -1             # range 0-4095, auto = -1
		cam.white_balance_green = -1             # range 0-4095, auto = -1
		cam.white_balance_red   = -1             # range 0-4095, auto = -1"""

		# open camera
		cam.open()

		print("... " + self.arm + " camera opened ...")

	#########################
	# 	callbacks 	#
	#########################

	def getImage(self):
		"""
			Get the image from the camera of the arm and casting it into opencv image
		"""
		msg=rospy.wait_for_message(self.topic_camera, Image)
		bridge = CvBridge()
		self.img = bridge.imgmsg_to_cv2(msg, "bgr8")

	def getARInfo(self):
		"""
			Get the list of ar-tag detected by ar track alvar in the camera frame
		"""
		msg=rospy.wait_for_message(self.topic_alvar_camera, AlvarMarkers)
		self.list_ar = msg.markers

	def getARBaseInfo(self):
		"""
			Get the list of ar-tag detected by ar track alvar in the base frame
		"""
		msg=rospy.wait_for_message(self.topic_alvar_base, AlvarMarkers)
		self.list_ar_base = msg.markers

	#########################
	# 	input		#
	#########################

	def waitingWheelButtonPressed(self):
		"""
			Wait until the wheel button of the baxter's arm	is detected pressed
		"""
		while not self.navigator.button0:
			continue

	#########################
	# 	output 		#
	#########################

	def drawARCorners(self, img, ar_crns):
		"""
			Draw the corners and the center of the ar-tag on the opencv image

			:param img: The opencv image on wich we want to draw
			:param ar_crns: The array of the position of the corners and
				center of the ar-tag
		"""
		cv2.circle(img, ar_crns[0], 2, (0, 255, 0), -1)
		cv2.circle(img, ar_crns[1], 2, (0, 0, 0), -1)
		cv2.circle(img, ar_crns[2], 2, (255, 0, 0), -1)
		cv2.circle(img, ar_crns[3], 2, (0, 0, 255), -1)
		cv2.circle(img, ar_crns[4], 2, (0, 0, 0), -1)

	def showImage(self, img, is_acquisition):
		"""
			Display the opencv image given and a text if necessary

			:param img: The opencv image that we want to display
			:param is_acquisition: The boolean saying if it is an
				image from acquisition phase
		"""
		cv2.imshow('img', img)
		if is_acquisition:
			print(self.text_acquisition_done)
		cv2.waitKey(100)

	#########################################
	# 	frame system changes 		#
	#########################################

	def pixelToCamera(self, px):
		"""
			Convert image pixel to camera point

			:param px: The pixel we want to convert into point
			:return: The point computed from the pixel
		"""
		x = (px[1] - (self.width / 2)) / self.focale
		y = (px[0] - (self.height / 2)) / self.focale

		return (x, y)

	def cameraToPixel(self, pt):
		"""
			Convert camera point to image pixel

			:param pt: The point we want to convert into pixel
			:return: The pixel computed from the point

				todo:: understand why there is a correction of +10 needed
		"""
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		x = int(round(pt[0] / pt[2] * self.focale + self.width / 2)) + 10
		y = int(round(pt[1] / pt[2] * self.focale + self.height / 2)) + 10

		return (x, y)

	def arFrameToCam(self, list_ar, ar_corners):
		"""
			Transfrom ar-tag's corners coordinate in ar-tag frame
			into ar-tag corners coordinates in camera frame

			:param list_ar: The list of the ar-tag detected by ar track alvar
			:param ar_corners: The coordinates of the corners of the ar-tag in
				ar-tag frame
			:return: The coordinates of the corners of the ar-tag in camera frame
		"""
		ar_pos = list_ar[0].pose.pose.position
		ar_ori = list_ar[0].pose.pose.orientation

		a = ar_ori.w
		b = ar_ori.x
		c = ar_ori.y
		d = ar_ori.z
		R00 = (a*a)+(b*b)-(c*c)-(d*d)
		R01 = 2*b*c-2*a*d
		R02 = 2*a*c+2*b*d
		R10 = 2*a*d+2*b*c
		R11 = (a*a)-(b*b)+(c*c)-(d*d)
		R12 = 2*c*d-2*a*b
		R20 = 2*b*d-2*a*c
		R21 = 2*a*b+2*c*d
		R22 = (a*a)-(b*b)-(c*c)+(d*d)

		Tcam_to_amer = np.matrix([[R00,R01,R02,ar_pos.x], \
					[R10,R11,R12,ar_pos.y], \
					[R20,R21,R22,ar_pos.z], \
					[0,  0,  0,  1]])
		
		ar_corners_cam = []
		for corner in ar_corners:
			ar_corners_cam.append(Tcam_to_amer*corner)

		return ar_corners_cam

	def camFrameTo2d(self, ar_crns_cam):
		"""
			Transfrom ar-tag's corners coordinates in camera frame
			into 2d points on the image from the camera

			:param ar_crns_cam: The coordinates of the corners of the ar in camera frame
			:return: The coordinates in 2d of the corners of the ar-tag on the image
				from the camera
		"""
		ar_corners_2d = []
		for ar_corner in ar_crns_cam:
			ar_corners_2d.append(self.cameraToPixel(ar_corner))

		return ar_corners_2d

	#################################
	# 	Tools' methods 		#
	#################################

	def getARCorners(self):
		"""
			Compute the coordinates of the ar-tag's corners in the ar-tag frame

			:return: The coordinates of the corners of the ar-tag in ar-tag frame
		"""
		dist_to_corner = self.ar_size / 2
		z = 0

		x = -dist_to_corner
		y = -dist_to_corner
		bot_left_crn = np.matrix([[x],[y],[z],[1]])
		x = -dist_to_corner
		y = dist_to_corner
		up_left_crn = np.matrix([[x],[y],[z],[1]])
		x = dist_to_corner
		y = dist_to_corner
		up_right_crn = np.matrix([[x],[y],[z],[1]])
		x = dist_to_corner
		y = -dist_to_corner
		bot_right_crn = np.matrix([[x],[y],[z],[1]])
		x = 0
		y = 0
		center = np.matrix([[x],[y],[z],[1]])

		return [bot_left_crn, up_left_crn, up_right_crn, bot_right_crn, center]

	def setARCorners(self, list_ar):
		"""
			Compute ar-tag corners 2d coordinates in the image from the ar-tag detected
				by ar track alvar

			:param list_ar: The list of the ar-tag detected by ar track alvar
			:return: The coordinates in 2d of the corners of the ar-tag on the image
				from the camera
		"""
		ar_corners_2d = []
		if list_ar:
			ar_corners = self.getARCorners()
			ar_corners_cam = self.arFrameToCam(list_ar, ar_corners)
			ar_corners_2d = self.camFrameTo2d(ar_corners_cam)
				
		return ar_corners_2d

	#########################
	# 	moving 		#
	#########################

	def settingPoseForIkService(self, frame, pos, ori):
		"""
			Transform a position and an orientation in a pose in a given frame
			usable by the IkService of baxter

			:param frame: The frame in wich the pose is correct
			:param pos: The position of the pose
			:param ori: The orientation of the pose
			:return: The pose usable by the IkService
		"""
		hdr = Header(stamp=rospy.Time.now(), frame_id=frame)
		pose = PoseStamped(header=hdr, pose=Pose(position=pos,orientation=ori))

		return pose

	def callingIkService(self, pose):
		"""
			Call the IkService of baxter with the given pose
			to move the arm to this pose

			:param pose: The pose to where we want the arm to move
		"""
		node = "ExternalTools/" + self.arm + "/PositionKinematicsNode/IKService"
		ik_service = rospy.ServiceProxy(node, SolvePositionIK)
		ik_request = SolvePositionIKRequest()
		ik_request.pose_stamp.append(pose)

		try:
			rospy.wait_for_service(node, 5.0)
			ik_response = ik_service(ik_request)
		except (rospy.ServiceException, rospy.ROSException), error_message:
			rospy.logerr("Service request failed: %s" % (error_message,))
			sys.exit("ERROR - baxter_ik_move - Failed to append pose")

		if ik_response.isValid[0]:
			print("PASS: Valid joint configuration found")
			# convert response to joint position control dictionary
			limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
			# move limb
			self.limb.move_to_joint_positions(limb_joints)
		else:
			print("fail to find a valid configuration")

	def moveToStartingPose(self):
		"""
			Move to the visual servoing starting pose for the arm
		"""
		curr_pose = self.limb.endpoint_pose()
		new_pos = Point()
		new_pos.x = curr_pose["position"].x
		new_pos.y = curr_pose["position"].y
		new_pos.z = curr_pose["position"].z - 0.1
		new_ori = curr_pose["orientation"]
		new_pose = self.settingPoseForIkService(self.base_frame, new_pos, new_ori)
		self.callingIkService(new_pose)

	def moveToArPos(self, arm_ori):
		"""
			Move to the ar-tag position with the orientation of the arm given

			:param arm_ori: The orientation of the arm we want to go
		"""
		print("... Moving to ar-tag pos ...")
		self.getARBaseInfo()
		if self.list_ar_base:
			ar_pose = self.list_ar_base[0]
			ar_pos = ar_pose.pose.pose.position
			ar_pos.z -= 0.07
			new_pose = self.settingPoseForIkService(self.base_frame, ar_pos, arm_ori)
			self.callingIkService(new_pose)
		else:
			print("The ar-tag is no more visible")

	def moveToArmPose(self, arm_pos, arm_ori):
		"""
			Move to the arm position and orientation given

			:param arm_pos: The position of the arm we want to go
			:param arm_ori: The orientation of the arm we want to go
		"""
		print("... Moving to arm previous pose ...")
		new_pose = self.settingPoseForIkService(self.base_frame, arm_pos, arm_ori)
		self.callingIkService(new_pose)

	def moveToDropPose(self):
		"""
			Move the arm to the saved drop pose
		"""
		print("... Moving to drop pose ...")
		pos = self.drop_pose["position"]
		ori = self.drop_pose["orientation"]
		arm_pose = self.settingPoseForIkService(self.base_frame, pos, ori)
	    	self.callingIkService(arm_pose)

	def applyQDot(self, q_dot):
		"""
			Apply the q dot values given to each joint

			:param q_dot: The q dot value for each joint
		"""
		cmd = {}
		for idx, name in enumerate(self.joint_names):
			v = q_dot[idx]
			cmd[name] = v
			
		self.limb.set_joint_velocities(cmd)

	#################################
	# 	learning phase 		#
	#################################

	def positionningArmToAcquire(self):
		"""
			Wait for the user to place the arm in position for acquisition and validate
			by pressing wheel button
		"""
		print("Please put the gripper on the object you want to grab." +
			" Then press on the wheel button")
		self.waitingWheelButtonPressed()

	def writeARInfo(self, z_ar, ar_corners):
		"""
			Write the height of the ar-tag and the ar-tag's corners coordinates
			at the time of the acquisition in a file

			:param z_ar: The altitude of the ar-tag at the time of the acquisition
			:param ar_corners: The coordinates of the corners of the ar-tag at the time
				of the acquisition
		"""
		file = open(self.ar_corners_path, "w")
		file.write(str(z_ar) + "\n")
		for ar_corner in ar_corners:
			file.write(str(ar_corner[0]) + "," + str(ar_corner[1]) + "\n")
		file.close()

	def validatingAcquisition(self, img, z_ar, ar_corners):
		"""
			Ask to the user if the acquisition is valid for him and wait for his answer

			:param img: The image of the acquisition
			:param z_ar: The altitude of the ar-tag at the time of the acquisition
			:ar_corners: The coordinates of the ar-tag's corners at te time of the
				acquisition
		"""
		valid_input = False
		while not valid_input:
			answer = raw_input("Is the object detection ok to you ? (y/n)")
			if answer == "y":
				valid_input = True
				self.acq_done = True
				cv2.imwrite(self.image_acquired_path, img)
				self.writeARInfo(z_ar, ar_corners)
			elif answer == "n":
				valid_input = True
			else:
				print("Answer must be 'y' or 'n'. Please enter your answer again")

		if not self.acq_done:
			print("Acquisition incorrect please do it again")

	def acquisition(self):
		"""
			Do the acquisition of the learning image for picking the object
			then validate it with the user
		"""
		while not self.acq_done:
			print("... Starting acquisition ...")
			self.positionningArmToAcquire()
			self.getImage()
			self.getARInfo()
			if self.list_ar:
				z_ar = -self.list_ar[0].pose.pose.position.z
				ar_corners_2d = self.setARCorners(self.list_ar)
				self.drawARCorners(self.img, ar_corners_2d)
				self.showImage(self.img, is_acquisition = True)
				self.validatingAcquisition(self.img, z_ar, ar_corners_2d)
			else:
				print("ar-tag not visible")
				print("Acquisition incorrect please do it again")
		print("... Acquisition done ...")
		cv2.destroyAllWindows()

	#########################################
	# 	Pick and place phase 		#
	#########################################

	def readARInfo(self):
		"""
			Read in a file and save the altitude of the ar-tag and the coordinates of
			the corners at the time of the acquisition

			:return: The coordinates of the the ar-tag's corners at the time of
				the acquisition
		"""
		try:
			file = open(self.ar_corners_path, "r")
		except:
			print("Please do the learning scenario at least once before the pick" + 
				" and place one")
			sys.exit(0)
		curr_line = file.readline()
		z_des = curr_line.split("\n")
		self.z_des = float(z_des[0])
		curr_line = file.readline()
		corners = []
		while curr_line != "":
			corner = []
			corner_pos = curr_line.split(",")
			for coord in corner_pos:
				corner.append(int(coord))
			corners.append(tuple(corner))
			curr_line = file.readline()

		return corners

	def setDesiredCorners(self):
		"""
			Get the acquisition image and the ar-tag's information at the time of the
			acquisition then show it to the user

			:return: The coordinates of the ar-tag's corners at the time of the
				acquisition
		"""
		des_img = cv2.imread(self.image_acquired_path)
		des_ar_corners = self.readARInfo()
		self.drawARCorners(des_img, des_ar_corners)
		self.showImage(des_img, is_acquisition = True)
		
		return des_ar_corners

	def gettingDropPose(self):
		"""
			Wait for the user to place the arm in position for dropping object
			and validate by pressing wheel button to save the current pose of
			the arm
		"""
		print("Please put the gripper in the place you want it to " +
			" drop the object. Then press on the wheel button")
		self.waitingWheelButtonPressed()
		self.drop_pose = self.limb.endpoint_pose()

	def setCurrentCorners(self, des_ar_corners):
		"""
			Get the current image of the camera and get and use the current ar-tag's 				informations then show it to the user

			:return: The current coordinates of the ar-tag's corners
		"""
		self.getImage()
		self.getARInfo()
		curr_ar_corners = self.setARCorners(self.list_ar)
		if curr_ar_corners:
			if not self.is_object_viewed:
				print("... Object detected ...")
				self.is_object_viewed = True

			self.drawARCorners(self.img, des_ar_corners)
			self.drawARCorners(self.img, curr_ar_corners)
			self.showImage(self.img, is_acquisition = False)

		else:
			if self.is_object_viewed:
				print("... Object is not on visual ...")
				self.is_object_viewed = False

		return curr_ar_corners

	def isPoseReached(self, e):
		"""
			Verify if the arm of the robot has reached the pose desired

			:param e: The error of the joints
			:return: True if the pose is reached False otherwise
		"""
		pose_reached = abs(e[0]) < self.pose_tolerance and \
			abs(e[1]) < self.pose_tolerance and \
			abs(e[2]) < self.pose_tolerance and \
			abs(e[3]) < self.pose_tolerance and \
			abs(e[4]) < self.pose_tolerance and \
			abs(e[5]) < self.pose_tolerance and \
			abs(e[6]) < self.pose_tolerance and \
			abs(e[7]) < self.pose_tolerance

		return pose_reached

	def computeL(self, pt_des, z):
		"""
			Compute the camera matrix interaction for a given point

			:param pt_des: The coordinates of the desired point
			:param z: The altitude of the camera to the desired point
			:return: The camera matrix interaction
		"""
		L = [[-1/z, 0.0, pt_des[0]/z, pt_des[0]*pt_des[1], \
			-(1+np.square(pt_des[0])), pt_des[1]] , \
		     [0.0, -1/z, pt_des[1]/z, 1+np.square(pt_des[1]), \
			-pt_des[0]*pt_des[1], -pt_des[0]]]
		
		return L

	def computeQDot(self, des_ar_corners, curr_ar_corners):
		"""
			Compute the error from the desired pose and the current pose of the
			ar-tag then compute the value of the command law (q dot for each joint)

			:param des_ar_corners: The desired coordinates of the ar-tag's corners
			:param curr_ar_corners: The current coordinates of the ar-tag's corners
			:return: The value of q dot for each joint
		"""
		pt_curr1 = self.pixelToCamera(des_ar_corners[0])
		pt_curr2 = self.pixelToCamera(des_ar_corners[1])
		pt_curr3 = self.pixelToCamera(des_ar_corners[2])
		pt_curr4 = self.pixelToCamera(des_ar_corners[3])
		s = np.array([pt_curr1[0], pt_curr1[1], pt_curr2[0], pt_curr2[1], \
			pt_curr3[0], pt_curr3[1], pt_curr4[0], pt_curr4[1]])

		pt_des1 = self.pixelToCamera(curr_ar_corners[0])
		pt_des2 = self.pixelToCamera(curr_ar_corners[1])
		pt_des3 = self.pixelToCamera(curr_ar_corners[2])
		pt_des4 = self.pixelToCamera(curr_ar_corners[3])
		s_des = np.array([pt_des1[0], pt_des1[1], pt_des2[0], pt_des2[1], \
			pt_des3[0], pt_des3[1], pt_des4[0], pt_des4[1]])

		e = s - s_des

		if(self.isPoseReached(e)):
			q_dot = np.zeros(7)
			self.is_on_object = True

		else:
			L1 = self.computeL(pt_des1, self.z_des)
			L2 = self.computeL(pt_des2, self.z_des)
			L3 = self.computeL(pt_des3, self.z_des)
			L4 = self.computeL(pt_des4, self.z_des)
			
			L = np.append(L1, L2, axis=0)
			L = np.append(L, L3, axis=0)
			L = np.append(L, L4, axis=0)

			q_dot = np.linalg.pinv(L*self.kin.jacobian()).dot(-self.gain*e)
			q_dot = np.transpose(q_dot)

			is_q_dot_ok = False
			
			while not is_q_dot_ok :
				min_coeff = abs(q_dot).min(axis = 0)
				max_coeff = abs(q_dot).min(axis = 0)
				if max_coeff > self.seuil_max:
					q_dot = q_dot*self.seuil_max/max_coeff
					is_q_dot_ok = True
					#print("Celerity lowered")
				elif min_coeff < self.seuil_min:
					q_dot = q_dot*self.seuil_min/min_coeff
					#print("Celerity increased")
				else:
					is_q_dot_ok = True

		return q_dot

	def visualServoing(self):
		"""
			Do the visual servoing of the arm to the ar-tag
		"""
		print("... Starting visual servoing ...")
		des_ar_corners = self.setDesiredCorners()
		self.gettingDropPose()
		self.robot.resettingRobot()
		while (not self.is_on_object) and (not rospy.is_shutdown()):
			curr_ar_corners = self.setCurrentCorners(des_ar_corners)
			if self.is_object_viewed:
				q_dot = self.computeQDot(curr_ar_corners, des_ar_corners)
				self.applyQDot(q_dot)
		print("... Visual servoing done ...")
		cv2.destroyAllWindows()

	def pickAndPlace(self):
		"""
			Pick the object, move it to the drop pose then drop it
		"""
		print("... Starting pick and place of the object ...")
		arm_pose = self.limb.endpoint_pose()
		arm_pos = arm_pose["position"]
		arm_ori = arm_pose["orientation"]
		self.moveToArPos(arm_ori)
		self.gripper.close()
		self.moveToArmPose(arm_pos, arm_ori)
		self.moveToDropPose()
		self.gripper.open()
		print("... Pick and place done ...")

def main(args):
	rospy.init_node("BaxterArm", anonymous=True)
	print("unitary test BaxterArm")

if __name__ == "__main__":
	main(sys.argv)
