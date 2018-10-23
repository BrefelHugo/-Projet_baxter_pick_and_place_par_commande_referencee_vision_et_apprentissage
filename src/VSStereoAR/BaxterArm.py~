#!/usr/bin/env python

import roslib
import sys
import rospy
import numpy as np
import math

from baxter_interface import Navigator
from baxter_interface import Limb
from baxter_interface import Gripper
from baxter_interface.camera import CameraController

from baxter_pykdl import baxter_kinematics
from cv_bridge import CvBridge
import cv2
import tf

from ar_track_alvar_msgs.msg import AlvarMarkers
from sensor_msgs.msg import Image
import std_srvs.srv
from baxter_core_msgs.srv import (
    SolvePositionIK, SolvePositionIKRequest)
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

class BaxterArm:
	
	def __init__(self, arm, robot):
		rospy.on_shutdown(self.manualShutdown)
		self.robot = robot
		self.arm = arm
		self.navigator = Navigator(arm)
		self.limb = Limb(arm)
		self.gripper = Gripper(arm)
		self.gripper.calibrate()
		self.joint_names = self.limb.joint_names()
		self.kin = baxter_kinematics(arm)
		self.listener = tf.TransformListener()
		print("")

		self.acq_done = False

		self.is_object_viewed = True
		self.is_on_object = False
		self.gain = 0.8					# Gain for the command law
		self.seuil_min = 0.0001				# Minimal celerety for a joint
		self.seuil_max = 0.7				# Maximal celerety for a joint
		self.pose_tolerance = 0.015			# Pose tolerance between s and s*

		self.ar_size = 0.051				# size of one of the ar-tag's side in 										meter
		self.base_frame = "/base"
		self.topic_camera = "/cameras/" + arm + "_hand_camera/image"
		self.topic_alvar_camera = "/ar_pose_marker_" + arm
		self.topic_alvar_picker_camera = "/ar_pose_marker_" + self.robot.camera + \
			"_to_" + self.robot.picker
		self.topic_alvar_base = "/ar_pose_marker_" + arm + "_to_base"
		self.path_to_file = "../Stage_Baxter/ros_ws/src/" + \
			"pick_and_place/src/VSStereoAR/"
		self.image_acquired_path = self.path_to_file + "Acquisition/acquired_image.jpeg"
		self.ar_corners_path = self.path_to_file + "Acquisition/ar_corners.txt"
		self.convergence_path = self.path_to_file + "Convergences/convergence.txt"
		self.text_acquisition_done = "Here is the acquisition wich has been done"

		self.width        = 1280			# Camera width resolution
		self.height       = 800				# Camera height resolution
		self.focale = 406.136462382			# Camera focale

		self.resetCameras()
		self.openCamera(self.width, self.height)

	def manualShutdown(self):
		"""
			function to execute on manual shutdown of the program
			destroy all the remaining cv's windows
		"""
		cv2.destroyAllWindows()

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

		cam.exposure            = 5		# range, 0-100 auto = -1
		cam.gain                = -1		# range, 0-79 auto = -1
		cam.white_balance_blue  = -1		# range 0-4095, auto = -1
		cam.white_balance_green = -1		# range 0-4095, auto = -1
		cam.white_balance_red   = -1		# range 0-4095, auto = -1

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

	def getARPickerInfo(self):
		"""
			Get the list of ar-tag detected by ar track alvar
			in the picker's camera frame
		"""
		msg=rospy.wait_for_message(self.topic_alvar_picker_camera, AlvarMarkers)
		self.list_ar_to_picker = msg.markers

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
		while not self.navigator.button0 and not rospy.is_shutdown():
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
		cv2.circle(img, ar_crns[4], 2, (255, 255, 255), -1)

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
		cv2.waitKey(200)

	#########################################
	# 	frame system changes 		#
	#########################################

	def pixelToCamera(self, px):
		"""
			Convert image pixel to camera point

			:param px: The pixel we want to convert into point
			:return: The point computed from the pixel
		"""
		x = -((px[1] - (self.width / 2)) / self.focale)
		y = (px[0] - (self.height / 2)) / self.focale

		return (x, y)

	def cameraToPixel(self, pt):
		"""
			Convert camera point to image pixel

			:param pt: The point we want to convert into pixel
			:return: The pixel computed from the point

				todo:: understand why there is a correction needed
		"""
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
		x = int(round(-pt[1] * self.focale + self.width / 2)) + 10
		y = int(round(pt[0] * self.focale + self.height / 2)) + 15

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
		R00 = (a*a) + (b*b) - (c*c) - (d*d)
		R01 = (2*b*c) - (2*a*d)
		R02 = (2*a*c) + (2*b*d)
		R10 = (2*a*d) + (2*b*c)
		R11 = (a*a) - (b*b) + (c*c) - (d*d)
		R12 = (2*c*d) - (2*a*b)
		R20 = (2*b*d) - (2*a*c)
		R21 = (2*a*b) + (2*c*d)
		R22 = (a*a) - (b*b) - (c*c) + (d*d)

		Tcam_to_amer = np.matrix([[R00,R01,R02,ar_pos.x], \
					[R10,R11,R12,ar_pos.y], \
					[R20,R21,R22,ar_pos.z], \
					[0,  0,  0,  1]])
		
		ar_corners_cam = []
		for corner in ar_corners:
			temp = Tcam_to_amer*corner
			z = temp.item(2)
			x = temp.item(0) / z
			y = temp.item(1) / z
			ar_corners_cam.append([x, y, z])

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
		x = 0.0
		y = 0.0
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
		ar_corners = self.getARCorners()
		ar_corners_cam = self.arFrameToCam(list_ar, ar_corners)
		ar_corners_2d = self.camFrameTo2d(ar_corners_cam)

		return [ar_corners_cam, ar_corners_2d]

	def getARBasePos(self):
		"""
			Get the pose of the ar-tag in the Baxter's frame : base
			
			:return: The pose of the ar-tag in the Baxter's frame : base
		"""
		ar_pos = []
		while not ar_pos:
			self.robot.picker_arm.getARBaseInfo()
			self.robot.camera_arm.getARBaseInfo()
			if self.robot.picker_arm.list_ar_base:
				ar_pos = self.robot.picker_arm.list_ar_base[0].pose.pose.position
			elif self.robot.camera_arm.list_ar_base:
				ar_pos = self.robot.camera_arm.list_ar_base[0].pose.pose.position
			else:
				print("The ar-tag is not visible")

		return ar_pos

	def getARList(self):
		"""
			Get the list of the ar-tags detected

			:return: The list of the ar-tags detected
		"""
		ar_list = []
		self.robot.picker_arm.getARInfo()
		self.robot.picker_arm.getARPickerInfo()

		if self.robot.picker_arm.list_ar:
			ar_list = self.list_ar
		elif self.list_ar_to_picker:
			ar_list = self.list_ar_to_picker

		return ar_list

	def getOriFromPositions(self, base_pos, target_pos):
		"""
			Get the orientation to set for looking at a target giving the position from 					wich we want to look and the position of the target to look

			:param base_pos: The position from where we want to look
			:param target_pos: The position we want to look
			:return: The quaternion to apply to look from base's position to target's 					position
		"""
		vx = target_pos.x - base_pos.x
		vy = target_pos.y - base_pos.y
		vz = target_pos.z - base_pos.z
		v = [vx, vy, vz]

		norm_v = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
		n_v = [v[0]/norm_v, v[1]/norm_v, v[2]/norm_v]
		u = [1, 0, 0]
		norm_u = math.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
		n_u = [u[0]/norm_u, u[1]/norm_u, u[2]/norm_u]
		vu = np.cross(n_v, n_u)
		mat = np.matrix([[n_u[0],vu[0],n_v[0]],
				[n_u[1],vu[1],n_v[1]],
				[n_u[2],vu[2],n_v[2]]])

		w=math.sqrt(1+mat[0,0]+mat[1,1]+mat[2,2])/2
		x=(mat[2,1]-mat[1,2])/(4*w)
		y=(mat[0,2]-mat[2,0])/(4*w)
		z=(mat[1,0]-mat[0,1])/(4*w)

		q = Quaternion(x,y,z,w)

		return q

	def cameraAdjustment(self):
		"""
			Set the camera arm position and orientation considering the ar-tag's pose
		"""
		print("... Adjusting camera ...")
		ar_pos = self.getARBasePos()
		self.moveToStartingPoseCamera(ar_pos)

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

	def moveToStartingPosePicker(self):
		"""
			Move to the visual servoing starting pose for the picker arm
		"""
		curr_pose = self.limb.endpoint_pose()
		new_pos = Point()
		new_pos.x = curr_pose["position"].x
		new_pos.y = curr_pose["position"].y
		new_pos.z = curr_pose["position"].z - 0.1
		new_ori = curr_pose["orientation"]
		new_pose = self.settingPoseForIkService(self.base_frame, new_pos, new_ori)
		self.callingIkService(new_pose)

	def moveToStartingPoseCamera(self, ar_pos):
		"""
			Move to the visual servoing starting pose for the camera arm

			:param ar_pos: The position of the ar-tag
		"""
		curr_pose = self.limb.endpoint_pose()
		new_pos = Point()
		new_pos.x = curr_pose["position"].x - 0.12
		new_pos.y = curr_pose["position"].y + 0.1
		new_pos.z = ar_pos.z + 0.3
		new_ori = self.getOriFromPositions(new_pos, ar_pos)
		new_pose = self.settingPoseForIkService(self.base_frame, new_pos, new_ori)
		self.callingIkService(new_pose)
		self.getImage()
		cv2.imshow('img', self.img)

	def moveToArPos(self, arm_ori):
		"""
			Move to the ar-tag position with the orientation of the arm given

			:param arm_ori: The orientation of the arm we want to go
		"""
		print("... Moving to ar-tag pos ...")
		ar_pos = self.getARBasePos()
		ar_pos.z -= 0.07
		new_pose = self.settingPoseForIkService(self.base_frame, ar_pos, arm_ori)
		self.callingIkService(new_pose)

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
			v = q_dot.item(idx)
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

	def writeARInfo(self, ar_corners):
		"""
			Write the height of the ar-tag and the ar-tag's corners coordinates
			at the time of the acquisition in a file

			:param z_ar: The altitude of the ar-tag at the time of the acquisition
			:param ar_corners: The coordinates of the corners of the ar-tag at the time
				of the acquisition
		"""
		ar_corners_cam = ar_corners[0]
		ar_corners_2d = ar_corners[1]
		file = open(self.ar_corners_path, "w")
		for ar_corner_cam in ar_corners_cam:
			file.write(str(ar_corner_cam[0]) + "," + str(ar_corner_cam[1])\
				 + "," + str(ar_corner_cam[2]) + "\n")
		file.write("\n")
		for ar_corner_2d in ar_corners_2d:
			file.write(str(ar_corner_2d[0]) + "," + str(ar_corner_2d[1]) + "\n")
		file.close()

	def validatingAcquisition(self, img, ar_corners):
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
				self.writeARInfo(ar_corners)
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
				ar_corners = self.setARCorners(self.list_ar)
				ar_corners_2d = ar_corners[1]
				self.drawARCorners(self.img, ar_corners_2d)
				self.showImage(self.img, is_acquisition = True)
				self.validatingAcquisition(self.img, ar_corners)
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
			print("Please do the learning scenario at least once before" +
				"the pick and place scenario (acquisition ar informations not found)")
			sys.exit(0)

		curr_line = file.readline()
		corners_cam = []
		while curr_line != "\n":
			corner_cam = []
			corner_cam_pos = curr_line.split(",")
			for coord in corner_cam_pos:
				corner_cam.append(float(coord))
			corners_cam.append(tuple(corner_cam))
			curr_line = file.readline()

		curr_line = file.readline()
		corners_2d = []
		while curr_line != "":
			corner_2d = []
			corner_2d_pos = curr_line.split(",")
			for coord in corner_2d_pos:
				corner_2d.append(int(coord))
			corners_2d.append(tuple(corner_2d))
			curr_line = file.readline()
			
		return [corners_cam, corners_2d]

	def setDesiredCorners(self):
		"""
			Get the acquisition image and the ar-tag's information at the time of the
			acquisition then show it to the user

			:return: The coordinates of the ar-tag's corners at the time of the
				acquisition
		"""
		des_img = cv2.imread(self.image_acquired_path)
		if des_img.size == 0:
			print("Please do the learning scenario at least once before the pick" + 
				" and place scenario (acquisition image not found)")
			sys.exit(0)
		des_ar_corners = self.readARInfo()
		des_ar_corners_cam = des_ar_corners[0]
		des_ar_corners_2d = des_ar_corners[1]
		self.drawARCorners(des_img, des_ar_corners_2d)
		self.showImage(des_img, is_acquisition = True)
		
		return des_ar_corners

	def computeLx(self, pt_des_cam):
		"""
			Compute the camera matrix interaction for a given point

			:param pt_des: The coordinates of the desired point
			:param z: The altitude of the camera to the desired point
			:return: The camera matrix interaction for this point
		"""
		x = pt_des_cam[0]
		y = pt_des_cam[1]
		Z = pt_des_cam[2]
		Lx = [[-1.0/Z, 0.0, x/Z, x*y, -(1+(x*x)), y], \
		     [0.0, -1.0/Z, y/Z, 1+(y*y), -x*y, -x]]
		
		return Lx

	def computeL(self, des_ar_corners_cam):
		"""
			Compute the camera matrix interaction for the four desired points (corners) 					of the ar-tag

			:param des_ar_corners_cam: The list of the coordinates of the four desired 					point
			:return: The camera matrix interaction
		"""
		L1 = self.computeLx(des_ar_corners_cam[0])
		L2 = self.computeLx(des_ar_corners_cam[1])
		L3 = self.computeLx(des_ar_corners_cam[2])
		L4 = self.computeLx(des_ar_corners_cam[3])

		L = np.append(L1, L2, axis=0)
		L = np.append(L, L3, axis=0)
		L = np.append(L, L4, axis=0)

		return L

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

	def setCurrentCorners(self, des_ar_corners_2d):
		"""
			Get the current image of the camera and get and use the current ar-tag's 				informations then show it to the user

			:return: The current coordinates of the ar-tag's corners
		"""
		#self.getImage()
		ar_list = self.getARList()
		curr_ar_corners = []
		if ar_list:
			if not self.is_object_viewed:
				print("... Object detected ...")
				self.is_object_viewed = True
			curr_ar_corners = self.setARCorners(ar_list)
			curr_ar_corners_cam = curr_ar_corners[0]
			curr_ar_corners_2d = curr_ar_corners[1]
			#self.drawARCorners(self.img, des_ar_corners_2d)
			#self.drawARCorners(self.img, curr_ar_corners_2d)
			#self.showImage(self.img, is_acquisition = False)

		else:
			if self.is_object_viewed:
				print("... Object is not on visual ...")
				self.is_object_viewed = False

		return curr_ar_corners

	def getSFromCorners(self, ar_corners_cam):
		"""
			Construct the numpy matrix of s from the coordinates of the ar-tag's corners

			:param ar_corners_cam: The coordinates of the ar-tag's corners
			:return: the numpy matrix s (current value of the four points of interest)
		"""
		pt_1 = ar_corners_cam[0]
		pt_2 = ar_corners_cam[1]
		pt_3 = ar_corners_cam[2]
		pt_4 = ar_corners_cam[3]
		s = np.matrix([[pt_1[0]], [pt_1[1]], [pt_2[0]], [pt_2[1]], \
			[pt_3[0]], [pt_3[1]], [pt_4[0]], [pt_4[1]]])

		return s

	def getConvergence(self, des_ar_corners_cam, curr_ar_corners_cam, start_time):
		"""
			Get the euclidean distance between current points and desired points

			:param des_ar_corners_cam: The desired coordinates of the ar-tag's corners
			:param curr_ar_corners_cam: The current coordinates of the ar-tag's corners
			:param start_time: The time at wich we started storing this values
			:return: A vector with the time elapsed since beginning of the storing, and 					the euclidean distance in meter of current points from desired points
		"""
		des_pt1 = np.matrix(des_ar_corners_cam[0])
		curr_pt1 = np.matrix(curr_ar_corners_cam[0])
		c_pt1 = np.linalg.norm(curr_pt1 - des_pt1)
		des_pt2 = np.matrix(des_ar_corners_cam[1])
		curr_pt2 = np.matrix(curr_ar_corners_cam[1])
		c_pt2 = np.linalg.norm(curr_pt2 - des_pt2)
		des_pt3 = np.matrix(des_ar_corners_cam[2])
		curr_pt3 = np.matrix(curr_ar_corners_cam[2])
		c_pt3 = np.linalg.norm(curr_pt3 - des_pt3)
		des_pt4 = np.matrix(des_ar_corners_cam[3])
		curr_pt4 = np.matrix(curr_ar_corners_cam[3])
		c_pt4 = np.linalg.norm(curr_pt4 - des_pt4)
		time_elapsed = (rospy.Time.now()-start_time).to_sec()
		convergence = [time_elapsed, c_pt1, c_pt2, c_pt3, c_pt4]

		return convergence

	def isPoseReached(self, e):
		"""
			Verify if the arm of the robot has reached the pose desired

			:param e: The error of the joints
			:return: True if the pose is reached False otherwise
		"""
		pose_reached = abs(e.item(0)) < self.pose_tolerance and \
			abs(e.item(1)) < self.pose_tolerance and \
			abs(e.item(2)) < self.pose_tolerance and \
			abs(e.item(3)) < self.pose_tolerance and \
			abs(e.item(4)) < self.pose_tolerance and \
			abs(e.item(5)) < self.pose_tolerance and \
			abs(e.item(6)) < self.pose_tolerance and \
			abs(e.item(7)) < self.pose_tolerance

		return pose_reached

	def writeConvergence(self, convergence):
		"""
			Write the values of convergence in a file

			:param convergence: Values of convergences
		"""
		file = open(self.convergence_path, "w")
		for value in convergence:
			file.write(str(value[0]) + "\t" + str(value[1]) + "\t" + str(value[2]) \
				+ "\t" + str(value[3]) + "\t" + str(value[4]) + "\n")
		file.close()

	def computeQDot(self, des_ar_corners_cam, curr_ar_corners_cam, s_des, convergence, \
		start_time, L): 
		"""
			##################################################################
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Compute the error from the desired pose and the current pose of the
			ar-tag then compute the value of the command law (q dot for each joint)

			:param des_ar_corners: The desired coordinates of the ar-tag's corners
			:param curr_ar_corners: The current coordinates of the ar-tag's corners
			:param s_des: the situation desired
			:param convergence: The matrix containing all the convergence until now
			:param start_time: The rospy Time of the beginning of the visual servoing
			:return: The value of q dot for each joint
		"""
		s = self.getSFromCorners(curr_ar_corners_cam)

		e = s - s_des

		convergence.append(self.getConvergence(des_ar_corners_cam, curr_ar_corners_cam, start_time))

		if(self.isPoseReached(e)):
			q_dot = np.zeros(7)
			self.is_on_object = True

		else:
			original_frame = "/base"
			target_frame = "/" + self.arm + "_hand_camera_axis"
			time = rospy.Time(0)
			(trans,quat) = self.listener.lookupTransform(target_frame, \
				original_frame, time)
			a = quat[3]
			b = quat[0]
			c = quat[1]
			d = quat[2]
			R00 = (a*a) + (b*b) - (c*c) - (d*d)
			R01 = (2*b*c) - (2*a*d)
			R02 = (2*a*c) + (2*b*d)
			R10 = (2*a*d) + (2*b*c)
			R11 = (a*a) - (b*b) + (c*c) - (d*d)
			R12 = (2*c*d) - (2*a*b)
			R20 = (2*b*d) - (2*a*c)
			R21 = (2*a*b) + (2*c*d)
			R22 = (a*a) - (b*b) - (c*c) + (d*d)
			rot = np.matrix([[R00, R01, R02, 0, 0, 0], \
					[R10, R11, R12, 0, 0, 0], \
					[R20, R21, R22, 0, 0 ,0], \
					[0, 0, 0, R00, R01, R02], \
					[0, 0, 0, R10, R11, R12], \
					[0, 0, 0, R20, R21, R22]])
			J = rot*self.kin.jacobian()
			q_dot = np.linalg.pinv(L*J)*(-(self.gain*e))
	
			is_q_dot_ok = False
			while not is_q_dot_ok :
				min_coeff = abs(q_dot).min(axis = 0)
				max_coeff = abs(q_dot).max(axis = 0)
				if max_coeff > self.seuil_max:
					q_dot = q_dot*self.seuil_max/max_coeff
					is_q_dot_ok = True
					print("Celerity lowered")
				elif min_coeff < self.seuil_min:
					q_dot = q_dot*self.seuil_min/min_coeff
					print("Celerity increased")
				else:
					is_q_dot_ok = True

		return q_dot

	def visualServoing(self):
		"""
			Do the visual servoing of the arm to the ar-tag
		"""
		print("... Starting visual servoing ...")
		des_ar_corners = self.setDesiredCorners()
		des_ar_corners_cam = des_ar_corners[0]
		des_ar_corners_2d = des_ar_corners[1]
		s_des = self.getSFromCorners(des_ar_corners_cam)
		L = self.computeL(des_ar_corners_cam)
		self.gettingDropPose()
		self.robot.resettingRobot()
		self.moveToStartingPosePicker()
		self.robot.camera_arm.cameraAdjustment()
		cv2.destroyAllWindows()
		convergence = []
		start_time = rospy.Time.now()
		while (not self.is_on_object) and (not rospy.is_shutdown()):
			curr_ar_corners = self.setCurrentCorners(des_ar_corners_2d)
			if self.is_object_viewed:
				curr_ar_corners_cam = curr_ar_corners[0]
				q_dot = self.computeQDot(des_ar_corners_cam, curr_ar_corners_cam, \
					s_des, convergence, start_time, L)
				self.applyQDot(q_dot)
		self.writeConvergence(convergence)
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
