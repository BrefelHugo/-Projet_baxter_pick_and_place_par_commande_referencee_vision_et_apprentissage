#!/usr/bin/env python

import roslib
import sys
import rospy

from BaxterArm import BaxterArm

from baxter_interface import RobotEnable
import tuck_arms

from time import sleep

class BaxterRobot:
	
	def __init__(self, picker_arm):
		baxter = RobotEnable()
		baxter.enable()
		self.picker_arm = BaxterArm(picker_arm, self)

	def resettingRobot(self):
		"""
			Reset the position of the robot
		"""
            	raw_input("When everyone is out of robot's action field," +
			"please press enter to continue")
		tucker = tuck_arms.Tuck(False)
    		rospy.on_shutdown(tucker.clean_shutdown)
    		tucker.supervised_tuck()
		sleep(1)
		self.picker_arm.moveToStartingPose()
    		rospy.loginfo("Finished tuck")

	def monoLearning(self):
		"""
			Do the procedure to learn how to be positionned in respect to the ar-tag at
			the end of the visual servoing
		"""
		self.picker_arm.acquisition()

	def monoPickAndPlace(self):
		"""
			Do the procedure of visual servoing in relation to the ar-tag
			then do the procedure of pick and place of the object
		"""
		self.picker_arm.visualServoing()
		self.picker_arm.pickAndPlace()

def main(args):
	rospy.init_node("BaxterRobot", anonymous=True)
	print("unitary test BaxterRobot")

if __name__ == "__main__":
	main(sys.argv)
