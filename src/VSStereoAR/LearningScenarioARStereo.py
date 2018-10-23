#!/usr/bin/env python

import roslib
import sys
import rospy

from BaxterRobot import BaxterRobot

def main(args):
	rospy.init_node("LearningScenario", anonymous=True)
	picker = "left"
	camera = "right"
	baxterRobot = BaxterRobot(picker, camera)
	baxterRobot.stereoLearning()
	
if __name__ == "__main__":
	main(sys.argv)
