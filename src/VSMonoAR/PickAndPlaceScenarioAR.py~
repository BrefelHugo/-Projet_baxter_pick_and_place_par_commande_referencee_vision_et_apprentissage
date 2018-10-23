#!/usr/bin/env python

import roslib
import sys
import rospy

from BaxterRobot import BaxterRobot

def main(args):
	rospy.init_node("PickAndPlaceScenario", anonymous=True)
	baxterRobot = BaxterRobot("left")
	baxterRobot.monoPickAndPlace()
	
if __name__ == "__main__":
	main(sys.argv)
