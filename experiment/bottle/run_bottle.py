import numpy as np
import torch
import gym
import json
import argparse
import cv2
import os
import time
import math
from tensorboardX import SummaryWriter

import nimblephysics as nimble

#robot_init_state = [ 0.07129488 ,-0.82724565, -2.68480527, -1.54026022 , 0.20306757,  1.29076029,
# -0.25689644,  0.0085 ,     0.0085    ]

robot_init_state = [-1.8848e+00, -3.5420e+00, -2.7650e+00, -2.0247e+00, -3.3566e+00,
         7.5730e-01, -3.3264e-02,  0.02, 0.02]

def create_world(time_step=0.001):
	# Create and configure world.
	world: nimble.simulation.World = nimble.simulation.World()
	world.setGravity([0, 0, -0.0])
	world.setTimeStep(time_step)

	return world

def model_base_assemble(desired_goal, bottle_init_pos, num_timesteps=100):
	# Set up Nimble
	flag = False
	num_timesteps_pick = num_timesteps
	num_timesteps_place = num_timesteps * 2

	world = create_world()

	table = world.loadSkeleton("./urdf/table_real/table.urdf", np.array([0, 0, -0.002]), np.array([0, 0, 0]))
	table.getBodyNodes()[0].setCollidable(False)
	
	franka = world.loadSkeleton("./urdf/franka_panda/panda.urdf",
								np.array([-0.39, 0.075, 0.0]),  # np.array([-0.6, 0.075, 0.0]),
								np.array([0, 0, math.pi / 2]))
	franka.setPositions(robot_init_state)
	for i in range(franka.getNumBodyNodes()):
		franka_node = franka.getBodyNode(i)
		franka_node.setFrictionCoeff(1000000.)
		franka_node.setGravityMode(False)
		franka_node.setCollidable(False)
	franka.getBodyNodes("panda_leftfinger")[0].setCollidable(True)
	franka.getBodyNodes("panda_rightfinger")[0].setCollidable(True)
	#franka.getBodyNodes("cap_link")[0].setCollidable(True)



	cap = world.loadSkeleton("./urdf/bottle_cap/cap.urdf", np.array([0.08, -.32, -0.13]), np.array([0, 0, 0]))
	cap.getBodyNodes()[0].setCollidable(True)
	for i in range(cap.getNumBodyNodes()):
		cap_node = cap.getBodyNode(i)
		cap_node.setFrictionCoeff(1000000.)
	# 前三个是角度，后三个是位置(y-,z+,x-)
	bottle = world.loadSkeleton("./urdf/bottle_cap/mobility.urdf", bottle_init_pos, np.array([0, 0, math.pi / 2])) 
	for i in range(bottle.getNumBodyNodes()):
		bottle_node = bottle.getBodyNode(i)
		bottle_node.setFrictionCoeff(1.)
	bottle.getBodyNodes()[0].setCollidable(False)

	


	ikMap = nimble.neural.IKMapping(world)
	hand_link = franka.getBodyNode("panda_hand")
	ikMap.addSpatialBodyNode(hand_link)

	actions_robot_pick = torch.tensor(np.load("actions_robot_pick.npy"))
	actions_robot_place = torch.tensor(np.load("actions_robot_place.npy"))
	actions_robot_place2 = torch.tensor(np.load("actions_robot_put.npy"))
	actions_robot_pick.requires_grad = True
	actions_robot_place.requires_grad = True
	actions_robot_place2.requires_grad = True

	#actions_robot_pick = torch.zeros((int(num_timesteps/10), 7), requires_grad=True)
	optimizer1 = torch.optim.Adam([{'params': actions_robot_pick}], lr=0.1)

	# actions_robot_place = torch.zeros((int(num_timesteps_place/10), 7), requires_grad=True)
	optimizer2 = torch.optim.Adam([{'params': actions_robot_place}], lr=0.5)

	#actions_robot_place2 = torch.zeros((int(num_timesteps_place/10), 7), requires_grad=True)
	optimizer3 = torch.optim.Adam([{'params': actions_robot_place2}], lr=0.05)

	init_state = torch.Tensor(world.getState())
	print("Initial state:", init_state)
	
	init_hand_pos = nimble.map_to_pos(world, ikMap, init_state)
	print("Initial EE:", init_hand_pos)
	pick_cap_pos = torch.Tensor([0.1, -.3, 0.11])
	place_pos = torch.Tensor([0.19, -0.09, 0.43])
	print("Target Position:", pick_cap_pos)

	# if args.vis:
	# 	gui = nimble.NimbleGUI(world)
	# 	gui.serve(8080)
	# 	gui.nativeAPI().renderWorld(world)

	print(world.checkPenetration(True))
	input()
	states_total = []

	#? pick up the cap
	for k in range(100000):
		state = init_state
		states = [state.detach()]
		hand_poses = [nimble.map_to_pos(world, ikMap, state).detach()]
		for i in range(num_timesteps_pick):
			action = torch.concat([actions_robot_pick[int(i/10)], torch.zeros(8,)], -1)
			state = nimble.timestep(world, state, action)
			state[7:9] = 0.02
			#print(state)
			states.append(state.detach())
			hand_poses.append(nimble.map_to_pos(world, ikMap, state).detach())
		# if args.vis:
		# 	gui.loopStates(states)
		hand_pos = nimble.map_to_pos(world, ikMap, state)
		# print(hand_pos)
		#? hand_pos = [angle, position]
		loss_grasp = ((hand_pos[3:] - pick_cap_pos)**2).mean()  + ((torch.abs(hand_pos[:3]) - torch.Tensor([np.pi, 0, 0]))**2).mean()
		loss = loss_grasp
		print("move", k, loss.detach().numpy())
		optimizer1.zero_grad()
		loss.backward()
		if np.isnan(actions_robot_pick.grad.numpy().sum()):
			break
		optimizer1.step()
		# input()
		if loss < 1e-4:
			print(state)
			for _ in range(num_timesteps_pick):
				states_total.append(states[_])
			#np.save("actions_robot_pick.npy", actions_robot_pick.detach().numpy())
			flag = True
			break
	print("finish pick up")
	#? clip the cap
	state[15:] = 0.0
	for i in range(100):
		#action = torch.concat([torch.zeros(7,), torch.tensor([-0.13, -0.13])], -1)
		action = torch.concat([torch.zeros(7,), torch.tensor([-0.5, -0.5]), torch.zeros(3,), torch.tensor([-0., -0., 0])], -1)
		# print(action)
		state = nimble.timestep(world, state, action)
		states_total.append(state.detach())
		states.append(state.detach())
	state[15:] = 0.0
	finger_pose = state[7:9].detach()
	print("finish clip")
	print(hand_pos)
	print(state[12:15])
	# if args.vis:
	# 	gui.loopStates(states_total)

	clip_state = state.detach()
	finger_pose = state[7:9].detach()
	cap.getBodyNodes()[0].setCollidable(False)
	#? place the cap
	for k in range(100000):
		state = clip_state
		state[15:] = 0.0
		states = [state.detach()]
		hand_poses = [nimble.map_to_pos(world, ikMap, state).detach()]
		for i in range(num_timesteps_place):
			action = torch.concat([actions_robot_place[int(i/10)], torch.tensor([-5, -5]), torch.zeros(6,)], -1)
			state = nimble.timestep(world, state, action)
			hand_poses.append(nimble.map_to_pos(world, ikMap, state).detach())
			hand_pos = nimble.map_to_pos(world, ikMap, state)
			state[7:9] = finger_pose
			state[12] = -hand_pos[4] - 0.315
			state[13] =  hand_pos[5] - 0.113
			state[14] = -hand_pos[3] + 0.0909
	

			states.append(state.detach())
		# if args.vis:
		# 	gui.loopStates(states)
		
		# print(hand_pos)
		#? hand_pos = [angle, position]
		# loss_grasp = ((hand_pos[3:] - (pick_cap_pos+torch.tensor([0,0,0.5])))**2).mean()  + ((torch.abs(hand_pos[:3]) - torch.Tensor([np.pi, 0, 0]))**2).mean()
		loss_grasp = ((hand_pos[3:] - (place_pos+torch.tensor([0,0,0.])))**2).mean()  + ((torch.abs(hand_pos[:3]) - torch.Tensor([np.pi, 0, 0]))**2).mean()
		loss_vel = ((state[15:])**2).mean() * 0.0
		loss_finger = ((state[7:9] - finger_pose)**2).mean() * 0.
		loss = loss_grasp + loss_vel + loss_finger
		print("move", k, "loss_grasp", loss_grasp.detach().numpy(), "loss_vel", loss_vel.detach().numpy(), "loss", loss.detach().numpy())
		optimizer2.zero_grad()
		loss.backward()
		if np.isnan(actions_robot_place.grad.numpy().sum()):
			break
		optimizer2.step()
		# input()
		#np.save("actions_robot_place_tmp.npy", actions_robot_place.detach().numpy())
		if loss < 1e-4:
			print(state)
			for _ in range(num_timesteps_place):
				states_total.append(states[_])
			# np.save("actions_robot_place.npy", actions_robot_place.detach().numpy())
			flag = True
			break
	print("finish place")

	clip_state = state.detach()
	finger_pose = state[7:9].detach()
	cap.getBodyNodes()[0].setCollidable(False)
	for k in range(100000):
		state = clip_state
		state[15:] = 0.0
		states = [state.detach()]
		hand_poses = [nimble.map_to_pos(world, ikMap, state).detach()]
		for i in range(num_timesteps_place):
			action = torch.concat([actions_robot_place2[int(i/10)], torch.tensor([-5, -5]), torch.zeros(6,)], -1)
			state = nimble.timestep(world, state, action)
			hand_poses.append(nimble.map_to_pos(world, ikMap, state).detach())
			hand_pos = nimble.map_to_pos(world, ikMap, state)
			state[7:9] = finger_pose
			state[12] = -hand_pos[4] - 0.315
			state[13] =  hand_pos[5] - 0.113
			state[14] = -hand_pos[3] + 0.0909
	

			states.append(state.detach())
		# if args.vis:
		# 	gui.loopStates(states)
		
		# print(hand_pos)
		#? hand_pos = [angle, position]
		# loss_grasp = ((hand_pos[3:] - (pick_cap_pos+torch.tensor([0,0,0.5])))**2).mean()  + ((torch.abs(hand_pos[:3]) - torch.Tensor([np.pi, 0, 0]))**2).mean()
		loss_grasp = ((hand_pos[3:] - (place_pos+torch.tensor([0.02,0.01,-0.11])))**2).mean()  + ((torch.abs(hand_pos[:3]) - torch.Tensor([np.pi, 0, 0]))**2).mean()
		loss_vel = ((state[15:])**2).mean() * 0.0
		loss_finger = ((state[7:9] - finger_pose)**2).mean() * 0.
		loss = loss_grasp + loss_vel + loss_finger
		print("move", k, "loss_grasp", loss_grasp.detach().numpy(), "loss_vel", loss_vel.detach().numpy(), "loss", loss.detach().numpy())
		optimizer2.zero_grad()
		loss.backward()
		if np.isnan(actions_robot_place2.grad.numpy().sum()):
			break
		optimizer3.step()
		if loss < 1e-1:
			print(state)
			for _ in range(num_timesteps_place):
				states_total.append(states[_])
			#np.save("actions_robot_put.npy", actions_robot_place2.detach().numpy())
			flag = True
			break
	print("finish put")

	state[15:] = 0.0
	for i in range(100):
		#action = torch.concat([torch.zeros(7,), torch.tensor([-0.13, -0.13])], -1)
		action = torch.concat([torch.zeros(7,), torch.tensor([0.5, 0.5]), torch.zeros(3,), torch.tensor([-0., -0., 0])], -1)
		# print(action)
		state = nimble.timestep(world, state, action)
		states_total.append(state.detach())
		states.append(state.detach())

	if args.vis:
		gui = nimble.NimbleGUI(world, useBullet=True, videoLogFile='path/to.mp4')
		gui.loopStates(states_total)
		gui.stopServing()

	return flag


if __name__ == "__main__":
	start_time = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="PandabottleJointsDense-v2")  # OpenAI gym environment name
	parser.add_argument("--exp_name", default="exp")
	parser.add_argument("--discount", default=0.9)
	parser.add_argument("--lr", default=0.5)
	parser.add_argument("--seed", default=970101, type=int)
	parser.add_argument("--num_timesteps", default=30, type=int)
	parser.add_argument("--data_dir", default="./data/demo")
	parser.add_argument("--rand_cam", action="store_true")
	parser.add_argument("--camera_mode", default="8dofreal")
	parser.add_argument("--vis", action="store_true")
	args = parser.parse_args()

	num_timesteps = args.num_timesteps
	

	
	bottle_init_pos = np.array((0.189, -0.101, 0.13))
	# bottle_init_pos = np.array((-0.1, -0.1, 0.07))
	desired_goal = np.array((-0.1, -0.1, 0.1))
	# desired_goal = np.array((-0.1, -0.1, 0.052))

	success = model_base_assemble(desired_goal, bottle_init_pos, num_timesteps)
	if success:
		print('success')
