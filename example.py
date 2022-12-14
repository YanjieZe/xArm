import xArm


env = xArm.make_env(
		task_name="peginsert",
		seed=0,
		episode_length=100,
        image_size=84,
		observation_type="state+image", # state, image, state+image
		action_space="xyzw",
		domain_randomization=0,
	)

obs, state, info = env.reset()
returns = 0
for i in range(100):
	action = env.action_space.sample()
	obs, state, reward, done, info = env.step(action)
	returns += reward

print("returns:", returns)

example_image = env.render_obs(mode='rgb_array', height=224, width=224, camera_id="camera_static") # 1x224x224x3, uint8

import torchvision, torch
torchvision.utils.save_image(torch.from_numpy(example_image).float().div(255).permute(0, 3, 1, 2), "example.png")



print("save image in example.png")

