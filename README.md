# pong_vision_master


Environment to use:
- https://ale.farama.org/environments/pong/
  - Where we have RGB or RAM versions (vision or no vision) and also deterministic versions, and also even grayscale version of the environment
  - Maybe we should use V5 environment, but i don't know if in the paper they specify which version they used
- Also i don't know if maybe we should modify the reward of the environment, i don't know if in the paper they do it, but maybe a reward for when the player hits the ball could help, because if the player is never able to score against the other player it will be difficult for the agent to understand that what is doing is right, because it will always get negative reward. This way we can have denser rewards and have faster learning

About pong:
- https://www.findingtheta.com/blog/mastering-ataris-pong-with-reinforcement-learning-overcoming-sparse-rewards-and-optimizing-performance

In Ram:
- it is possible to get the positions of the ball (x,y) and y position only of paddles return {
            "ball_x": ram[49],
            "ball_y": ram[50],
            "player_y": ram[51],
            "player_x": ram[52],
            "opponent_x": ram[53],
            "opponent_y": ram[54],
        }
- But x position is not available, but it seems the positions are x=74 for opponent and x=187 for the player
- It seems this is maybe the correct information in PongNoFrameSkip-v4 (https://colab.research.google.com/drive/1Szy7ySmKxdEVMthZXIHjKdjDAaQGDPIP?usp=sharing#scrollTo=tRBgYbDiwuTd) {cpu_score = ram[13]      # computer/ai opponent score 
            player_score = ram[14]   # your score            
            cpu_paddle_y = ram[50]     # Y coordinate of computer paddle
            player_paddle_y = ram[51]  # Y coordinate of your paddle
            ball_x = ram[49]           # X coordinate of ball
            ball_y = ram[54]           # Y coordinate of ball


Size of pong environment:
- x: 210, y: 160, it seems (Original game resolution: say, 210×160 (Atari 2600 default))

`make_atari_env` function already preprocess the image to make it grayscale and to crop the images to only have the necessary information to play the game

- Active-gym
  - It seems the pong environment for fovea in active gym (maybe they normalize the colors to 0 to 1), but it says it uses float32 instead of uint8, so the problem is that the buffer get to use a lot of size because they are of 32 bit each and we have buffer size * frame_stack * observation_space * 4bytes. So a buffer_size of 500_000 uses 50GB>But this seems strange eto me did they use 80gb gpus? or is there something wrong here
  - The **fov loc** expects and has the values in order of y and x not x and y (because an image numpy array is first rows then columns)

Extra:
srun -t 400 -A cs-503 --qos=cs-503 --gres=gpu:1 --mem=32G --pty bash
