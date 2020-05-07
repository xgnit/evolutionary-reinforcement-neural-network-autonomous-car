
# An self learning autonomous vehicle based on evolutionary neural network and reinforcement learning neural network

This program is mean to demonstrate how a self learning autonomous vehicle works, with eith evolutionary algorithm or reinforcement learning. That means you can switch between these two methods.


Start the program with evolutionary methode by running: python Game.py

If you want to starts the program with reinforcement learning, run it with:
python Game.py --rl True

You can try to change the game by editing the Config.py file, e.g. change the map size, change the car size etc.

Following is the example result of a evolutionary algorithm test:

This is the first generation

![alt text](./showcase/1.gif?raw=true "First Generation")

This is the last generation which has a individual fits the max required fitness

![alt text](./showcase/9.gif?raw=true "Last Generation")


Furthermore following are the result of reinfocement learning:

This is one of the tests in the beginning:

![alt text](./showcase/11.gif?raw=true "Bad bot :(")

This is one test that passed the max required fitness

![alt text](./showcase/12.gif?raw=true "Good bot :)")

Have fun!
