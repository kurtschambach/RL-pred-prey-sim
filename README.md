# Simple Predator-Prey Simulation

I wanted to get some experience with this kind of things (Reinforcement Learning), so here we are.

![](images/pred-prey-sim.png)

## Getting it to run

First, create a venv:
```bash
python3 -m venv venv
```

and activate it:
```bash
# bash
source ./venv/bin/activate

# fish
source ./venv/bin/activate.fish
```

Then install the dependencies:
```bash
python3 -m pip install -r ./requirements
```

Now just run the `main.py` file.

> [!NOTE]
> To speed the simulation and training process up, 
> or slow it down, use the arrow keys.
> 
> You can also stop the simulation using the space bar 
> for even faster training

Currently, you'll have to change the `RUN_TITLE` variable
every run, otherwise TensorBoard will only show the latest run.

## Monitoring

You can use the Matplotlib graph for some quick stats,
although TensorBoard is probably better, 
partly because the output in my case needed to be smoothed, 
which is easy with TensorBoard.

To see TensorBoard stats, run:
```bash
tensorboard --logdir=runs
```

## Updating deps

```bash
pip freeze > requirements.txt
```
