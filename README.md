# Revel

Revel is a tool for verified training in reinforcement learning. Revel offers
guaranteed safety at training time using an iteratively updated shield to bound
the behavior of the learner. See our
[NeurIPS paper](https://arxiv.org/abs/2009.12612) for more details.

## Requirements
Note: all of this is written using Python 3, and we will assume in these
instructions that `pip` and `python` refer to `pip3` and `python3`
respectively.

Revel relies on several python packages which can be installed from the
included requirements file:

    pip install -r requirements.txt

Additionally, Revel requires [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
and [Apron](http://apron.cri.ensmp.fr/library/). Please refer to their webpages
to install these. Note that `setup.py` will need to be modified by replacing
'path/to/eigen' with the path to the Eigen headers after you have installed
them. The relevant Apron headers and objects will need to be somewhere where
your C++ compiler can find them. You will need a C++17 compliant compiler.

Once all of the dependencies are installed you can build Revel:

    python setup.py install

At this point you should be ready to run the benchmarks.

## Running
The benchmarks used to evaluate Revel are available in the `benchmarks` folder.
To execute revel on a particular benchmark:

    python benchmarks/<benchmark>.py --retrain_nn --retrain_shield --nn_test --shield_test --max_episodes 1000 --episode_len 100 --safe_training --shields 5 --penalty_ratio 1.0

Running without the `safe_training`, `shields`, and `penalty_ratio` arguments
will result in a model learned using standard DDPG without safe training.
This command will generate a large log file containing the rewards and safety
violations at each episode as well as information about the performance and
synthesis of new shields. It will also contain a sample trajectory of the final
policy. We recommend dumping the output of this command into a log file to be
examined later.

For most benchmarks the above command is exactly what we used to generate
results. For `mountain-car.py`, you should instead use `episode_len 200`. For
`obstacle.py`, `mid_obstacle.py`, and `car-racing.py` you should use
`episode_len 200` and `max_episodes 2000`.

Additional hyperparameters may be changed in the `args` object within each
benchmark file. The current settings are the ones used in the reported results.

Our CPO experiments were run using the OpenAI implementation of CPO available
[here](https://github.com/openai/safety-starter-agents).

Note that Apron has been known to have memory errors at times. In our
experiments we use a fork of Apron where we have fixed a few issues, but we
don't provide a link to that fork in order to maintain anonymity. I believe
those issues have also been fixed in the main Apron repository now, but if the
code crashes with segfaults that may be why.
