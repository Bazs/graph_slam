## GraphSLAM educational implementation

A bare-bones GraphSLAM implementation based on Chapter 11 of [1]. Mostly for self-educational purposes. Currently has two entry points: 
* ```graph_slam.py``` can be executed as a sort of single-shot application. Minimal dependencies.
* ```visualizer/visualizer.py``` is a Qt5 + PyQt5 based interactive application, offers a richer experience.

### Implemented functionality

The application can:
* Generate a simple world with point-like landmarks.
* Generate a random ego-path using a constant turn rate and velocity motion model.
* Generate landmark observations for each ego state along the path using a simple stochastic sensor model.
* Perform the algorithmic steps of GraphSLAM:
  * Initialize
  * Linearize
  * Reduce
  * Solve

[1]: Probabilistic Robotics, Thrun, S. and Burgard, W. and Fox, D. and Arkin, R.C. 2005 MIT Press