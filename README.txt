Brain factory

The brain is only plain neurons with a nucleus position. Your project decides
which neurons are sensors, motors, etc.

1. python factory_brain.py                   makes neurons.csv  (id, x, y, z)
2. python connect_neurons.py                 makes connections.csv
3. python brain_roles.py neurons.csv sensor=12:front motor_left=6:left motor_right=6:right
                                             makes roles.csv (or use example_project.py)
4. python view_brain.py neurons.csv connections.csv roles.csv

Needs: Python 3 and numpy (pip install numpy). Viewer and roles need nothing extra.
More options: python factory_brain.py --help   (shapes, size, config file)
