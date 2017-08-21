# HOW TO RUN

To run the IBM PowerAI Distributed Deep Learning MNIST training example:

        $ source /opt/DL/ddl-tensorflow/bin/ddl-tensorflow-activate

        $ mpirun -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -n 2 python ddl_mnist.py
