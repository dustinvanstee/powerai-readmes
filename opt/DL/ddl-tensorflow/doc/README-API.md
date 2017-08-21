# Running programs that use DDL

Programs using DDL are run with `mpirun`. They generally require an MPI
rank file, a choice of DDL initialization parameters, and the proper
command invocation:

   1. Create a rank file to map GPUs to each MPI client, refer to
      `/opt/DL/ddl/doc/README.md`

   2. Choose DDL initialization parameters, refer also to 
      `/opt/DL/ddl/doc/README.md`

   3. Execute the program with `mpirun`, for example:

           $ mpirun -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -n <N> -rf <rank_file_to_use> python <script_to_run.py>

      Here, `<N>` is the number of instances or learners and should match the
      dimension used in the `Init` function (described below).
      
      For example if the program calls:
      
              ddl.init(4, mode = '-mode n:4x3 -dump_iter 100')
      
      Then it might be run as:
      
              $ mpirun -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -n 12 -rf 4x3.rf python ddl_mnist.py


# DDL functions and semantics

## Init function

        .Input("num_gpu: int32") => number of gpus to use on each host machine
        .Output("rank: int32")   => MPI rank
        .Output("size: int32")   => MPI size
        .Output("gpuid: int32")  => assigned gpuid
        .Attr("mode: string")    => mode, refer to /opt/DL/ddl/doc/README.md
 
This must be called before any native TensorFlow operators. Typically,
we can execute this op on CPU using an additional session. The input is
the DDL configuration described in `/opt/DL/ddl/doc/README.md`. This
will inform the targeted network topology and learner mapping to it. The
output consists of MPI information (rank, size) and GPU assignment.

## Bcast function

        .Input("input: T")   => input tensor to broadcast, always rank0 broadcasts to others
        .Output("output: T") => output tensor with the broadcasted result
        .Attr("T: {float}")  => supported datatype
 
`Bcast` is used to synchronize all the trainable parameters (ie. weights
and biases) before the training. `Bcast` can be called after `Init` has
been called and completed on the assign GPU device. Each and every
trainable parameter must be broadcast to ensure good convergence.
   
## AllReduce function

        .Input("input: T")          => input tensor to AllReduce
        .Output("output: T")        => output tensor
        .Attr("op: {'sum', 'avg'}") => reduce operation
        .Attr("T: {float}")         => supported datatype
        .Attr("mpi: bool = false")  => use pure MPI_Allreduce to get a baseline
        .Attr("check: float = 0.0") => used to verify DDL AllReduce

`AllReduce` is used to synchronize the gradients of the trainable
parameters. It must be called after `Init` and `Bcast` in a typical
deep-learning network in TensorFlow. `AllReduce` is functionally
equivalent to `MPI_Allreduce` but allows DDL to exploit multi-tier
network bandwidth. The input is a tensor to `AddReduce` and the output is
the `AllReduce`d tensor.

The check attribute will:

   1. Run DDL `AllReduce`
   2. Run `MPI_Allreduce` with the same input
   3. Compare the results and flag error if the relative error it too large
      (`fabs(difference/mpi_value) > check value`) OR if "NaN" is detected.
      A dump file (`ddl.dumpXXX`) will be created if an error is
      detected.

The `check` value is tricky to set as error depends on the scale of the
values. We found 0.17 was sufficient enough to pass all the cases in
MINST with 8 learners. The reason for this error is obviously due to
different orders of additions.

`AllReduce` can be called on each trainable parameter or on concatenated
parameters (as in Caffe). If it is called on each, it is critical to
ensure the order among parameters, as also required in `MPI_Allreduce`.
If different learners `AllReduce` on different parameters, it will cause
a dead-lock situation. 
    
## AllReduceN function

        .Input("input: N*T")        => input tensors to AllReduce
        .Output("output: N*T")      => output tensors
        .Attr("op: {'sum', 'avg'}") => reduce operation
        .Attr("T: {float}")         => supported datatype
        .Attr("mpi: bool = false")  => use pure MPI_Allreduce to get a baseline
        .Attr("check: float = 0.0") => used to verify DDL AllReduce
      
This is a aggregated version of `AllReduce`. Essentially, this takes an
array of N tensors, performs `AllReduce` in a single shot, and return an
array of N reduced tensors. The benefit of using `AllReduceN` is better
performance and simpler integration.      
