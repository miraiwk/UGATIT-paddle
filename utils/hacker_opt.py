from paddle import fluid
from paddle.fluid import framework
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

@framework.dygraph_only
def set_lr(self, value):
    """
    :api_attr: imperative
    
    Set the value of the learning rate manually in the optimizer. If the optimizer use LearningRateDecay,
    this API cannot be invoked, because it will lead to conflict.

    Args:
        value (float|Variable): the value of learning rate

    Returns:
        None
      
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
                    
            with fluid.dygraph.guard():
                linear = fluid.dygraph.nn.Linear(10, 10)

                adam = fluid.optimizer.Adam(0.1, parameter_list=linear.parameters())

                # set learning rate manually by python float value
                lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                for i in range(5):
                    adam.set_lr(lr_list[i])
                    lr = adam.current_step_lr()
                    print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.2
                #    current lr is 0.3
                #    current lr is 0.4
                #    current lr is 0.5
                #    current lr is 0.6


                # set learning rate manually by framework Variable
                lr_var = fluid.layers.create_global_var(
                    shape=[1], value=0.7, dtype='float32')
                adam.set_lr(lr_var)
                lr = adam.current_step_lr()
                print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.7



    """
    if not isinstance(value, (framework.Variable, float)):
        raise TypeError(
            "The type of 'value' in optimizer.set_lr must be (float, Variable), but received %s."
            % (type(value)))
    if isinstance(self._learning_rate, LearningRateDecay):
        raise RuntimeError(
            "optimizer's learning rate can't be LearningRateDecay when invoke this API, because this will lead to conflict."
        )
    if isinstance(value, float):
        self._learning_rate = value
        current_lr = self._global_learning_rate()
        if current_lr is not None:
            global_block = framework.default_main_program().global_block()
            global_block.append_op(
                type='fill_constant',
                outputs={'Out': [current_lr]},
                attrs={
                    'dtype': current_lr.dtype,
                    'shape': list(current_lr.shape),
                    'value': float(value)
                },
                stop_gradient=True)
    else:
        assert len(value.shape) == 1 and value.shape[
            0] == 1, "optimizer's learning rate must be 1-D Tensor with shape[1]"
        self._learning_rate_map[framework.default_main_program()] = value


if not hasattr(fluid.optimizer.Optimizer, 'set_lr'):
    fluid.optimizer.Optimizer.set_lr = set_lr
