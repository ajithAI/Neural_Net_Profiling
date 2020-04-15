## https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

import tensorflow as tf
import tensorflow.python.framework.ops as ops 
g = tf.Graph()
with g.as_default():
  A = tf.Variable(tf.random_normal( [25,16] ))
  B = tf.Variable(tf.random_normal( [16,9] ))
  C = tf.matmul(A,B) # shape=[25,9]
for op in g.get_operations():
  flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
  if flops is not None:
    print 'Flops should be ~',2*25*16*9
    print '25 x 25 x 9 would be',2*25*25*9 # ignores internal dim, repeats first
    print 'TF stats gives',flops
    
g = tf.Graph()
run_meta = tf.RunMetadata()
with g.as_default():
    A = tf.Variable(tf.random_normal( [25,16] ))
    B = tf.Variable(tf.random_normal( [16,9] ))
    C = tf.matmul(A,B) # shape=[25,9]

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    if flops is not None:
        print('Flops should be ~',2*25*16*9)
        print('25 x 25 x 9 would be',2*25*25*9) # ignores internal dim, repeats first
        print('TF stats gives',flops.total_float_ops)
