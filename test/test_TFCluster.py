import unittest
import test
from tensorflowonspark import TFCluster, TFNode
import logging


class TFClusterTest(test.SparkTest):
  @classmethod
  def setUpClass(cls):
    super(TFClusterTest, cls).setUpClass()

  @classmethod
  def tearDownClass(cls):
    super(TFClusterTest, cls).tearDownClass()

  def test_basic_tf(self):
    """Single-node TF graph (w/ args) running independently on multiple executors."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      x = tf.constant(args['x'])
      y = tf.constant(args['y'])
      result = tf.add(x, y)
      assert result.numpy() == 3

    args = {'x': 1, 'y': 2}
    cluster = TFCluster.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers, num_ps=0)
    cluster.shutdown()

  def test_inputmode_spark(self):
    """Distributed TF cluster w/ InputMode.SPARK"""
    def _map_fun(args, ctx):
      import tensorflow as tf
      cluster, server = TFNode.start_cluster_server(ctx)
      if ctx.job_name == "ps":
        server.join()
      elif ctx.job_name == "worker":

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
          tf_feed = TFNode.DataFeed(ctx.mgr, False)
          
          while not tf_feed.should_stop():
            outputs = tf.square(tf_feed.next_batch(10))
            logging.info("===== output {}".format(outputs))
            tf_feed.batch_results(outputs)

          logging.info(" === end of map fn")

    input = [[x] for x in range(1000)]    # set up input as tensors of shape [1] to match placeholder
    rdd = self.sc.parallelize(input, 10)
    cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.SPARK)
    rdd_out = cluster.inference(rdd)
    rdd_sum = rdd_out.sum()
    self.assertEqual(rdd_sum.numpy(), sum([x * x for x in range(1000)]))
    cluster.shutdown()


if __name__ == '__main__':
  unittest.main()
