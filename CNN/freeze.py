import tensorflow as tf
import sys
import cv2
import numpy as np
import tensorflow as tf
from BGNet import BGNet
class Freeze:
    def freeze_graph(self,model_dir, output_node_names):
        if not tf.gfile.Exists(model_dir):
            raise AssertionError("Export directory doesn't exists. Please specify an export directory: %s" % model_dir)
        if not output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/bgnet_model.pb"
        clear_devices = True
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
            saver.restore(sess, input_checkpoint)
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess,tf.get_default_graph().as_graph_def(),output_node_names.split(",")) 
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
        return output_graph_def
    
    def test_frozen_graph(self,frozen_graph_filename,test_image,input_op,output_op):
        bgnet=BGNet()
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        x = graph.get_tensor_by_name(input_op)
        y = graph.get_tensor_by_name(output_op)
        probabilities=tf.nn.softmax(y,name="softmax")
        with tf.Session(graph=graph) as sess:
            test=bgnet.load_images(test_image_path)
            image=np.array(test).reshape(bgnet.input)
            y_out = sess.run(probabilities, feed_dict={x:image})
            print y_out
					
if __name__ == '__main__':    
    mode=(sys.argv[1]).upper()
    if mode=='FREEZE':
        freeze=Freeze()
        model='./model/'
        output_node_names="output_layer/BiasAdd"
        freeze.freeze_graph(model,output_node_names)
    elif mode=='TEST':
        freeze=Freeze()
        test_image_path="/home/balaji/Downloads/balaji/tensorflow_code/custom_network/code/test_batch/19.jpg"
        model_dir="./model/bgnet_model.pb"
        input_op='prefix/input:0'
        output_op='prefix/output_layer/BiasAdd:0'
        freeze.test_frozen_graph(model_dir,test_image_path,input_op,output_op)
    else:
        print "Only FREEZE and TEST options available"
