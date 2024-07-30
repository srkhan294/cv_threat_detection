Demo Video: https://youtu.be/WyMSimOmK_U
## Documentation:-

### Objective: Develop a computer vision solution to monitor and track people at an airport for security and operational efficiency.

Tracking is process of identifying the positions of objects throughout multiple sequence of photos (i.e., video), tracking is getting the initial set of detections, assigning unique ids, and tracking them throughout frames of the video feed while maintaining the assigned ids. It's a 2 step process:-

1. Detection and localization of the object in the frame using some object detector like YOLOv8, CenterNet, etc.
2. Predicting the future motion of the object using its past information using a tracking algorithm.

The deep_sort folder in the repo has the original deep sort implementation, complete with the Kalman filter, hungarian algorithm and feature extractor. But the original repo is built only for validating the algorithm with the MARS test dataset. So, we have written a custom class Tracker.py for ourself utilizing the original repo.

Made Changes in generate_detections.py mudule to support latest version of tendsorflow.
1. "np.int" is changed to "int"

2.      self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

    changed to:

        self.session = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)


### Solution:
The solution uses yolov8n-pose for object detection and extraction of bone joint coordinates, it also uses deepSORT for tracking objects throughout the frames.
Once we have the object detected and tracked we try to figure out whether he's a threat or not by estimating his hand position. Assuming that if his hand is held high and extended 
then he might be holding an object and pointing it at some direction which maybe a gun or knife, etc.
Based on this assumption the code tries to figure out an extend arm position using coordinates from wrists, elbows and shoulder joints.
We can further implement a threshold of frames/time after which which may consider a person as threat instead of instantly considering him/her as threat 
as soon as the criteria is met, which is the case with this demo solution.
