#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def main():
    rospy.init_node('base_footprint_publisher')

    odom_frame = rospy.get_param('~odom_frame', 'odom')
    base_frame = rospy.get_param('~base_frame', 'base_link')
    footprint_frame = rospy.get_param('~footprint_frame', 'base_footprint')

    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(rospy.get_param('~rate', 50.0))
    last_stamp = None

    while not rospy.is_shutdown():
        try:
            tf_in = buf.lookup_transform(odom_frame, base_frame, rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        # Re-broadcasting the same stamp makes tf2 log TF_REPEATED_DATA at high volume.
        if last_stamp is not None and tf_in.header.stamp <= last_stamp:
            rate.sleep()
            continue
        last_stamp = tf_in.header.stamp

        q = tf_in.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        yaw_only = quaternion_from_euler(0.0, 0.0, yaw)

        tf_out = TransformStamped()
        tf_out.header.stamp = tf_in.header.stamp
        tf_out.header.frame_id = odom_frame
        tf_out.child_frame_id = footprint_frame
        tf_out.transform.translation.x = tf_in.transform.translation.x
        tf_out.transform.translation.y = tf_in.transform.translation.y
        tf_out.transform.translation.z = 0.0
        tf_out.transform.rotation.x = yaw_only[0]
        tf_out.transform.rotation.y = yaw_only[1]
        tf_out.transform.rotation.z = yaw_only[2]
        tf_out.transform.rotation.w = yaw_only[3]

        broadcaster.sendTransform(tf_out)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
