ifconfig wlan0 192.168.137.25
cd /home/sample-apps-for-robotics-platforms/RB5/linux_kernel_5_x/GStreamer-apps/cpp/gst_streaming || exit
make
./tcp_server 0 192.168.137.25 34808 
cd /home/yolov5