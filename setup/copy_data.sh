REMOTE_IP='192.168.0.221'
# scp from another machine
echo "Copying data from $REMOTE_IP"
# scp -r "louis@$REMOTE_IP:~/Documents/Projects/assistive-gym/examples/data" "../examples/data/"

scp -r "louis@$REMOTE_IP:~/Documents/Projects/assistive-gym/experimental/urdf" "../experimental/urdf/"