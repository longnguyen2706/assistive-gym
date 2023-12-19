# This script used to copy necessary data (smpl files + urdf files) from a remote machine to the current machine.
# Run this script on newly installed assisstive-gym machine

REMOTE_IP='192.168.0.221'
# scp from another machine
echo "Copying data from $REMOTE_IP"
# Copy smpl
scp -r "louis@$REMOTE_IP:~/Documents/Projects/assistive-gym/examples/data" "../../examples/data/"

# Copy urdf (might need to rename absolute path to .obj file in urdf file)
 scp -r "louis@$REMOTE_IP:~/Documents/Projects/assistive-gym/experimental/urdf" "../../experimental/urdf/"