# copy all urdf files from URDF_SRC to URDF_DST (skip the existing files)
# also, count the number of copied file

# URDF DIR
URDF_DST='/home/louis/Documents/hrl/synthetic_dataset/urdf'
URDF_SRC='/media/louis/Data/hrl/synthetic_dataset/urdf'

# count 
count=0
# list all subdirectories in URDF_DIR
for dir in $(ls $URDF_SRC); do
    # if directory not in URDF_DST
    if [ ! -d "$URDF_DST/$dir" ]; then
        echo "Copying $dir"
        cp -r -n "$URDF_SRC/$dir" "$URDF_DST/$dir"
    fi
    count=$((count+1))
done

echo "Copied $count directories"

echo "Number of urdf files in $URDF_DST: $(find $URDF_DST -mindepth 1 -maxdepth 1 -type d | wc -l)"