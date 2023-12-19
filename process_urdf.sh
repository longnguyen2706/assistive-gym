# URDF DIR 
URDF_DIR='/mnt/collectionssd/urdf'
URDF_OLDPATH='/nethome/nnagarathinam6/Documents/Projects/assistive-gym/experimental/urdf/'
# URDF_NEWPATH='/home/louis/Documents/Projects/urdf/'
URDF_NEWPATH="$URDF_DIR/"

# count 
count=0
# list all subdirectories in URDF_DIR
for dir in $(ls $URDF_DIR); do
    echo "Processing $dir"
    # if directory contains a urdf file
    if [ -f "$URDF_DIR/$dir/human.urdf" ]; then
        # replace old path with new path
        # echo "$URDF_DIR/$dir/human.urdf"
        # echo "$URDF_OLDPATH"
       sed -i "s|$URDF_OLDPATH|$URDF_NEWPATH|g" "$URDF_DIR/$dir/human.urdf"
       sed -i "s|$URDF_DIR/|$URDF_NEWPATH|g" "$URDF_DIR/$dir/human.urdf"
       sed -i "s|URDF_DIR|$URDF_NEWPATH|g" "$URDF_DIR/$dir/human.urdf"
       sed -i "s|\$\$/mnt/collectionssd/urdf///|$URDF_NEWPATH|g" "$URDF_DIR/$dir/human.urdf"
       sed -i "s|/mnt/collectionssd/urdff|/mnt/collectionssd/urdf/f|g" "$URDF_DIR/$dir/human.urdf"

    fi
    count=$((count+1))
done

echo "Processed $count directories"
