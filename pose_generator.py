# generate pose and shape parameters for the SMPL model
# 1. randomize betas and joint angles
# 2. check joint angle within limits
# 3. generate mesh
# 4. check self collisions
# 5. check stability: drop to ground and check if it is stable (not moving after 300 steps)
# 6. render it out

