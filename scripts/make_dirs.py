import os
import pickle as pk



# PULL: information
if __name__ == '__main__':
    main_folder = "data_gen_results/"
    if True:
        print("making folders in ", main_folder)
        i = 101
        folders = []
        while i <= 120:
            f = 'p' + str(i)
            folders.append(f)
            i+= 1
        
        for folder in folders:
            os.mkdir(main_folder + folder)
    else:
        # do nothing
        print("doing nothing")
