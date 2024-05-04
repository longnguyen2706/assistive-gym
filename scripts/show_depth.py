import time
import matplotlib
import argparse
import cv2


class DepthPrinter():
    def __init__(self, args):
        self.filepath = args.file
        self.depth_arr = cv2.imread(self.filepath)
        self.print_out()
        print("total: ", self.count_point_cloud())
        self.show()

    def print_out(self):
        print(self.depth_arr)
    
    def show(self):
        cv2.imshow("Depth", self.depth_arr)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
    
    def count_point_cloud(self):
        total = 0
        for mat in self.depth_arr:
            for row in mat:
                for val in row:
                    if val < 1: total += 1
        return total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    dp = DepthPrinter(args)
