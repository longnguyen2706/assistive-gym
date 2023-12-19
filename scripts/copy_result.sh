# Copy CMAES results from another machine to this machine
# since we are using sshpass to ssh to the machine with password, we could not pass value as argument to the script.
# So, we need to update the script everytime we want to copy from different folder

# copy cane - original data
# sshpass -p "hello_2020" rsync -avz --ignore-existing louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_rerun_dec15/* '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/'
# sshpass -p "hello_2020" rsync -avz --ignore-existing louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_rerun_dec15/* '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/'
# cp -r -n ~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_rerun_dec15/* '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/'
# echo "Number of result files: $(find '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/' -mindepth 1 -maxdepth 2 -type d | wc -l)"

# copy cup - original data
sshpass -p "hrobolab2021" rsync -avz --ignore-existing hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_rerun_dec15/* '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cup/'
echo "Number of result files: $(find '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/' -mindepth 1 -maxdepth 2 -type d | wc -l)"

# report stats - run this to report stats to file
# sshpass -p "hrobolab2021" rsync --dry-run --stats hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/' >> ./report.txt
# sshpass -p "hrobolab2021" rsync --dry-run --stats hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt
# sshpass -p "hello_2020" rsync --dry-run --stats louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt
# sshpass -p "hello_2020" rsync --dry-run --stats louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt
# echo "Number of result files: $(find '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cane/' -mindepth 1 -maxdepth 2 -type d | wc -l)"