# copy with ignore existing
sshpass -p "hrobolab2021" rsync -avz --ignore-existing hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'
sshpass -p "hrobolab2021" rsync -avz --ignore-existing hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'
sshpass -p "hello_2020" rsync -avz --ignore-existing louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'
sshpass -p "hello_2020" rsync -avz --ignore-existing louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'
cp -r -n ~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'

# report stats
# sshpass -p "hrobolab2021" rsync --dry-run --stats hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/' >> ./report.txt
# sshpass -p "hrobolab2021" rsync --dry-run --stats hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt
# sshpass -p "hello_2020" rsync --dry-run --stats louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt
# sshpass -p "hello_2020" rsync --dry-run --stats louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/* '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'  >> ./report.txt

echo "Number of result files: $(find '/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/' -mindepth 1 -maxdepth 1 -type d | wc -l)"


# RESULT_DIR='/home/louis/Documents/hrl/results/HumanComfort-v1_augmented/'
# mkdir -p $RESULT_DIR


# USERNAMES=('hrl_gpu_1' 'louis')
# PASSWORDS=('hrobolab2021' 'hello_2020')
# IP_ADDRESSES=('192.168.0.195' '192.168.0.169')
# PATH=('~/Documents/Projects/assistive-gym/trained_models' '~/Documents/Projects/assistive-gym/trained_models')
# FOLDERS=('HumanComfort-v1_augmented_dec10' 'HumanComfort-v1_augmented_dec11')

# # from hrl_gpu_1 - will override
# sshpass -p "hrobolab2021" scp -r hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/ /home/louis/Documents/hrl/results/HumanComfort-v1_augmented/
# sshpass -p "hrobolab2021" scp -r hrl_gpu_1@192.168.0.195:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/ /home/louis/Documents/hrl/results/HumanComfort-v1_augmented/

# from hrl_hela - will override
# sshpass -p "hello_2020" scp -r louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec10/ /home/louis/Documents/hrl/results/HumanComfort-v1_augmented/
# sshpass -p "hello_2020" scp -r louis@192.168.0.169:~/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_augmented_dec11/ /home/louis/Documents/hrl/results/HumanComfort-v1_augmented/


# for i in {0..1}
# do  
#     for j in {0..1} 
#     do
#         echo "${USERNAMES[$i]}", "${PASSWORDS[$i]}", "${IP_ADDRESSES[$i]}", "${PATH[$i]}", "${FOLDERS[$j]}"
#         sshpass -p ${PASSWORDS[$i]} rsync -avz --ignore-existing ${USERNAMES[$i]}@${IP_ADDRESSES[$i]}:${PATH}/${FOLDERS[$j]}/* $RESULT_DIR
#     done
# done


# for i in {0..1}
# do 
#     echo "${USERNAMES[$i]}", "${PASSWORDS[$i]}", "${USERNAMES[$i]}@${IP_ADDRESSES[$i]}:${PATH}/${FOLDERS[$i]}/"
#     sshpass -p ${PASSWORDS[$i]} rsync --progress --dry-run --stats ${USERNAMES[$i]}@${IP_ADDRESSES[$i]}:${PATH}/${FOLDERS[$i]}/* $RESULT_DIR
# done


