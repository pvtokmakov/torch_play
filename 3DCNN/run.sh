# if [ ! -d "/local_sysdisk/USERTMP/ptokmako/flows89" ]; then
#   cd /local_sysdisk/USERTMP/ptokmako  
#   echo "copying file"
#   cp /scratch/clear/ptokmako/datasets/UCF101/frames/resized89/flows89.tar flows89.tar
#   echo "file copied"
#   mkdir flows89
#   tar -xf flows89.tar -C flows89
#   echo "file unpacked"
#   rm -f flows89.tar
# fi

# mkdir /local_sysdisk/USERTMP/ptokmako/ramflow89
# sudo ramdisk_create.sh /local_sysdisk/USERTMP/ptokmako/ramflow89 43g
# cp -a /scratch/clear/ptokmako/gvarol/. /local_sysdisk/USERTMP/ptokmako/ramflow89/
# cp -a /local_sysdisk/USERTMP/ptokmako/flows89/. /local_sysdisk/USERTMP/ptokmako/ramflow89/

cd /scratch/clear/ptokmako/torch/projects/3DCNN

th train.lua -gpu $(gpu_getIDs.sh)

# sudo ramdisk_delete.sh /local_sysdisk/USERTMP/ptokmako/ramflow89/
# 
# rm -rf /local_sysdisk/USERTMP/ptokmako/ramflow89/