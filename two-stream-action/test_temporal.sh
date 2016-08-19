if [ ! -d "/local_sysdisk/USERTMP/ptokmako/flows" ]; then
  cd /local_sysdisk/USERTMP/ptokmako  
  echo "copying file"
  cp /scratch/clear/ptokmako/datasets/UCF101/frames/resized/flows.tar flows.tar
  echo "file copied"
  mkdir flows
  tar -xf flows.tar -C flows
  echo "file unpacked"
  rm -f flows.tar
fi

mkdir /local_sysdisk/USERTMP/ptokmako/ramflow
sudo ramdisk_create.sh /local_sysdisk/USERTMP/ptokmako/ramflow 15g
echo "copying flow to memory"
cp -a /local_sysdisk/USERTMP/ptokmako/flows/. /local_sysdisk/USERTMP/ptokmako/ramflow/

cd /scratch/clear/ptokmako/torch/projects/two-stream-action

th test_ucf_temporal.lua -gpu $(gpu_getIDs.sh)

sudo ramdisk_delete.sh /local_sysdisk/USERTMP/ptokmako/ramflow/

rm -rf /local_sysdisk/USERTMP/ptokmako/ramflow/