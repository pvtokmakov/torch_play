cd /scratch/clear/ptokmako/datasets/UCF101/frames/resized89
rm -f flows89.tar
tar -cf flows89.tar ../resized/dummy.txt

for d in /scratch/clear/ptokmako/datasets/UCF101/frames/resized89/* ; do
   folder=`basename $d`
   for v in /scratch/clear/ptokmako/datasets/UCF101/frames/resized89/$folder/* ; do
        video=`basename $v`
        echo $folder/$video/flow_jpg
        tar --append -f flows89.tar $folder/$video/flow_jpg
   done
done

