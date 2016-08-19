cd /scratch/clear/ptokmako/src/FastVideoSegment

for d in /scratch/clear/ptokmako/datasets/UCF101/frames/* ; do
   folder=`basename $d`

    for v in /scratch/clear/ptokmako/datasets/UCF101/frames/$folder/* ; do
        vid_name=`basename $v`        
        /softs/stow/matlab-2015b/bin/matlab -nojvm -nodisplay -nosplash -r "computeUCFFlow('${folder}', '${vid_name}'); quit";        
    done
done
