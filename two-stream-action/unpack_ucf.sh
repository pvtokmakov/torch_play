for d in /scratch2/clear/pweinzae/UCF101/original/videos/* ; do
   folder=`basename $d`
    mkdir /scratch/clear/ptokmako/datasets/UCF101/frames/$folder

    for v in /scratch2/clear/pweinzae/UCF101/original/videos/$folder/* ; do
        video=`basename $v`
        IFS='.'
        vid_name=$video
        mkdir /scratch/clear/ptokmako/datasets/UCF101/frames/$folder/${vid_name[0]}
        ffmpeg -i /scratch2/clear/pweinzae/UCF101/original/videos/$folder/$video.avi -vf fps=30 /scratch/clear/ptokmako/datasets/UCF101/frames/$folder/${vid_name[0]}/%05d.jpg
    done
done
