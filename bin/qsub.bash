qsub -N config_brats_fold3_swin-unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold3_swin-unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold4_swin-unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold4_swin-unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold0_unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold0_unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold1_unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold1_unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold2_unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold2_unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold3_unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold3_unetr.yaml" train_jobscript.pbs
qsub -N config_brats_fold4_unetr -l nodes=1:ppn=8:gpus=2:cascadelake -l walltime=02:00:00:00 -l partition=gpu -l pmem=20gb -v config="configs/brats/config_brats_fold4_unetr.yaml" train_jobscript.pbs




