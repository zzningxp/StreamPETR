export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

#name=stream_petr_r50_flash_704_bs2_seq_428q_nui_60e
#name=stream_petr_r50_flash_704_bs4_seq_428q_nui_60e
#name=stream_petr_r50_flash_704_bs2_seq_90e
#name=stream_petr_vov_flash_800_bs2_seq_24e
#projects/configs/StreamPETR/${name}.py \
name=sparse4d

tools/dist_train.sh \
projects/configs/Sparse4d/sparse4d.py \
4 \
--resume-from work_dirs/${name}/latest.pth
