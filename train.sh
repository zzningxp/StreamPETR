name=stream_petr_r50_flash_704_bs2_seq_428q_nui_60e

tools/dist_train.sh \
projects/configs/StreamPETR/${name}.py \
4 \
--work-dir \
work_dirs/${name}/
