python /root/dof_project/infer_single_ct_plane.py \
  --ct-path /root/dof_project/3dct_point_dataset_DS4/data/1/cta/1ct.nii.gz \
  --model-path /root/dof_project/model/vnet_3points_BS8_0.001lr_1rotloss_data_right/best_model.pth \
  --output-dir /root/dof_project/single_infer_case1_crop96 \
  --output-size 160 \
  --center-crop-size 96 \
  --device cpu


best model pth：
https://drive.google.com/file/d/1_KEBs8q-DYm6toutgbuny3hYh9ds4h3i/view?usp=drive_link
