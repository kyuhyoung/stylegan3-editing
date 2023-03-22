dir_img=notebooks/images_multi_face
dir_cropped=${dir_img}_croped
dir_aligned=${dir_img}_aligned
dir_transform=${dir_img}_landmark_transform
dir_edit=${dir_img}_edit
#dir_img=notebooks/images
#python prepare_data/preparing_faces_parallel.py --mode align --root_path ${dir_img} 
#python prepare_data/preparing_faces_parallel.py --mode crop --root_path ${dir_img} --random_shift 0.05
#python prepare_data/compute_landmarks_transforms.py --raw_root ${dir_img} --aligned_root ${dir_aligned} --cropped_root ${dir_cropped} --output_root ${dir_transform}
#python prepare_data/compute_landmarks_transforms.py --raw_root ${dir_img} --aligned_root ${dir_aligned} --cropped_root ${dir_img} --output_root ${dir_transform}
python prepare_data/compute_landmarks_transforms.py --raw_root ${dir_img} --aligned_root ${dir_aligned} --cropped_root ${dir_cropped} --output_root ${dir_transform}

#: << END
args_edit=(--output_path ${dir_edit}
        #--checkpoint_path experiments/restyle_e4e_ffhq_encode/checkpoints/best_model.pt 
        #--checkpoint_path pretrained_models/restyle_e4e_ffhq.pt 
        --checkpoint_path pretrained_models/restyle_pSp_ffhq.pt 
        #--data_path /path/to/test_data 
        #--data_path ${dir_cropped} 
        --data_path ${dir_aligned} 
        #--data_path ${dir_img} 
        --test_batch_size 4
        --test_workers 4 
        --n_iters_per_batch 3
        #--edit_directions "[age,pose,smile]"
        #--factor_ranges "[(-5_5),(-5_5),(-5_5)]" 
        --edit_directions "[age]"
        --factor_ranges "[(-3_3)]" 
        --landmarks_transforms_path ${dir_transform}/landmarks_transforms.npy
    )
   
#export LANG=en_GB.UTF-8 && python inversion/scripts/inference_editing.py "${args_edit[@]}"
python inversion/scripts/inference_editing.py "${args_edit[@]}"
#END
: << 'END'
python inversion/scripts/inference_editing.py \
--output_path /path/to/experiment/inference \ #--checkpoint_path experiments/restyle_e4e_ffhq_encode/checkpoints/best_model.pt \
--checkpoint_path pretrained/restyle_e4e_ffhq.pt \
--data_path /path/to/test_data \
--test_batch_size 4 \
--test_workers 4 \
--n_iters_per_batch 3 \
--edit_directions "[age,pose,smile]" \
--factor_ranges "[(-5_5),(-5_5),(-5_5)]" \
--landmarks_transforms_path /path/to/landmarks_transforms.npy
END
