model_name=Crossformer
vmd_dim=6

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./Data/ \
  --data_path DYG_sgc_${vmd_dim}.csv \
  --model_id jn_sgc_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim} \
  --dec_in ${vmd_dim} \
  --c_out ${vmd_dim} \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --devices '0,1'\
  --target "jn_sgc"

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./Data/ \
  --data_path DYG_sgc_${vmd_dim}.csv \
  --model_id nd_sgc_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim} \
  --dec_in ${vmd_dim} \
  --c_out ${vmd_dim} \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --devices '0,1'\
  --target "nd_sgc"


  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./Data/ \
  --data_path DYG_sgc_${vmd_dim}.csv \
  --model_id ht_sgc_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim} \
  --dec_in ${vmd_dim} \
  --c_out ${vmd_dim} \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --devices '0,1'\
  --target "ht_sgc"

