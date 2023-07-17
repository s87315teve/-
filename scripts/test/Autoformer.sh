#export CUDA_VISIBLE_DEVICES=4
seq_len=20
pred_len=4
#root_path=./dataset/cellular_ch_rsrp_rsrq/
#data_path=AST.csv
root_path=./dataset/test/degree/
data_path=processed_data_all.csv

for model_name in Autoformer Informer Transformer
do 

case=only_rssi_degree
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id UAV_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --label_len 10\
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 >logs/$model_name'_'univariate_$case'_'$seq_len'_'$pred_len''.log


data_path=processed_data_all.csv
case=all_sensor_degree
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id UAV_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 10\
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1 > logs/$model_name'_'multivariate_$case'_'$seq_len'_'$pred_len''.log

data_path=processed_data_dis.csv
case=dis_degree
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id UAV_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 10\
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --des 'Exp' \
  --itr 1 > logs/$model_name'_'multivariate_$case'_'$seq_len'_'$pred_len''.log


data_path=processed_data_dis_ang.csv
case=dis_ang_degree
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id UAV_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 10\
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --des 'Exp' \
  --itr 1 > logs/$model_name'_'multivariate_$case'_'$seq_len'_'$pred_len''.log

data_path=processed_data_dis_ang_mot.csv
case=dis_ang_mot_degree
python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id UAV_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 10\
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --itr 1 > logs/$model_name'_'multivariate_$case'_'$seq_len'_'$pred_len''.log

done

