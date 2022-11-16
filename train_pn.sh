model=GCN_pn
for dataset in  'Amazon_eletronics'  'cora-full' #'AmazonCoBuyComputerDataset' 'WikiCSDataset' #  'reddit'
do
python -u train_pn.py --model $model --task_num 4 --epochs 500 --num_repeat 1 --dataset $dataset > logs/$model-$dataset.log 

done