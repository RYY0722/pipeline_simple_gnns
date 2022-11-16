model=GCN_maml
# for dataset in 'Amazon_eletronics' # 'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'Amazon_clothing' #     'Amazon_eletronics' #  'reddit'
# done: 'dblp' 'cora-full' 'email' 
for dataset in 'email' 'CoauthorCSDataset' 'Amazon_clothing' 'Amazon_eletronics'  'dblp' 'cora-full'  'AmazonCoBuyComputerDataset' 'WikiCSDataset' #  'reddit'
do
python -u train_MAML.py --model $model --task_num 4 --epochs 500 --num_repeat 5 --dataset $dataset > logs/$model-$dataset.log &
# echo $!
done