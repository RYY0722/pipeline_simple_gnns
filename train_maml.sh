model=GCN_maml
# for dataset in 'Amazon_eletronics' # 'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'Amazon_clothing' #     'Amazon_eletronics' #  'reddit'
# done: 'dblp' 'cora-full' 'email' 
for dataset in 'dblp' 'cora-full' 'CoauthorCSDataset' 'Amazon_clothing' 'Amazon_eletronics' 'email' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' #  'reddit'
do
python -u train_MAML.py --model $model --task_num 16 --epochs 500 --num_repeat 5 --dataset $dataset > logs/$model-$dataset.log
echo $!
done