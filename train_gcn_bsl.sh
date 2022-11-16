model=GCN_basic
# for dataset in 'Amazon_eletronics' # 'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'Amazon_clothing' #     'Amazon_eletronics' #  'reddit'
# done: 'dblp' 'cora-full' 'email' 
for dataset in 'dblp' 'cora-full' 'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'Amazon_clothing' 'Amazon_eletronics' 'email'#  'reddit'
do
python -u main_traingcn_baseline.py --model $model --epochs 500 --num_repeat 5 --dataset $dataset > logs/$model-$dataset.log
echo $!
done