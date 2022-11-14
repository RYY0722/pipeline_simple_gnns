model=GCN  # [GAT, GCN, GPN, GraghSage]
for dataset in 'dblp' #'cora-full' 'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'Amazon_clothing' 'Amazon_eletronics' 'dblp' 'email' 'reddit'
do
python -u main_train.py --model $model --episodes 15 --num_repeat 2 --dataset $dataset > logs/$model-$dataset.log
echo $!
done