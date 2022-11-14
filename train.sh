model=GCN  # [GAT, GCN, GPN, GraghSage]
for dataset in 'cora-full' #'CoauthorCSDataset' 'AmazonCoBuyComputerDataset' 'WikiCSDataset' 'cora'
do
python -u main_train.py --model $model --episodes 15 --num_repeat 2 --dataset $dataset > logs/$model-$dataset.log
echo $!
done