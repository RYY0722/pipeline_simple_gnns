model=GCN  # [GAT, GCN, GPN, GraghSage]
for dataset in 'CoauthorCSDataset'
do
python -u main_train.py --model $model --episodes 500 --num_repeat 5 --dataset $dataset > logs/$model-$dataset.log
echo $!
done