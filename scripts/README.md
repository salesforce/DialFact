# Claim Verification

## Claim verification model training
The training code is based on the VitaminC repo. The training code is a fork of the [VitaminC repo](https://github.com/TalSchuster/VitaminC) and uploaded to a drive folder here: [Code](https://drive.google.com/drive/folders/1eu0ZDdFH7z4wSG5OZ4rF-HH57K6d2AkV?usp=sharing). The data used for training Augwow model is present in the same drive link in the data folder here: [data](https://drive.google.com/drive/folders/1CSciq9f3ZOvLuNk9m3aDElPVVLjpYfuv?usp=sharing)
The script for training is in the above folder:
```
cd scripts
bash train_fact_verification.sh
```

### Trained model
AugWoW model trained is present here [link](https://drive.google.com/drive/folders/1LCos8WNj01LBYiDVXy8yiGzuhQs0mM3J?usp=sharing)

## Model inference
For augwow model 
```
python run_fever_scoringctx.py --input_file valid_split.jsonl --output_folder $OUTFOLDER --typeprefix augwow_ --model $MODEL_FOLDER --outputprefix wctx
```

For VitaminC or colloquial models
```
python run_fever_scoring.py --input_file valid_split.jsonl --output_folder $OUTFOLDER --typeprefix augwow_ --model $MODEL_FOLDER --outputprefix wctx
```
for VitaminC, $MODEL_FOLDER = tals/albert-base-vitaminc-fever


## Output evaluation
INPUT_FILE_WITHSCORES is the file generated from the last step. sample input file testsmall.jsonl is provided
```
python run_metrics.py --input_file $INPUT_FILE_WITHSCORES
```


# Evidence Retrieval
DPR models trained on WoW data is uploaded on drive [link](https://drive.google.com/drive/folders/1d7U_nenH8iRvRWcAMZh79rhifdox1DfS?usp=sharing)

To run DPR based retrieval:
```
cd retrieval
python dpr_docretrieval.py -i $INPUT_FILE --topk 100 --model_type ftwctx --use_context
```

To run evidence sentence selection after document retrieval, use the get_evidence_scores function from the bert_evidence_selection.py
```