import os
import sys
import json
import pandas as pd
from absl import app

from data import TsvData
from utils.Configuration import Config
from utils.text import TextProcessing
from model_search import single_trainer

sys.argv = sys.argv[:1]
try:
  app.run(lambda argv: None)
except:
  pass

config = Config()
tokenizer_path = os.path.join(config.stagging_folder, config.tokenizer_file)

if not os.path.exists(config.stagging_folder):
  os.mkdir(config.stagging_folder)
  print(f'[INFO] => creating directory {config.stagging_folder}')


if not os.path.exists(tokenizer_path):
  data_provider = TsvData.provider(**config.dataset_config)

  input_func = data_provider.get_input_fn(None, None, 1)
  dataset = input_func()
  tokenizer = TextProcessing(vocab_size = config.vocab_size, max_sequence_length = config.max_sequence_length)
  print('[INFO] => started adapting')
  tokenizer.adapt(dataset)

  print(f'[INFO] => Exporting configs to {tokenizer_path}')
  tokenizer.export(tokenizer_path)


trainer = single_trainer.SingleTrainer(
      TsvData.provider(
        **config.dataset_config,
        tokenizer_path = tokenizer_path
        ) ,
     spec=config.spec_path)

trainer.try_models(
    number_models=config.number_models,
    train_steps=config.train_steps,
    eval_steps=config.eval_steps,
    root_dir= config.search_staging_dir,
    batch_size=config.dataset_config['batch_size'],
    experiment_name=config.experiment_name,
    experiment_owner="model_search_user")


path = os.path.join(config.search_staging_dir, config.experiment_name)
trial_ids = os.listdir(path)

records = []
for trial in trial_ids:
  file = os.path.join(path, trial, 'trial.json')
  if os.path.isdir(os.path.join(path, trial)):
    with open(file, 'r') as handler:
      data = json.load(handler)
    temp_ = data['metrics']['metrics']
    if temp_:
      row = {'accuracy' : temp_['accuracy']['observations'][0]['value'][0],
          'global_step' : temp_['global_step']['observations'][0]['value'][0],
          'loss': temp_['loss']['observations'][0]['value'][0],
          'num_parameters': temp_['loss']['observations'][0]['value'][0],
          'score': data['score'],
          'status' : data['status'],
          'trial_id' : data['trial_id']
      } 
      records.append(row)

csv_path = os.path.join(config.search_staging_dir, 'experiment_report.csv')
data = pd.DataFrame(records)
data.to_csv(csv_path, index = False)

print(f'[INFO] => experiment short summary is exported in {csv_path}')
print(f'[NOTE] => for more details on individual trial check files in "{path}"')