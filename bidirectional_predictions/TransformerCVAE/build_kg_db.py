import logging
import argparse
import os
import torch
from find_kg_embeddings import from_text_to_kb
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == '__main__':
  logger = logging.getLogger("transformers")

  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, default='data')

  args = parser.parse_args()

  logger.info('Setup data...')
  print('Loading wikiplot dataset...')
  data_plots = os.path.join(args.data_dir, 'wikiPlots/plots_paragraph')
  data_titles = os.path.join(args.data_dir, 'wikiPlots/titles')
  with open(data_plots, errors='ignore') as fp:
      plots_text = fp.readlines()
      plots_text = [p.strip() for p in plots_text]
  with open(data_titles, errors='ignore') as ft:
      titles = ft.readlines()

  plots = []
  current_plot = ''

  for line in plots_text:
      current_plot += line.strip()
      if '<EOS>' in line:
          plots.append(current_plot)
          current_plot = ''

  logger.info('Done.')

  logger.info('Creating knowledge graphs...')

  if not os.path.exists(os.path.join(args.data_dir, 'wikiPlots/kg')):
    os.makedirs(os.path.join(args.data_dir, 'wikiPlots/kg'))

  if not os.path.exists(os.path.join(args.data_dir, 'wikiPlots/kg/titles')):
    os.makedirs(os.path.join(args.data_dir, 'wikiPlots/kg/titles'))

  if not os.path.exists(os.path.join(args.data_dir, 'wikiPlots/kg/plots')):
    os.makedirs(os.path.join(args.data_dir, 'wikiPlots/kg/plots'))

  PARALLEL = True
  LAST_EPOCH = 24215

  if PARALLEL:
    with ThreadPoolExecutor(max_workers=16) as executor:
      futures = []
      i = 0
      for title in titles:
        if i > LAST_EPOCH:
          futures.append(executor.submit(from_text_to_kb, title))
        i += 1

      i = LAST_EPOCH
      for future in as_completed(futures):
        kg = future.result()
        torch_kg = kg.to_torch_kg()
        if torch_kg is not None:
          torch.save(torch_kg, os.path.join(args.data_dir, 'wikiPlots/kg/titles', f'title{i}.pt'))

        i += 1

    with ThreadPoolExecutor(max_workers=16) as executor:
      futures = []
      i = 0
      for plot in plots:
        if i > LAST_EPOCH:
          futures.append(executor.submit(from_text_to_kb, plot))
        i += 1

      i = LAST_EPOCH
      for future in as_completed(futures):
        kg = future.result()
        torch_kg = kg.to_torch_kg()
        if torch_kg is not None:
          torch.save(torch_kg, os.path.join(args.data_dir, 'wikiPlots/kg/plots', f'title{i}.pt'))
      
        i += 1

  else:
    for title in titles:
      kg = from_text_to_kb(title)
      torch_kg = kg.to_torch_kg()
      if torch_kg is not None:
        torch.save(torch_kg, os.path.join(args.data_dir, 'wikiPlots/kg', title + '.pt'))

    for plot in plots:
      kg = from_text_to_kb(plot)
      torch_kg = kg.to_torch_kg()
      if torch_kg is not None:
        torch.save(torch_kg, os.path.join(args.data_dir, 'wikiPlots/kg', plot + '.pt'))

  logger.info('Finished creating knowledge graphs.')
