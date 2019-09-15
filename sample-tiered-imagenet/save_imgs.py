import torch
import os, sys
from PIL import Image
from pathlib import Path


if __name__ == '__main__':

  assert len(sys.argv) == 2, 'invalid sys.argv : {:}'.format(sys.argv)
  dataset_save_dir = Path.home() / 'datasets' / sys.argv[1]
  assert dataset_save_dir.exists(), 'invalid : {:}'.format( dataset_save_dir )
  cache_file_path = dataset_save_dir / 'cache-raw-data-tiered.pth'
  assert cache_file_path.exists(), 'invalid : {:}'.format( cache_file_path )

  print ('start loading {:}'.format(cache_file_path))
  idx2img = torch.load( '{:}'.format(cache_file_path) )[ "all_images" ]
  saveDir = dataset_save_dir / 'IMGS'
  saveDir.mkdir(exist_ok=True)
  print ('idx2img is a {:}, with shape = {:}'.format( type(idx2img), idx2img.shape ))

  for i in range( len(idx2img) ):
    img = idx2img[i]
    pil_image = img[:, :, ::-1]
    save_path = saveDir / "{:}.pth".format(i)
    torch.save(pil_image, save_path)
    if i % 1000 == 0: print('{:05d}/{:05d} : {:}'.format(i, len(idx2img), save_path))
