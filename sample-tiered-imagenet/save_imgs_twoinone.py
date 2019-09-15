import torch
import os, sys
from PIL import Image
from pathlib import Path


if __name__ == '__main__':

  assert len(sys.argv) == 4, 'invalid sys.argv : {:}'.format(sys.argv)
  
  dataset_load_dir1 = Path.home() / 'datasets' / sys.argv[1]
  dataset_load_dir2 = Path.home() / 'datasets' / sys.argv[2]

  if ( Path.home() / 'datasets' / sys.argv[3] ).exists():
    print("IMGS already saved")
  else:
    print("start to save IMGS")
  
    
    assert dataset_load_dir1.exists(), 'invalid : {:}'.format( dataset_load_dir1 )
    assert dataset_load_dir2.exists(), 'invalid : {:}'.format( dataset_load_dir2 )
    cache_file_path = dataset_load_dir1 / 'cache-raw-data-tiered.pth'
    assert cache_file_path.exists(), 'invalid : {:}'.format( cache_file_path )

    print ('start loading {:}'.format(cache_file_path))
    idx2img = torch.load( '{:}'.format(cache_file_path) )[ "all_images" ]
    saveDir = Path.home() / 'datasets' / sys.argv[3] 
    saveDir.mkdir(exist_ok=True)
    print ('idx2img is a {:}, with shape = {:}'.format( type(idx2img), idx2img.shape ))

    for i in range( len(idx2img) ):
      img = idx2img[i]
      pil_image = img[:, :, ::-1]
      save_path = saveDir / "{:}.pth".format(i)
      torch.save(pil_image, save_path)
      if i % 1000 == 0: print('{:05d}/{:05d} : {:}'.format(i, len(idx2img), save_path))
  '''
  print("start to delete unuseful info ... ")
  for load_dir in [dataset_load_dir1, dataset_load_dir2]:
    os.system("rm {}/cache-class2leaf-info.pth".format(load_dir)) 
    os.system("rm {}/cache-raw-data-tiered.pth".format(load_dir)) 
    os.system("rm {}/cache-train_test_class-info.pth".format(load_dir)) 
    #os.system("rm {}/idx2wordid.pth".format(load_dir)) 
    os.system("rm {}/all_info.pth".format(load_dir)) 
    os.system("rm {}/test_info.pth".format(load_dir)) 
    os.system("rm {}/train_info.pth".format(load_dir)) 
    #os.system("rm {}/wordid2idx.pth".format(load_dir)) 
 '''   


