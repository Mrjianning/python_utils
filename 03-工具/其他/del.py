import os
import shutil
filelist=[]
rootdir=r"H:\人工智能数据集\exp"
filelist=os.listdir(rootdir)
for f in filelist:
  filepath = os.path.join( rootdir,f)
  if os.path.isfile(filepath):
    os.remove(filepath)
  elif os.path.isdir(filepath):
    shutil.rmtree(filepath,True)
    print('remove successful')

shutil.rmtree(rootdir,True)
