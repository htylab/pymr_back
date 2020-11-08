import pymri.heart.molli as molli
im, Invtime = molli.read_molli_dir(r'C:\expdata\19+25_zip\10\PRE_B')
T1map, Amap, Bmap,T1starmap,Errmap = molli.T1LLmap(im, Invtime)