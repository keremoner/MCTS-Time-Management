import pandas as pd
import modin.pandas as mpd
import time
 
# Reading demo.csv file into pandas df
df = pd.read_csv("../datasets/deneme.csv")
 
s = time.time()
df = df.fillna(value=0)
 
e = time.time()
print(f"Pandas fillna Time: {e-s}")
 
# Reading demo.csv file into modin df
modin_df = mpd.read_csv("demo.csv")
s = time.time()
       
modin_df = modin_df.fillna(value=0)
e = time.time()
print(f"Modin fillna Time: {e - s}")