"""Bounded one-batch lookahead; preserve original frame and video order."""
from concurrent.futures import ThreadPoolExecutor

def ordered_batches(items, reader, predict_many, batch_size):
    if batch_size not in (1,4,8):raise ValueError('Qualified batch sizes only')
    def load(start):
        part=items[start:start+batch_size]
        values=[reader(item)for item in part]
        depths=predict_many([v[1]for v in values])
        if len(depths)!=len(part):raise ValueError('Prediction count differs')
        return [(start+i,item,*value,depth)for i,(item,value,depth)in enumerate(zip(part,values,depths))]
    with ThreadPoolExecutor(max_workers=1)as pool:
        future=pool.submit(load,0)if items else None
        for start in range(0,len(items),batch_size):
            current=future.result()
            future=pool.submit(load,start+batch_size)if start+batch_size<len(items)else None
            yield from current
