from multiprocessing import Process, Queue, Pool
import multiprocessing
import numpy as np

def multi_work(thelist,func,arguments=[],iterable_input=True,scaling_number=4,on_disk=True):
	low = 0
	threads_number = multiprocessing.cpu_count()-1
	gap = int(np.ceil(len(thelist)/(threads_number*scaling_number)))
	queue = Queue()
	NP = 0
	subprocesses = []
	if iterable_input==True:
#		p = Pool()
		while low < len(thelist):
#			x= p.map(func=func, iterable=thelist,*arguments)
			p = Process(target=func, args=[thelist[low:low+gap],queue]+arguments)
			NP += 1
			p.start()
			low += gap
			subprocesses.append(p)
	elif iterable_input==False:
		def single_mapper(xs,q=None,func=None,on_disk=True,arguments=[]):
#			_thelist = thelist[0:1454]
#			xs = _thelist
#			q=queue
			outputs = []
			if type(xs[0]) != tuple:
				enumerater = list(enumerate(xs))
			else:
				enumerater = xs
			orders=[]
			for i,x in enumerater:
				orders.append(i)
				y=func(*[x]+arguments)
				outputs.append(y)

			out = list(zip(orders,outputs))
			if on_disk==False:
				if q !=None:
					q.put(out)
			return out
		i = 0
		while low < len(thelist):
			print('Starting thread {}...'.format(i))
			p = Process(target=single_mapper, args=[thelist[low:low+gap],queue]+[func]+[on_disk]+arguments)
			NP += 1
			p.start()
			low += gap
			subprocesses.append(p)
			i+=1
	#merge
	outs = []
	if on_disk==False:
		for i in range(NP):
			outs.append([queue.get()])
		end = [p.terminate() for p in subprocesses]

	return outs


"""
thelist=list(enumerate(dfs))
func=get_bin_dfs
arguments=[[n_pieces,now_utc,past_time_utc,spam_list]]
iterable_input=False
scaling_number=1
on_disk=False
"""
