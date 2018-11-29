import numpy as np
from scipy.spatial.distance import euclidean as ec_dist
import math
from timeit import default_timer as timer

"""kmeans.py: implements kmeans class"""

__author__		= "Nikhil Tumkur Ramesh"
__copyright__	= "Copyright 2018, IIT Mandi" 

def mag2(X):
	""" mag2: Returns the sum of squares of a vector.
			  To be read as "Magnitude Square"
	"""
	return np.sum(X**2, dtype=np.float64)



class kmeans():

	'''
	Attributes:
	n_clusters: stores the number of clusters to be made.
				default value: 2
	max_iter:	stores the upper limit for EM iteration.
				default value: 300
	means:		stores the mean vectors.
	clusters:	stores the current allocation of points.
	old_clusters:	stores the previous allocation of points. It is used in the
				stopping criteria.
	'''
	def __init__(self, n_clusters=2, max_iter=300):
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def initialize_clusters(self, X):
		self.clusters=np.array([-1]*X.shape[0])
		self.old_clusters=np.array([-2]*X.shape[0])

	def initialize_means(self, X, n_clusters, debug_mode=False):
		self.means=X[0:n_clusters,:].astype(dtype=np.float64)
		self.initialize_clusters(X)
		# Perform check to ensure the cluster means are different. Otherwise you
		# will run into all kinds of errors later.
		last_index=n_clusters-1
		all_different=False
		while not all_different:
			all_different=True
			for idx, mean in enumerate(self.means):
				for i in range(idx):
					if np.array_equal(self.means[i], mean):
						self.means[idx]=X[last_index+1,:].astype(dtype=np.float64)
						last_index+=1
						all_different=False
						break
		self.old_means=np.zeros_like(self.means, dtype=np.float64)

	''' Method: fit(input, number_of_clusters)
		The main method used for performing the kmeans algorithm. it is first 
		initialized and then the EM loop is run untill convergence criteria is
		reached.
	'''
	def fit(self, X, n_clusters, display_progress=False, debug_mode=False, calculate_cov=True, cap=True):
		self.X=X
		# start = timer()
		self.n_clusters = n_clusters
		self.initialize_means(X, n_clusters, debug_mode)
		counter=0
		dist=0.0
		for idx in range(self.means.shape[0]):
			dist+=ec_dist(self.means[idx], self.old_means[idx])
		if display_progress:
			print("K=", self.n_clusters, ":Iteration #", counter,  ":", dist, "distance shifted")
		while (dist>1e-3) and (not cap or counter < self.max_iter):
			start = timer()
			# print('loop started at:', start)
			self.assign_clusters(X)
			self.update_means(X)
			counter += 1
			if display_progress:
				dist=0.0
				for idx in range(self.means.shape[0]):
					dist+=ec_dist(self.means[idx], self.old_means[idx])
				if display_progress:
					print("K=", self.n_clusters, ":Iteration #", counter,  ":", dist, "distance shifted")
			end = timer()
			if display_progress:
				print('That took', end-start, 'seconds.')

		if calculate_cov:
			self.calculate_cov(X)

	''' Method: assign_clusters(input)
		A helper method for allocating the points to clusters. Other required 
		data being used are the attributes means. 
	'''
	def assign_clusters(self, X):
		self.transfer_clusters(X)
		for i in range(X.shape[0]):
			self.distances=np.array([], dtype=np.float64)
			for mean in self.means:
				self.distances=np.append(self.distances,math.sqrt(mag2(X[i]-mean)))
			self.clusters[i] = self.distances.argmin()


	def update_means(self, X):
		self.old_means = np.copy(self.means)
		for i in range(self.means.shape[0]):
			points=[]
			for idx, point in enumerate(X):
				if self.clusters[idx]==i:
					points.append(point)
			self.means[i]=np.mean(points,axis=0, dtype=np.float64)


	def transfer_clusters(self, X):
		try:
			self.old_clusters = np.copy(self.clusters)
		except:
			self.initialize_clusters(X)

	def cluster_means(self):
		return self.means

	def gen_list_of_points(self):
		self.list_of_points=[[] for i in range(self.n_clusters)]
		for idx, i in enumerate(self.clusters):
			self.list_of_points[i].append(idx)
		
	def calculate_cov(self, X):
		self.covariances=np.ones_like(self.means, dtype=np.float64)
		self.list_of_points=[[] for i in range(self.n_clusters)]
		for idx, i in enumerate(self.clusters):
			self.list_of_points[i].append(idx)
		for idx, l in enumerate(self.list_of_points):
			if len(l)<2:
				print("the ", idx, "cluster has ", len(l), " point(s). skip? Y/N")
				c=input(">>")
				if(not c=="N"):
					print("cov retained as ones.")
					continue
			self.covariances[idx]=np.var(X[l,:], axis=0)

	def terminate(self):
		np.savetxt('terminate.txt', self.old_means)

		