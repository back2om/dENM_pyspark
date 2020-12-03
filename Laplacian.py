import os
from pyspark import SparkConf, SparkContext
import math
import sys
import timeit
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry,RowMatrix,IndexedRowMatrix
import argparse
import numpy as np
import psutil
import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext  

if __name__ == "__main__":
	t0 = timeit.default_timer()


	parser = argparse.ArgumentParser()
	parser.add_argument('pdb', type=str, help='Path to Input pdb file')
	parser.add_argument('cutoff',type=int, help='Cutoff')
	parser.add_argument('gamma',type=int, help='gamma')
	parser.add_argument('k',type=int, help='Number of singular modes')
	args = parser.parse_args()

	conf = SparkConf().setAppName("KirchhoffApp")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)  

	textFile = sc.textFile(args.pdb);
	pairs=textFile.map(lambda line: (line.split()[0],line));
	CAATOMlines = pairs.filter(lambda (key, value): value.split()[0]=="ATOM" and value.split()[2]=="CA");
	chain_coor = CAATOMlines.map(lambda (key, value): (str(value)[21:22], float(str(value)[30:38]), float(str(value)[39:46]), float(str(value)[47:54]) ));
	Chains_Natoms=CAATOMlines.map(lambda (key, value): (str(value)[21:22],1)).reduceByKey(lambda x, y: x + y).sortByKey();
	Chains = Chains_Natoms.map(lambda line: line[0]);
	Natoms = Chains_Natoms.map(lambda line: line[1]);


	coo_matrix_input = sc.emptyRDD()


	Total_atoms = 0
	cutoff=args.cutoff
	gamma=args.gamma
	mb = 1024*1024

	Mat_indices=[];
	for i in range(Natoms.count() + 1):
   		Mat_indices.append(sum(Natoms.take(i)))

	for count1, chain in enumerate(Chains.collect()): 
		nrow = Natoms.collect()[count1]
		Total_atoms += nrow;
		# Get All Chain Combinations*/
		combinations = Chains.cartesian(Chains).filter( lambda (x,y): x==chain and x<=y );
		for count2, comb in enumerate(combinations.collect()):
			#print comb
			if (comb[0] == comb[1]):
				coord = chain_coor.filter(lambda chain_coor: chain_coor[0]==comb[0]).map(lambda x: (x[1],x[2],x[3]));
				Coord_ind = coord.zipWithIndex().map(lambda (values, key): (values, key));
				Combs=Coord_ind.cartesian(Coord_ind);
				# Compute RMSD*/
				rdd_cartesian=Combs.map(lambda (((x1, y1,z1),id1), ((x2, y2,z2),id2)): (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2))));
				rdd2=rdd_cartesian.map(lambda x: (x[0], x[1],-gamma) if (x[2] < cutoff)  else (x[0], x[1],0.0))
				rdd3= rdd2.filter(lambda x: (x[2]!= 0));
				coo_matrix_input = coo_matrix_input.union(rdd3.map(lambda (i,j,v): (i + Mat_indices[count1],j+Mat_indices[count2 + count1],v)));
				coo_matrix_input.cache();
				print comb, coo_matrix_input.count()
			else:
				coord1=chain_coor.filter(lambda chain_coor: chain_coor[0]==comb[0]).map(lambda x: (x[1],x[2],x[3])).zipWithIndex().map(lambda (values, key): (values, int(key)));
				coord2=chain_coor.filter(lambda chain_coor: chain_coor[0]==comb[1]).map(lambda x: (x[1],x[2],x[3])).zipWithIndex().map(lambda (values, key): (values, int(key)));
				Combs=coord1.cartesian(coord2);
				rdd_cartesian=Combs.map(lambda (((x1, y1,z1),id1), ((x2, y2,z2),id2)): (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2))));
				rdd2=rdd_cartesian.map(lambda x: (x[0], x[1],-gamma) if (x[2] < cutoff)  else (x[0], x[1],0.0))
				rdd3= rdd2.filter(lambda x: (x[2]!= 0));
				coo_matrix_input = coo_matrix_input.union(rdd3.map(lambda (i,j,v): (i + Mat_indices[count1],j+Mat_indices[count2 + count1],v)));
				coo_matrix_input.cache();
				print comb, coo_matrix_input.count()



	#Transpose the matrix
	coo_matrix_input_LT = coo_matrix_input.map(lambda (i,j,k): (j,i,k));
	coo_matrix_input_all = coo_matrix_input_LT.union(coo_matrix_input).distinct();
	#coo_matrix_input_all.cache()	


	# Diagonalize RDD  
	diag_entries = coo_matrix_input_all.map(lambda (row, _, value): (row, value)).reduceByKey(lambda x, y: x + y).map(lambda (row,value): (row, row,-value -1));
	nondiag_entries = coo_matrix_input_all.filter(lambda (i,j,k): i!=j);	

	coo_matrix_input_all = nondiag_entries.union(diag_entries);

	#SAVE TO A FILE
	coo_matrix_input_all.repartition(1).saveAsTextFile("./Laplacian_V9_4v7o_16cores_1")
	t2 = timeit.default_timer()
	print("Elapsed time for construction: {:} s".format((t2 - t0)/1000000000.0))


	#Singular value decomposition
	
	coo_matrix_entries = coo_matrix_input_all.map(lambda e: MatrixEntry(e[0], e[1], e[2]));
	coo_matrix = CoordinateMatrix(coo_matrix_entries);
	dataRows = coo_matrix.toRowMatrix().rows


	#entries = sc.parallelize([(0, 0, 1.2), (1, 0, 2.1), (2, 1, 3.7)]);
	#coo_matrix = CoordinateMatrix(entries);
	#dataRows = coo_matrix.toRowMatrix().rows;dataRows = coo_matrix.toRowMatrix().rows;dataRows = coo_matrix.toRowMatrix().rows;

	k = int(args.k) #N_singvalues
	svd = RowMatrix(dataRows.persist()).computeSVD(k, computeU=True)
	U = svd.U # The U factor is a RowMatrix.
	s = svd.s # The singular values are stored in a local dense vector.
	V = svd.V #The V factor is a local dense matrix

	#np.save("EigenVectors_4v7o_4cores", np.array(V))
	#np.save("EigenValues_4v7o_4cores", np.array(s))
	sc.parallelize(V.toArray()).repartition(1).saveAsTextFile("EigenVectors_4v7o_4cores")
	sc.parallelize(s.toArray()).repartition(1).saveAsTextFile("EigenValues_4v7o_4cores")

	t4 = timeit.default_timer()
	print("Elapsed time for SVD: {:} s".format((t4 - t2)/1000000000.0))
	print("Total memory = {:}, used memory = {:}, free memory = {:}".format(psutil.virtual_memory().total/mb, (psutil.virtual_memory().total - psutil.virtual_memory().free) / mb, psutil.virtual_memory().free/mb));

	print("System size = {:} atoms".format(Natoms.sum()))
	print("No. of chains ={:}".format(Chains.count()))






