val chennai=spark.read.format("csv").option("inferSchema","true").option("header","true").load("./data.csv")

chennai.printSchema

val cx=chennai.withColumnRenamed("Name of Restaurant","name")


// csv val to list
import org.apache.spark.sql.functions.split   
// val omrTopDishes=omr.withColumn("dishes",saplit(col("Top Dishes"),","))


// chennai Cuisine exploded
val chennaiCEx=cx.withColumn("c",explode(split(col("Cuisine"),",")))         


val restCuisine=chennaiCEx.select('name,'c)   

import org.apache.spark.ml.feature.StringIndexer 
// an instance of string Indexer
val si = new StringIndexer()
// set the io
si.setInputCol("c")
si.setOutputCol("cuisineIndex")  

OR
val si = new StringIndexer().setInputCol("c").setOutputCol("cuisineIndex")  

// this just assigns an id to the string values.. jus like the dict you were plannin to do
// need to activate this guy, using our data.. which is aka fit.

val restCuisineModel=si.fit(restCuisine)
val restCuisineIndexed=restCuisineModel.transform(restCuisine)
restCuisineIndexed.show



// Enter OneHotEncoder..
import org.apache.spark.ml.feature.OneHotEncoder
val cuisineOHE=new OneHotEncoder().setInputCol("cuisineIndex").setOutputCol("cuisineFeatures")
val cuisineEncoded=cuisineOHE.transform(restCuisineIndexed)
cuisineEncoded.show

// indexing name
val siName = new StringIndexer().setInputCol("name").setOutputCol("nameIndex")
// get name Index

//  we need this so we can use this against test data for predictions.
val cuisineE1NameModel=siName.fit(cuisineEncoded)



val cuisineE1=siName.fit(cuisineEncoded).transform(cuisineEncoded)
 

// get vector for name Index
val cuisineOHEName=new OneHotEncoder().setInputCol("nameIndex").setOutputCol("nameVec")
val cuisineAllVec=cuisineOHEName.transform(cuisineE1)
cuisineAllVec.show


import org.apache.spark.ml.feature.VectorAssembler  
val va=new VectorAssembler().setInputCols(Array("cuisineFeatures","nameVec")).setOutputCol("cusineF") 
val cuisineFeatures=va.transform(cuisineAllVec)

// now that we have a feature vector which has the vector for name and vector for cuisine, 
//  lets cluster them!!

// bring in Kmeans
 import org.apache.spark.ml.clustering.KMeans

//  create a cluster with 2000 clusters and set the Feature column to be "cusineF" the one that
// has both name and cuisine vectors

// val k=new KMeans().setK(2000).setFeaturesCol("cusineF")

val k=new KMeans().setK(200).setFeaturesCol("cusineF")

val BModel=k.fit(cuisineFeatures)

val cuisineClusters=BModel.transform(cuisineFeatures)

cuisineClusters.show

// avoid mem error.. 10G 
// val bData=cuisineFeatures.filter('name.like("%yani"))
// val BModel=k.fit(bData)
// val biriyaniClusters=BModel.transform(bData)    

val cusinePredicted=cuisineClusters.select('name,'c,'prediction)

val ctByCluster=cusinePredicted.groupBy('prediction,'c).agg(count('name) as "restInCluster")

val x=ctByCluster.orderBy('restInCluster.desc)

def showRest(p:Int,x:org.apache.spark.sql.DataFrame)=x.filter('prediction===p).show(500)

showRest(1,cusinePredicted) 

// evalucate 
import org.apache.spark.ml.evaluation.ClusteringEvaluator
val ee=new ClusteringEvaluator()
ee.setFeaturesCol("cusineF").setPredictionCol("prediction")
ee.evaluate(cuisineClusters)

// sample prediction
val tData=Seq(("Sukkubhai Biriyani","Biryani")).toDF("name","c")
// repeat till u have vecs


// index Cuisine
val tci=restCuisineModel.transform(tData)
val tcF=cuisineOHE.transform(tci)

// index name
// cuisineE1NameModel.setHandleInvalid("keep") 

val tni=cuisineE1NameModel.transform(tcF)

// feature name, feature cuisine
val tnF=cuisineOHEName.transform(tni)

val testFeatures=va.transform(tnF)

val Y=BModel.transform(testFeatures)

// 
cusinePredicted.join(f,cusinePredicted.col("prediction")===f.col("prediction")).select('name).distinct.show

def showSim(name:String, ty:String)={
    val f=cusinePredicted.filter('name===name).filter(trim('c)===ty).select('prediction) 
    cusinePredicted.join(f,cusinePredicted.col("prediction")===f.col("prediction")).select('name).distinct.show
}