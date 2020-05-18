val chennai=spark.read.format("csv").option("inferSchema","true").option("header","true").load("./data.csv")

chennai.printSchema

val cx=chennai.withColumnRenamed("Name of Restaurant","name")

val ctByLoc=cx.groupBy('Location).agg(count(lit(1)).as("ct")).orderBy('ct.desc)

val perambur=cx.where("Location = 'Perambur'").select('name,'Address)

perambur.selectExpr("Address","name","cast(votes as int) as v").select('name,'Address,'v).show(10,false)

val Bi=cx.filter('Cuisine.like("%Biry%")).select('name,'Location).filter('Location.like("%mbur%"))

val withCtByLoc=cx.withColumn("b_ct",count(lit(1)) over Window.partitionBy('Location))  

val biriuaniByLocationCt = Bi.withColumn("LocBiriyaniCt",count('name) over Window.partitionBy('Location)).orderBy('LocBiriyaniCt.desc)


// csv val to list
import org.apache.spark.sql.functions.split   
val omrTopDishes=omr.withColumn("dishes",saplit(col("Top Dishes"),","))


// exploding:
val omrx= omrTopDishes.select('name,'dishes) 
val omrxpl=omrx.withColumn("ex_di",explode('dishes))

// perambur Exploded Dishes
val peramburDishes=perambur.withColumn("dishes",split(col("Top Dishes"),",")).withColumn("d",explode('dishes))

val p=peramburDishes.select('name,'Address,'d).filter('d!=="Invalid").orderBy('d).show(100,false)


// chennai Cuisine exploded
val chennaiCEx=cx.withColumn("c",explode(split(col("Cuisine"),",")))         


// ML
// VectorAssembler

import org.apache.spark.ml.feature.VectorAssembler

// fun fun
//  restaurantName and Cuisine served => get a StringIndex.

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


// find length of value in a column
chennaiCEx.withColumn("c-t",trim('c)).withColumn("cl",length('c)).withColumn("ccl",length(col("c-t"))).show

// ///////////////////////////////////////////////

// trim cuisine 
val chnData=chennaiCEx.withColumn("ct",trim('c)).drop('c)

val restCuisine=chnData.select('name,'ct)   


val si = new StringIndexer().setInputCol("ct").setOutputCol("cuisineIndex")  


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

// ///////////////////////////////////////////////

// now we have vectorized or Featurized..
//  cuisineEncoded now has features... Wow!

scala> cuisineEncoded.show
+--------------------+------------+------------+---------------+
|                name|          ct|cuisineIndex|cuisineFeatures|
+--------------------+------------+------------+---------------+
|    The Big Barbeque|North Indian|         1.0| (98,[1],[1.0])|
|    The Big Barbeque| Street Food|         7.0| (98,[7],[1.0])|
|    The Big Barbeque|    Desserts|         6.0| (98,[6],[1.0])|
|    The Big Barbeque|         BBQ|        21.0|(98,[21],[1.0])|
|AB's - Absolute B...|         BBQ|        21.0|(98,[21],[1.0])|
|AB's - Absolute B...|North Indian|         1.0| (98,[1],[1.0])|
|AB's - Absolute B...|    European|        37.0|(98,[37],[1.0])|
|          McDonald's|      Burger|        19.0|(98,[19],[1.0])|
|          McDonald's|   Fast Food|         3.0| (98,[3],[1.0])|
|  Sukkubhai Biriyani|     Biryani|         5.0| (98,[5],[1.0])|
|  Sukkubhai Biriyani|North Indian|         1.0| (98,[1],[1.0])|
|  Sukkubhai Biriyani|     Mughlai|        18.0|(98,[18],[1.0])|
|  Sukkubhai Biriyani|    Desserts|         6.0| (98,[6],[1.0])|
|  Sukkubhai Biriyani|   Beverages|         4.0| (98,[4],[1.0])|
|      Coal Barbecues|North Indian|         1.0| (98,[1],[1.0])|
|      Coal Barbecues|     Chinese|         0.0| (98,[0],[1.0])|
|Yaa Mohaideen Bri...|     Biryani|         5.0| (98,[5],[1.0])|
|     The Black Pearl|North Indian|         1.0| (98,[1],[1.0])|
|     The Black Pearl|     Mughlai|        18.0|(98,[18],[1.0])|
|     The Black Pearl|   Fast Food|         3.0| (98,[3],[1.0])|
+--------------------+------------+------------+---------------+
only showing top 20 rows

// what we have done here is, we have converted text to a number value. 
// we have set the cusineType as the label and converted that label into a vector.. (N,idx,value)

// indexing name
val siName = new StringIndexer().setInputCol("name").setOutputCol("nameIndex")
// get Index for name index 
val cuisineE1=siName.fit(cuisineEncoded).transform(cuisineEncoded)
scala> cuisineE1.show
+--------------------+------------+------------+---------------+---------+
|                name|          ct|cuisineIndex|cuisineFeatures|nameIndex|
+--------------------+------------+------------+---------------+---------+
|    The Big Barbeque|North Indian|         1.0| (98,[1],[1.0])|   1358.0|
|    The Big Barbeque| Street Food|         7.0| (98,[7],[1.0])|   1358.0|
|    The Big Barbeque|    Desserts|         6.0| (98,[6],[1.0])|   1358.0|
|    The Big Barbeque|         BBQ|        21.0|(98,[21],[1.0])|   1358.0|
|AB's - Absolute B...|         BBQ|        21.0|(98,[21],[1.0])|   2612.0|
|AB's - Absolute B...|North Indian|         1.0| (98,[1],[1.0])|   2612.0|
|AB's - Absolute B...|    European|        37.0|(98,[37],[1.0])|   2612.0|
|          McDonald's|      Burger|        19.0|(98,[19],[1.0])|   5098.0|
|          McDonald's|   Fast Food|         3.0| (98,[3],[1.0])|   5098.0|
|  Sukkubhai Biriyani|     Biryani|         5.0| (98,[5],[1.0])|    510.0|
|  Sukkubhai Biriyani|North Indian|         1.0| (98,[1],[1.0])|    510.0|
|  Sukkubhai Biriyani|     Mughlai|        18.0|(98,[18],[1.0])|    510.0|
|  Sukkubhai Biriyani|    Desserts|         6.0| (98,[6],[1.0])|    510.0|
|  Sukkubhai Biriyani|   Beverages|         4.0| (98,[4],[1.0])|    510.0|
|      Coal Barbecues|North Indian|         1.0| (98,[1],[1.0])|   4836.0|
|      Coal Barbecues|     Chinese|         0.0| (98,[0],[1.0])|   4836.0|
|Yaa Mohaideen Bri...|     Biryani|         5.0| (98,[5],[1.0])|   6722.0|
|     The Black Pearl|North Indian|         1.0| (98,[1],[1.0])|   2006.0|
|     The Black Pearl|     Mughlai|        18.0|(98,[18],[1.0])|   2006.0|
|     The Black Pearl|   Fast Food|         3.0| (98,[3],[1.0])|   2006.0|

// get vector for name
val cuisineOHEName=new OneHotEncoder().setInputCol("nameIndex").setOutputCol("nameVec")
val cuisineAllVec=cuisineOHEName.transform(cuisineE1)
cuisineAllVec.show

//  now we have 2 of our cols indexed and vectorized.
//  lets get a feature vector out of these 2.

import org.apache.spark.ml.feature.VectorAssembler  
val va=new VectorAssembler().setInputCols(Array("cuisineFeatures","nameVec")).setOutputCol("cusineF") 
val cuisineFeatures=va.transform(cuisineAllVec)

scala> val cuisineFeatures=va.transform(cuisineAllVec)
cuisineFeatures: org.apache.spark.sql.DataFrame = [name: string, ct: string ... 5 more fields]

scala> .show
+--------------------+------------+------------+---------------+---------+-------------------+--------------------+
|                name|          ct|cuisineIndex|cuisineFeatures|nameIndex|            nameVec|             cusineF|
+--------------------+------------+------------+---------------+---------+-------------------+--------------------+
|    The Big Barbeque|North Indian|         1.0| (98,[1],[1.0])|   1358.0|(7890,[1358],[1.0])|(7988,[1,1456],[1...|
|    The Big Barbeque| Street Food|         7.0| (98,[7],[1.0])|   1358.0|(7890,[1358],[1.0])|(7988,[7,1456],[1...|
|    The Big Barbeque|    Desserts|         6.0| (98,[6],[1.0])|   1358.0|(7890,[1358],[1.0])|(7988,[6,1456],[1...|



// now that we have a feature vector which has the vector for name and vector for cuisine, 
//  lets cluster them!!

// bring in Kmeans
 import org.apache.spark.ml.clustering.KMeans

//  create a cluster with 2000 clusters and set the Feature column to be "cusineF" the one that
// has both name and cuisine vectors

val k=new KMeans().setK(2000).setFeaturesCol("cusineF")

// now fit
// memory error.. so lets use 

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

val ctByCluster=cusinePredicted.groupBy('prediction).agg(count('name) as "restInCluster")

val ctByCluster=cusinePredicted.groupBy('prediction,'c).agg(count('name) as "restInCluster")

def getNames(p:Int,x:org.apache.spark.sql.DataFrame)=x.filter('prediction===p).show

val x=ctByCluster.orderBy('restInCluster.desc)
x.show
+----------+--------------+-------------+
|prediction|             c|restInCluster|
+----------+--------------+-------------+
|         6|       Biryani|          375|
|         1|       Chinese|          203|
|        11|  North Indian|          103|
|         0|  South Indian|           71|
|        12|       Biryani|           57|
|         2|  North Indian|           40|
|         5|     Fast Food|           36|
|        28|       Chinese|           33|
|        36|     Chettinad|           21|
|        85|       Mughlai|           20|
|         3|  South Indian|           16|
|         8|       Seafood|           15|
|        53|       Arabian|           14|
|        16|     Beverages|           11|
|        40|           BBQ|            6|
|        56|    Hyderabadi|            6|
|        56|         Kebab|            6|
|        10|       Arabian|            5|
|        56|       Mughlai|            4|
|        43|     Chettinad|            4|
|        73|        Kerala|            3|
|        80|        Andhra|            3|
|       168|       Biryani|            2|
|        13|     Fast Food|            2|
|       187|  North Indian|            2|
|         4|      Desserts|            2|
|        35|         Momos|            2|
|        69|         Tamil|            2|
|       153|          Naga|            1|
|        52|       Chinese|            1|
|        37|         Rolls|            1|
|       151|       Biryani|            1|
|        24|        Andhra|            1|
|        98| Maharashtrian|            1|
|        56|          Naga|            1|
|        56|           600|            1|
|        17|     Ice Cream|            1|
|        14|           BBQ|            1|
|        47|   Street Food|            1|
|       123|           BBQ|            1|
|        49|      Lebanese|            1|
+----------+--------------+-------------+

def showRest(p:Int,x:org.apache.spark.sql.DataFrame)=x.filter('prediction===p).show(500)

showRest(1,cusinePredicted) 