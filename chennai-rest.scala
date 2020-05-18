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

// evalucate 
import org.apache.spark.ml.evaluation.ClusteringEvaluator
val ee=new ClusteringEvaluator()
ee.setFeaturesCol("cusineF").setPredictionCol("prediction")
ee.evaluate(cuisineClusters)
res94: Double = 0.16987298070930856


scala> a.groupBy(trim('Location)).count().orderBy('count.desc).show(1000,false)
+---------------------------------------------------------------------------------+-----+
|trim(Location)                                                                   |count|
+---------------------------------------------------------------------------------+-----+
|Porur                                                                            |293  |
|Velachery                                                                        |254  |
|T. Nagar                                                                         |229  |
|Anna Nagar East                                                                  |226  |
|Ambattur                                                                         |212  |
|Nungambakkam                                                                     |206  |
|Perungudi                                                                        |197  |
|Perambur                                                                         |189  |
|Tambaram                                                                         |167  |
|Anna Nagar West                                                                  |164  |
|Ramapuram                                                                        |160  |
|Medavakkam                                                                       |152  |
|Thuraipakkam                                                                     |147  |
|Kolathur                                                                         |145  |
|Selaiyur                                                                         |143  |
|Madipakkam                                                                       |141  |
|GST Road                                                                         |140  |
|Sholinganallur                                                                   |135  |
|Pallavaram                                                                       |132  |
|Kilpauk                                                                          |132  |
|Chromepet                                                                        |121  |
|Mogappair                                                                        |121  |
|Adyar                                                                            |118  |
|Mylapore                                                                         |116  |
|Thiruvanmiyur                                                                    |106  |
|Potheri                                                                          |106  |
|Poonamalle                                                                       |103  |
|Kodambakkam                                                                      |102  |
|Ashok Nagar                                                                      |102  |
|Egmore                                                                           |95   |
|Kanchipuram District                                                             |94   |
|Avadi                                                                            |93   |
|Alwarpet                                                                         |89   |
|Taramani                                                                         |87   |
|Choolaimedu                                                                      |82   |
|Washermenpet                                                                     |82   |
|Triplicane                                                                       |82   |
|Vadapalani                                                                       |81   |
|Adambakkam                                                                       |80   |
|K.K. Nagar                                                                       |80   |
|Besant Nagar                                                                     |80   |
|Royapettah                                                                       |78   |
|Pallikaranai                                                                     |76   |
|Valasaravakkam                                                                   |75   |
|Purasavakkam                                                                     |75   |
|Navallur                                                                         |73   |
|Chengalpattu                                                                     |73   |
|Maduravoyal                                                                      |69   |
|Tiruvottiyur                                                                     |64   |
|Nanganallur                                                                      |63   |
|George Town                                                                      |61   |
|Aminijikarai                                                                     |59   |
|Karapakkam                                                                       |59   |
|Thiruvallur                                                                      |54   |
|Saligramam                                                                       |54   |
|Teynampet                                                                        |53   |
|Vepery                                                                           |53   |
|West Mambalam                                                                    |52   |
|Vandalur                                                                         |52   |
|Mahabalipuram                                                                    |51   |
|Virugambakkam                                                                    |50   |
|Madhavaram                                                                       |49   |
|Arumbakkam                                                                       |47   |
|Padur                                                                            |45   |
|Saidapet                                                                         |42   |
|Thousand Lights                                                                  |42   |
|Royapuram                                                                        |42   |
|Sowcarpet                                                                        |41   |
|Red Hills                                                                        |40   |
|Old Mahabalipuram Road (OMR)                                                     |39   |
|Egatoor                                                                          |38   |
|Gopalapuram                                                                      |36   |
|Kelambakkam                                                                      |36   |
|Park Town                                                                        |35   |
|Ekkaduthangal                                                                    |34   |
|RA Puram                                                                         |34   |
|Koyambedu                                                                        |33   |
|Palavakkam                                                                       |27   |
|Chetpet                                                                          |27   |
|Guindy                                                                           |26   |
|Anna Salai                                                                       |25   |
|Parrys                                                                           |24   |
|Alandur                                                                          |24   |
|OMR Food Street, Navallur                                                        |24   |
|Neelangarai                                                                      |23   |
|Shenoy Nagar                                                                     |23   |
|Nandanam                                                                         |23   |
|Kottivakkam                                                                      |22   |
|Kanathur                                                                         |22   |
|Phoenix Market City, Velachery                                                   |22   |
|Kotturpuram                                                                      |21   |
|Injambakkam                                                                      |21   |
|Ascendas IT Park, Taramani                                                       |21   |
|St. Thomas Mount                                                                 |20   |
|Mandaveli                                                                        |18   |
|VR Mall, Anna Nagar                                                              |17   |
|Semmancheri                                                                      |16   |
|OMR Food Street, Kandanchavadi                                                   |15   |
|null                                                                             |12   |
|Mahindra World City, Chengalpattu                                                |12   |
|Santhome                                                                         |12   |
|East Coast Road (ECR)                                                            |11   |
|Express Avenue Mall,Royapettah                                                   |11   |
|OMR Food Street, Injambakkam                                                     |11   |
|OMR Food Street, Ambattur                                                        |10   |
|Kora Food Street, Anna Nagar West                                                |10   |
|Forum Vijaya Mall, Vadapalani                                                    |10   |
|Okkiyampet                                                                       |10   |
|OMR Food Street, Guduvancheri                                                    |10   |
|ITC Grand Chola, Guindy                                                          |10   |
|Akkarai                                                                          |10   |
|Muttukadu                                                                        |9    |
|Crowne Plaza Chennai Adyar Park, Alwarpet                                        |8    |
|Kovalam                                                                          |7    |
|The Leela Palace, MRC Nagar                                                      |7    |
|The Savera Hotel, RK Salai (Cathedral Road)                                      |7    |
|Hyatt Regency, Teynampet                                                         |7    |
|Abhiramapuram                                                                    |6    |
|Ambit IT Park, Ambattur                                                          |6    |
|The Park, Nungambakkam                                                           |6    |
|The Raintree, Teynampet                                                          |6    |
|MRC Nagar                                                                        |6    |
|Ramada, Egmore                                                                   |6    |
|Vettuvankeni                                                                     |6    |
|Hotel Radha Regent, Arumbakkam                                                   |5    |
|Hilton Chennai, Guindy                                                           |5    |
|SKLS Galaxy Mall, Redhills                                                       |5    |
|The Residency Towers, T. Nagar                                                   |5    |
|Taj Coromandel, Nungambakkam                                                     |5    |
|Taj Club House, Thousand Lights                                                  |5    |
|Le Royal Meridien, St. Thomas Mount                                              |5    |
|Chennai Citi Centre, Mylapore                                                    |5    |
|Fortune Select Grand, Chengalpattu                                               |5    |
|The Raintree, Alwarpet                                                           |4    |
|Mayajaal Multiplex, Kanathur                                                     |4    |
|Hotel Vassi Palaze, Kanchipuram District                                         |4    |
|Sathyam Cinemas Complex, Royapettah                                              |4    |
|Green Park Hotel, Vadapalani                                                     |4    |
|Dash@OMR, Old Mahabalipuram Road (OMR)                                           |4    |
|WelcomHotel, RK Salai (Cathedral Road)                                           |4    |
|Ramada Plaza, Guindy                                                             |4    |
|The Residency, T. Nagar                                                          |4    |
|The Accord Metropolitan, T. Nagar                                                |4    |
|Chennai Food Town, Thuraipakkam                                                  |4    |
|Radisson Blu, GST Road                                                           |4    |
|Feathers, A Radha Hotel                                                          |4    |
|Novotel Chennai Sipcot                                                           |4    |
|Ispahani Centre, Nungambakkam                                                    |4    |
|Kipling, East Coast Road (ECR)                                                   |4    |
|Radisson Blu Temple Bay, Mamallapuram                                            |4    |
|Turyaa Chennai                                                                   |4    |
|Sheraton Grand, Neelangarai                                                      |3    |
|InterContinental Chennai Mahabalipuram Resort, East...                           |3    |
|Radisson Blu, Egmore                                                             |3    |
|Trident, GST Road                                                                |3    |
|Citadines                                                                        |3    |
|Holiday Inn Chennai OMR IT Expressway                                            |3    |
|The Westin Chennai, Velachery                                                    |3    |
|JP Hotel, Koyambedu                                                              |3    |
|Meenambakkam                                                                     |3    |
|Hablis Hotel, Guindy                                                             |3    |
|Paati Veedu, T.Nagar                                                             |3    |
|Foodies Kitchen                                                                  |3    |
|Grand Residence Hotel, Porur                                                     |3    |
|Novotel Chennai, Nandanam                                                        |3    |
|Ambassador Pallava, Egmore                                                       |3    |
|Hotel Centre Point, Sholinganallur                                               |3    |
|Hotel Ranjith, Nungambakkam                                                      |3    |
|Zone by The Park, Pallikaranai                                                   |3    |
|BKR Grand Hotel, T. Nagar                                                        |3    |
|Vestin Park Hotel, Egmore                                                        |3    |
|Ideal Beach Resort, East Coast Road (ECR)                                        |3    |
|New Woodlands Hotel, Mylapore                                                    |3    |
|The Spring Hotel, Nungambakkam                                                   |3    |
|Taj Fisherman's Cove Resort & Spa, Kanchipuram District                          |3    |
|Hotel Rajpark, Alwarpet                                                          |3    |
|Hotel Manhattan, Mylapore                                                        |3    |
|Gokulam Park Hotel, Ashok Nagar                                                  |3    |
|Grand by GRT Hotels                                                              |3    |
|Gokulam Park Sabari - OMR, Old Mahabalipuram Road                                |3    |
|Hotel Park Elanza, Nungambakkam                                                  |3    |
|Vivanta Chennai, IT Expressway, Sholinganallur                                   |3    |
|Hotel Joyland, Chengalpattu                                                      |3    |
|Clarion Hotel, Mylapore                                                          |3    |
|Ambica Empire, Vadapalani                                                        |3    |
|Ranga Residency, Chengalpattu                                                    |3    |
|The Checkers Hotel, Saidapet                                                     |2    |
|Hotel Pratap Plaza, Kodambakkam                                                  |2    |
|Hotel Shelter, Mylapore                                                          |2    |
|Lemon Tree Shimona, Ramapuram                                                    |2    |
|Park Hyatt, Guindy                                                               |2    |
|Courtyard by Marriott, Teynampet                                                 |2    |
|Chariot Beach Resort, East Coast Road (ECR)                                      |2    |
|Harrisons Hotel, Nungambakkam                                                    |2    |
|Hotel Maris, Gopalapuram                                                         |2    |
|Hotel Mamallaa Heritage, East Coast Road (ECR)                                   |2    |
|Hotel Anitha Towers, Triplicane                                                  |2    |
|The Grand Mall, Velachery                                                        |2    |
|Hotel Fortel, Egmore                                                             |2    |
|Nayagara Hotels, Kodambakkam                                                     |2    |
|The King's Hotel, Egmore                                                         |2    |
|Novotel Chennai, OMR                                                             |2    |
|V7 Hotel                                                                         |2    |
|Spencer Plaza Mall, Thousand Lights                                              |2    |
|Mercure, Sriperumbudur                                                           |2    |
|Ponnis Grand Inn Hotel, Thiruvallur                                              |2    |
|OMR Food Street, Perumbakkam                                                     |2    |
|Hotel Chennai Deluxe, Koyambedu                                                  |2    |
|Benzz Park, T. Nagar                                                             |2    |
|Somerset Greenways                                                               |2    |
|Eldoris                                                                          |2    |
|Fairfield by Marriot, Chengalpattu                                               |2    |
|Hotel Palmgrove, Nungambakkam                                                    |2    |
|Hotel Vee Yes, Egmore                                                            |2    |
|Hotel Marina Inn, Egmore                                                         |2    |
|Hotel Goutham Manor, Nungambakkam                                                |2    |
|Four Points by Sheraton, East Coast Road (ECR)                                   |2    |
|Hotel Raj Palace, T. Nagar                                                       |2    |
|Saaral Residency, Mogappair                                                      |2    |
|Massbunk Complex, Purasavakkam                                                   |2    |
|E Hotel, Royapettah                                                              |2    |
|The Slate                                                                        |2    |
|Golden Sun Resort, Mamallapuram                                                  |2    |
|Hotel Sathyam Grand Resort, Kanchipuram District                                 |2    |
|Oragadam                                                                         |2    |
|Lemon Tree Hotel, Guindy                                                         |2    |
|Park Plaza OMR, Thuraipakkam                                                     |2    |
|Ampa Skywalk Mall, Aminijikarai                                                  |2    |
|Hotel SRR Grand, Vandalur                                                        |2    |
|The Hotel Royal Plaza, Koyambedu                                                 |2    |
|Keys Hotel, Thiruvanmiyur                                                        |2    |
|Hotel Milestonnez, Kanchipuram District                                          |2    |
|Grand Galada Mall, Meenambakkam                                                  |2    |
|La Woods Hotel, Thousand Lights                                                  |2    |
|Hotel Sudhara, T. Nagar                                                          |2    |
|Mamalla Beach Resort, Mahabalipuram                                              |2    |
|Aloft Chennai, Sholinganallur                                                    |2    |
|GRT Regency, Kanchipuram                                                         |2    |
|Deccan Plaza, Royapettah                                                         |2    |
|Hotel Beverly, Kilpauk                                                           |2    |
|Green Meadows Resorts, Palavakkam                                                |2    |
|Quality Inn Sabari, T. Nagar                                                     |2    |
|Hotel Bhimaas, Vadapalani                                                        |1    |
|Hotel Aadithya, Vadapalani                                                       |1    |
|Jade Resorts, East Coast Road (ECR)                                              |1    |
|Hotel Blue Diamond, Kilpauk                                                      |1    |
|Hotel Sky Park, Anna Nagar West                                                  |1    |
|Hotel Royal Regency, Vepery                                                      |1    |
|Hotel Maurya International, Vadapalani                                           |1    |
|Biryani"                                                                         |1    |
|Hotel Abu Palace, Egmore                                                         |1    |
|Liberty Park Hotel, Kodambakkam                                                  |1    |
|Hotel Peninsula, T. Nagar                                                        |1    |
|Tea,Baklava Cake,Fish,Pepper Chicken,Prawn,Biryani,Saffron Rice                  |1    |
|Spectrum The Grand Venus Mall, Perambur                                          |1    |
|Biryani,Momos,Tandoori Chicken,Fried Rice Chicken,Prawn,Manchurian,Afghan Chicken|1    |
|IBIS Chennai, OMR                                                                |1    |
|Green Coconut Resort, Kanchipuram District                                       |1    |
|Eat And Pack, Ambattur                                                           |1    |
|TNHB Complex, Adyar                                                              |1    |
|Days Hotel, Old Mahabalipuram Road (OMR)                                         |1    |
|Hotel Blue Nile, Pallavaram                                                      |1    |
|Grande Bay Resort                                                                |1    |
|IBIS Hotel, Old Mahabalipuram Road (OMR)                                         |1    |
|Hotel Poigai, Arumbakkam                                                         |1    |
|The Pride Hotel, Kilpauk                                                         |1    |
|Abu Sarovar Portico, Egmore                                                      |1    |
|Invalid                                                                          |1    |
|Hotel Victoria, Egmore                                                           |1    |
|OMR Food Street,Thuraipakkam                                                     |1    |
|Hotel NRS Sakithyan, T. Nagar                                                    |1    |
|Southern Residency Hotel, OMR                                                    |1    |
|TNHB Complex, Besant Nagar                                                       |1    |
|The Vijay Park Hotel, Arumbakkam                                                 |1    |
|Velachery"                                                                       |1    |
|Hotel Mount Heera, Alandur                                                       |1    |
|GLM Meridian Hotel, T. Nagar                                                     |1    |
+---------------------------------------------------------------------------------+-----+
