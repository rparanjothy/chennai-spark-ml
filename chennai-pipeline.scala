// get your pipes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler  
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StandardScaler

// define your transforms and estimators
// rename name and price

val chennai=spark.read.format("csv").option("inferSchema","true").option("header","true").load("./data.csv")

val x=chennai.drop($"Zomato URL").withColumnRenamed("Price for 2","p").withColumnRenamed("Name of Restaurant","name").withColumnRenamed("Ratings","r")          
val x1=x.select('name,'Address,'Location,'p.cast("double"),'r.cast("double"))        
val x2=x1.na.drop("any",Seq("p","r"))      

// clean location
val x3=x2.select(trim('name).as("n"),trim('Address).as("a"),trim('Location).as("l"),'p,'r)
// l to index
val locationIDX = new StringIndexer().setInputCol("l").setOutputCol("lIndex")  
// assemble
val va=new VectorAssembler().setInputCols(Array("lIndex","p","r")).setOutputCol("features") 
// standardize-test
val stdz=new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
// algo
val k=new KMeans().setK(500).setFeaturesCol("features")
// withSTD

val k=new KMeans().setK(500).setFeaturesCol("scaledFeatures")

// build a pipeline
val pipe=new Pipeline().setStages(Array(locationIDX,va,stdz,k))    
// save the pipeline
pipe.write.overwrite().save("./pipeline/chennai-food")
// load pipe
val pipeLoaded=Pipeline.load("./pipeline/chennai-food")

// get the model
val model=pipe.fit(x3)
// save the model
model.write.overwrite().save("./model/chennai-food")
// get the model
val modelLoaded=PipelineModel.load("./model/chennai-food")
// predicts..
val xr=model.transform(x3)

def showRest(p:Int,x:org.apache.spark.sql.DataFrame)=x.filter('prediction===p).orderBy('Ratings.desc).show(500,false) 

def showClusters(r:org.apache.spark.sql.DataFrame)=r.groupBy('prediction,'p,'r,'l).count.orderBy('count.desc).show(200,false)    
def showClustersByP(r:org.apache.spark.sql.DataFrame)=r.groupBy('prediction,'p,'r,'l).count.orderBy('prediction).show(100,false)    


def buildTrain(ct:Int,data:org.apache.spark.sql.DataFrame):org.apache.spark.sql.DataFrame={
        // l to index
    val locationIDX = new StringIndexer().setInputCol("l").setOutputCol("lIndex")  
    // assemble
    val va=new VectorAssembler().setInputCols(Array("lIndex","p","r")).setOutputCol("features") 
    
    // standardize-test
    val stdz=new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

    // algo
// val k=new KMeans().setK(ct).setFeaturesCol("features")
    val k=new KMeans().setK(ct).setFeaturesCol("scaledFeatures")

    // build a pipeline
    // val pipe=new Pipeline().setStages(Array(locationIDX,va,k))
    val pipe=new Pipeline().setStages(Array(locationIDX,va,stdz,k))
    val model=pipe.fit(x3)
    val xr=model.transform(x3)
    xr
}

val xr500=buildTrain(500,x3)
val xr400=buildTrain(400,x3)
val xr1500=buildTrain(1500,x3)
val xr3000=buildTrain(3000,x3)


xr1500.select('prediction).distinct.count

showClusters(xr3000)
showClustersByP(xr3000)

showClustersByP(buildTrain(50,x3))
showClustersByP(buildTrain(150,x3))

showRest(19,xr3000)
