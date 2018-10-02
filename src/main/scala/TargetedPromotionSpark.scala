import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import scala.collection.mutable

object TargetedPromotionSpark {


  val NUMBER_ITERATIONS: Int = 50
  val K = 1

  def main(args: Array[String]): Unit = {


    val transactionsPath = "./data/sales_fact_1997.csv"
    val productsPath = "./data/chinese_products.csv"
    val categoriesPath = "./data/chinese_categories.csv"

    val ss = SparkSession.builder().master("local").appName("TargetedPromotionSpark").getOrCreate()
    val sc = ss.sparkContext

    import ss.implicits._

    val categoriesMap = ss.read.option("header", "true").csv(categoriesPath).collect().map(r => (r.getString(0), r.getString(1))).toMap
    val productsDF = ss.read.option("header", "true").csv(productsPath)
    val salesDF = ss.read.option("header", "true").csv(transactionsPath).
      drop("promotion_id", "store_id", "store_sales", "store_cost", "unit_sales")

    val productsMap = productsDF.collect().map(r => (r.getString(0), r.getString(1))).toMap
    val bcategoriesSize = ss.sparkContext.broadcast(categoriesMap.size)
    val bproductMap = sc.broadcast(productsMap)

    //transactions
    val raw_transactionsRDD = salesDF.map(r => (r.getString(1) + ":" + r.getString(2), r.getString(0)))
      .rdd.reduceByKey((x, y) => x.toString + "-" + y.toString)

    val transactionsRDD = raw_transactionsRDD.map(r => {
      var line = r._2.split("-")
      var categories_id = line.map(s => bproductMap.value.get(s).get)

      var indicies = categories_id.flatMap(r => r.split('-')).distinct.map(_.toInt).map(_ - 1).sorted
      var values = Array.fill(indicies.size)(1.toDouble)

      (r._1, new SparseVector(bcategoriesSize.value, indicies, values))
    })

    var bcenters = sc.broadcast(transactionsRDD.takeSample(false, K).zipWithIndex.map(r => (r._2, r._1._2)).toMap)

    val initClustersRDD = transactionsRDD.map(r => {
      val min_center = bcenters.value.map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
        .minBy(_._2)

      (min_center._1, r)
    })

    val customerMeanTransactionsRDD = transactionsRDD.map(r => (r._1.split(':')(1), (r._2, 1))).reduceByKey((x, y) => (Utils.addSpVectors(x._1, y._1), x._2 + y._2))
      .map(r => (r._1, Utils.divSpVector(r._2._1, r._2._2))).persist()


    var centersInfo = initClustersRDD.map(r => (r._1, r._2._2))
      .reduceByKey((x, y) => Utils.addSpVectors(x, y))
      .map(r => (r._1, Utils.divSpVector(r._2, r._2.values.max.toInt)))
      .collect().toMap

    bcenters.destroy()
    bcenters = sc.broadcast(centersInfo)

    //TODO
    var i = 0
    while (i < NUMBER_ITERATIONS) {
      centersInfo = customerMeanTransactionsRDD.map(r => {
        val min_center = bcenters.value.map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
          .minBy(_._2)

        (min_center._1, r)
      }).map(r => (r._1, r._2._2))
        .reduceByKey((x, y) => Utils.addSpVectors(x, y))
        .map(r => (r._1, Utils.divSpVector(r._2, r._2.values.max)))
        .collect().toMap

      bcenters.destroy()
      bcenters = sc.broadcast(centersInfo)

      i += 1
    }

    val customerClusterMap = customerMeanTransactionsRDD.map(r => {
      val min_center = bcenters.value.map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
        .minBy(_._2)

      (r._1, min_center._1)
    }).collect().toMap
    val bcustomerClusterMap = sc.broadcast(customerClusterMap)

    val transactionsFPG = transactionsRDD.map(r => {
      var clusterID = bcustomerClusterMap.value.get(r._1.split(':')(1)).get
      var transactions = Array(r._2.indices.map(r => r.toString))

      (clusterID, transactions)
    }).reduceByKey((x, y) => x ++ y)
      .persist()


    val minConfidence = 0.6
    var clusterID = 0
    for (clusterID <- 0 to K - 1) {

      val clusterTransactionsRDD = transactionsFPG.filter(r => r._1 == clusterID).flatMap(r => r._2)

      val fpg = new FPGrowth()
        .setMinSupport(0.1)
        .setNumPartitions(10)

      val model = fpg.run(clusterTransactionsRDD)

      model.freqItemsets.collect().foreach { itemset =>
        var catItems = itemset.items.map(r => categoriesMap.get((r.toInt + 1).toString).get)
        println(s"${catItems.mkString("[", ",", "]")},${itemset.freq}")
      }

      model.generateAssociationRules(minConfidence).collect().foreach { rule =>
        var anteCatItems = rule.antecedent.map(r => categoriesMap.get((r.toInt + 1).toString).get)
        var conseCatItems = rule.consequent.map(r => categoriesMap.get((r.toInt + 1).toString).get)

        println(s"${anteCatItems.mkString("[", ",", "]")}=> " +
          s"${conseCatItems.mkString("[", ",", "]")},${rule.confidence}")
      }

      println("END OF FPGROWTH")
      println("cluster id = " + clusterID)
      println("count = " + clusterTransactionsRDD.count())

    }

  }

}