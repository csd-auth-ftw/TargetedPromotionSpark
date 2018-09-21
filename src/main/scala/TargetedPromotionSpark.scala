import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import scala.collection.mutable

object TargetedPromotionSpark {

  def main(args: Array[String]): Unit = {

    val transactionsPath = "./data/sales_fact_1997.csv"
    val productsPath = "./data/chinese_products.csv"
    val categoriesPath = "./data/chinese_categories.csv"

    val ss = SparkSession.builder().master("local").appName("TargetedPromotionSpark").getOrCreate()

    import ss.implicits._

    val categoriesDF = ss.read.option("header", "true").csv(categoriesPath)
    val productsDF = ss.read.option("header", "true").csv(productsPath)
    val salesDF = ss.read.option("header", "true").csv(transactionsPath).
      drop("promotion_id", "store_id", "store_sales", "store_cost", "unit_sales")

    val productsMap = productsDF.collect().map(r => (r.getString(0), r.getString(1))).toMap
    val bcategoriesSize = ss.sparkContext.broadcast(categoriesDF.collect().size)
    val bproductMap = ss.sparkContext.broadcast(productsMap)

    //transactions
    val transactionsRDD = salesDF.map(r => (r.getString(1) + ":" + r.getString(2), r.getString(0)))
      .rdd.reduceByKey((x, y) => x.toString + "-" + y.toString)


    transactionsRDD.map(r => {
      var line = r._2.split("-")
      var categories_id = line.map(s => bproductMap.value.get(s).get)

      var indicies = categories_id.flatMap(r => r.split('-')).distinct.map(_.toInt).map(_-1).sorted
      var values = Array.fill(indicies.size)(1.toDouble)

      (r._1, new SparseVector(bcategoriesSize.value, indicies, values))
    }).collect().foreach(println)


  }

}