import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.sql.SparkSession

object TargetedPromotionSpark {

  val LOG_LEVEL = "ERROR"
  val NUMBER_ITERATIONS: Int = 50
  val K = 4

  def main(args: Array[String]): Unit = {

    val transactionsPath = "./data/sales_fact_1997.csv"
    val productsPath = "./data/foodmart_products.csv"
    val categoriesPath = "./data/foodmart_categories.csv"

    val ss = SparkSession
      .builder()
      .master("local")
      .appName("TargetedPromotionSpark")
      .getOrCreate()
    val sc = ss.sparkContext
    sc.setLogLevel(LOG_LEVEL)

    import ss.implicits._

    val categoriesMap = ss.read
      .option("header", "true")
      .csv(categoriesPath)
      .collect()
      .map(r => (r.getString(0), r.getString(1)))
      .toMap
    val productsDF = ss.read.option("header", "true").csv(productsPath)
    val salesDF = ss.read
      .option("header", "true")
      .csv(transactionsPath)
      .drop("promotion_id", "store_id", "store_sales", "store_cost")

    val productsMap = productsDF
      .collect()
      .map(
        r =>
          (r.getString(0),
           (r.getString(1),
            r.getString(2).toDouble,
            r.getString(3).toDouble,
            r.getString(4).toDouble)))
      .toMap
    val bcategoriesSize = ss.sparkContext.broadcast(categoriesMap.size)
    val bproductMap = sc.broadcast(productsMap)

    //transactions
    val raw_transactionsRDD = salesDF
      .map(
        r =>
          (r.getString(1) + ":" + r.getString(2),
           r.getString(0) + "-" + r.getString(3)))
      .rdd
      .map(x => {
        var tokens = x._2.split("-")
        (x._1, Array((tokens(0).toDouble.toInt, tokens(1).toDouble.toInt)))
      })
      .reduceByKey((acc, y) => acc ++ y)

    val transactionUnitRDD = raw_transactionsRDD
      .map(r => {

        var categoriesSales = r._2
          .flatMap(
            s =>
              bproductMap.value
                .get(s._1.toString)
                .get
                ._1
                .split("-")
                .map(id => (id.toInt, s._2.toDouble)))
          .groupBy(_._1)
          .map(r => (r._1 - 1, r._2.map(_._2).sum)) // -1 different indices in files and runtime
          .toArray
          .sortBy(_._1)

        (r._1,
         new SparseVector(bcategoriesSize.value,
                          categoriesSales.map(_._1),
                          categoriesSales.map(_._2)))
      })
      .persist()

    val transactionsRDD = transactionUnitRDD.map(
      transaction =>
        (transaction._1,
         new SparseVector(transaction._2.size, transaction._2.indices, Array.fill(transaction._2.indices.length)(1.toDouble))))

    var bcenters = sc.broadcast(
      transactionsRDD
        .takeSample(false, K)
        .zipWithIndex
        .map(r => (r._2, r._1._2))
        .toMap)

    val initClustersRDD = transactionsRDD.map(r => {
      val min_center = bcenters.value
        .map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
        .minBy(_._2)

      (min_center._1, r)
    })

    val customerMeanTransactionsRDD = transactionsRDD
      .map(r => (r._1.split(':')(1), (r._2, 1)))
      .reduceByKey((x, y) => (Utils.addSpVectors(x._1, y._1), x._2 + y._2))
      .map(r => (r._1, Utils.divSpVector(r._2._1, r._2._2)))
      .persist()

    var centersInfo = initClustersRDD
      .map(r => (r._1, r._2._2))
      .reduceByKey((x, y) => Utils.addSpVectors(x, y))
      .map(r => (r._1, Utils.divSpVector(r._2, r._2.values.max.toInt)))
      .collect()
      .toMap

    bcenters.destroy()
    bcenters = sc.broadcast(centersInfo)

    //TODO
    var i = 0
    while (i < NUMBER_ITERATIONS) {
      centersInfo = customerMeanTransactionsRDD
        .map(r => {
          val min_center = bcenters.value
            .map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
            .minBy(_._2)

          (min_center._1, r)
        })
        .map(r => (r._1, r._2._2))
        .reduceByKey((x, y) => Utils.addSpVectors(x, y))
        .map(r => (r._1, Utils.divSpVector(r._2, r._2.values.max)))
        .collect()
        .toMap

      bcenters.destroy()
      bcenters = sc.broadcast(centersInfo)

      i += 1
    }

    val customerClusterMap = customerMeanTransactionsRDD
      .map(r => {
        val min_center = bcenters.value
          .map(c => (c._1, 1 - Utils.tanimoto(c._2, r._2)))
          .minBy(_._2)

        (r._1, min_center._1)
      })
      .collect()
      .toMap
    val bcustomerClusterMap = sc.broadcast(customerClusterMap)

    val transactionsFPG = transactionsRDD
      .map(r => {
        var clusterID = bcustomerClusterMap.value.get(r._1.split(':')(1)).get
        var transactions = Array(r._2.indices.map(r => r.toString))

        (clusterID, transactions)
      })
      .reduceByKey((x, y) => x ++ y)
      .persist()

    //Gross margin for each transaction m(t)

    val minConfidence = 0.6
    var clusterID = 0
    for (clusterID <- 0 to K - 1) {

      val clusterTransactionsRDD = transactionsFPG
        .filter(r => r._1 == clusterID)
        .flatMap(r => r._2)
        .persist()

      val fpg = new FPGrowth()
        .setMinSupport(0.02)
        .setNumPartitions(10)

      val model = fpg.run(clusterTransactionsRDD)
      val itemsIndex = scala.collection.mutable.Set[Int]()
      val itemsets = model.freqItemsets.collect()
      itemsets.foreach { itemset =>
        itemset.items.foreach(index => itemsIndex.add(index.toInt))
      }

      val freqClusterTransactionsRDD =
        clusterTransactionsRDD.filter(transaction => {
          transaction
            .map(r => r.toInt)
            .toSet[Int]
            .intersect(itemsIndex)
            .size > 1
        })

      val model2 = fpg.setMinSupport(0).run(freqClusterTransactionsRDD)
      val rulesMap: collection.mutable.Map[String, Double] =
        collection.mutable.Map()

      model2.generateAssociationRules(0).collect().foreach { rule =>
        // TODO make key function
        val ante = rule.antecedent.map(item => item.toInt).sorted.mkString(",")
        val cons = rule.consequent.map(item => item.toInt).sorted.mkString(",")
        val key = ante + "::" + cons

        rulesMap.update(key, rule.confidence)
      }

      val allConfItemset = itemsets
        .map(itemset => {
          val items = itemset.items.map(item => item.toInt)

          var allConf = 0.0
          if (items.size > 1) {
            val subsets = items.toSet[Int].subsets.map(_.toList).toList
            allConf = 1
            subsets.foreach(antecedent => {
              val consequent = items.diff(antecedent)

              if (antecedent.size > 0 && consequent.size > 0) {
                val ante = antecedent.sorted.mkString(",")
                val cons = consequent.sorted.mkString(",")
                val key = ante + "::" + cons
                val conf = rulesMap.get(key).get

                if (conf < allConf)
                  allConf = conf
              }

            })
          }
          (itemset, allConf)
        })
        .sortBy(-_._2)
        .filter(_._1.items.length > 1)
        .take(8)

    }

  }

}
