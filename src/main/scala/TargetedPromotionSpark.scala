import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

object TargetedPromotionSpark {

  val LOG_LEVEL = "ERROR"
  val NUMBER_ITERATIONS: Int = 50
  var TOP_K_ITEMSETS: Int = 8
  val K = 4

  def itemsetToSparseVector(itemset: FPGrowth.FreqItemset[String],
                            size: Int): SparseVector = {
    val indices = itemset.items.map(r => r.toInt - 1).sorted
    val values = Array.fill(indices.length)(1.toDouble)

    new SparseVector(size, indices, values)
  }

  //TODO check itemset -1
  def isItemsetSubset(transaction: SparseVector,
                      itemset: FreqItemset[String]) = {
    val ids = itemset.items.map(_.toInt - 1).toSet

    transaction.indices.toSet.intersect(ids).size == ids.size
  }

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

    // (T,((f(i), Profit(T)))
    val transactionUnitRDD = raw_transactionsRDD
      .map(r => {
        var categoriesSales = r._2
          .flatMap(
            s =>
              bproductMap
                .value(s._1.toString)
                ._1
                .split("-")
                .map(id => (id.toInt, s._2.toDouble)))
          .groupBy(_._1)
          .map(r => (r._1 - 1, r._2.map(_._2).sum)) // -1 different indices in files and runtime
          .toArray
          .sortBy(_._1)

        // profit T computation
        val profitMargin = r._2
          .map(product => {
            val productInfo = productsMap(product._1.toString)

            (productInfo._2 - productInfo._3) * product._2
          }).sum


        (r._1,
         (new SparseVector(bcategoriesSize.value,
                           categoriesSales.map(_._1),
                           categoriesSales.map(_._2)),
          profitMargin))
      })
      .persist()

    val transactionsRDD = transactionUnitRDD.map(
      transaction =>
        (transaction._1,
         new SparseVector(
           transaction._2._1.size,
           transaction._2._1.indices,
           Array.fill(transaction._2._1.indices.length)(1.toDouble))))

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

    var allFreqItemset = new Array[Array[FPGrowth.FreqItemset[String]]](K)

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
        .filter(_._1.items.length > 1)
        .sortBy(-_._2)
        .take(TOP_K_ITEMSETS)

      allFreqItemset(clusterID) = allConfItemset.map(_._1)
    }

    val bAllFreqItemset = sc.broadcast(allFreqItemset)

    val includedCategories = allFreqItemset.flatten
      .map(r => itemsetToSparseVector(r, categoriesMap.size))
      .reduce((acc, y) => {
        val accSet = acc.indices.toSet
        val ySet = y.indices.toSet
        val indices = accSet.union(ySet).toArray.sorted

        new SparseVector(acc.size,
                         indices,
                         Array.fill(indices.length)(1.toDouble))
      })

    val bIncludedCategories = sc.broadcast(includedCategories.indices.toSet)

    //V(a) computation
    val grossSalesMargin = transactionUnitRDD
      .filter(transaction => {
        val indices = transaction._2._1.indices.toSet

        bIncludedCategories.value.intersect(indices).size > 1
      })
      .map(transaction => {

        val freqItemsetSizes = bAllFreqItemset.value.map(_.length)
        val partialVA =
          Array.fill(bAllFreqItemset.value.flatten.length)(0.toDouble)

        var clusterID = 0
        for (clusterID <- 0 until K) {
          var clusterFreqItemsets = bAllFreqItemset.value(clusterID)

          var itemsetList =
            new ListBuffer[(FPGrowth.FreqItemset[String], Int)]()
          var freqSum = 0.0
          for (i <- clusterFreqItemsets.indices) {
            if (isItemsetSubset(transaction._2._1, clusterFreqItemsets(i))) {
              freqSum += clusterFreqItemsets(i).freq
              itemsetList += ((clusterFreqItemsets(i), i))

            }

          }

          //weight computation
          for (values <- itemsetList) {
            var index = freqItemsetSizes.slice(0, clusterID).sum + values._2
            partialVA(index) = values._1.freq / freqSum * transaction._2._2

          }

        }

        partialVA

      }).reduce( (acc, pva) => acc.zip(pva).map { case (x, y) => x + y } )

    x1 + 23 +

    grossSalesMargin.foreach(println)

  }

}
