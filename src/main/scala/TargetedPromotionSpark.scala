import org.apache.spark.sql.SparkSession

object TargetedPromotionSpark {

  def main(args: Array[String]): Unit = {
    val ss = SparkSession.builder().master("local").appName("TargetedPromotionSpark").getOrCreate()
  }

}