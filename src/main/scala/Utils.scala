import org.apache.spark.ml.linalg.SparseVector
import scala.util.control.Breaks.{break, breakable}
import breeze.linalg.{SparseVector => SP}

object Utils {

  /**
    * Given two sparse vectors c,q compute
    * dot product in O( c.size + q.size )
    *
    * @param c
    * @param q
    * @return
    */
  private def dot(c: SparseVector, q: SparseVector): Double = {
    if (c.indices.size != 0 && q.indices.size != 0) {

      var dot_product = 0.0

      var c_pos = 0
      var q_pos = 0

      var c_index = c.indices(c_pos)
      var q_index = q.indices(q_pos)

      var c_value = c.values(c_pos)
      var q_value = q.values(q_pos)

      breakable {
        while (c_pos <= c.indices.size && q_pos <= q.indices.size) {

          if (c_index < q_index) {
            c_pos += 1

          } else if (c_index > q_index) {
            q_pos += 1

          } else {
            dot_product += q_value * c_value

            c_pos += 1
            q_pos += 1
          }

          try {
            c_index = c.indices(c_pos)
            q_index = q.indices(q_pos)

            c_value = c.values(c_pos)
            q_value = q.values(q_pos)
          } catch {
            case e: ArrayIndexOutOfBoundsException => break
          }
        }

      }

      dot_product

    } else {
      0
    }
  }

  /**
    * Given two sparse vectors compute tanimoto similarity
    *
    * @param c
    * @param q
    * @return
    */
  def tanimoto(c: SparseVector, q: SparseVector): Double = {
    // find dot product
    var cq_dot = dot(c, q)

    // find magnitube
    var c_magnitube = c.values.map(c => Math.pow(c, 2)).sum
    var q_magnitube = q.values.map(q => Math.pow(q, 2)).sum

    // return tanimoto similarity
    cq_dot / (c_magnitube + q_magnitube - cq_dot)
  }

  /**
    * Given two sparse vectors compute addition and return SparseVector
    *
    * @param sp1
    * @param sp2
    * @return
    */
  def addSpVectors(sp1: SparseVector, sp2: SparseVector): SparseVector = {
    val x = new SP(sp1.indices, sp1.values, sp1.size)
    val y = new SP(sp2.indices, sp2.values, sp2.size)

    val out = (x + y).asInstanceOf[SP[Double]]

    new SparseVector(out.length, out.index, out.data)
  }

  /**
    * Divide sparse vector values with given value
    *
    * @param sp
    * @param value
    * @return
    */
  def divSpVector(sp: SparseVector, value: Int): SparseVector = {
    val sp_vals = sp.values.map(r => r / value)

    new SparseVector(sp.size, sp.indices, sp_vals)
  }

  def divSpVector(sp: SparseVector, value: Double): SparseVector = {
    val sp_vals = sp.values.map(r => r / value)

    new SparseVector(sp.size, sp.indices, sp_vals)
  }

  def addArrays(arr1: Array[Double], arr2: Array[Double]): Array[Double] = {
    arr1.zip(arr2).map { case (x, y) => x + y }
  }

}
