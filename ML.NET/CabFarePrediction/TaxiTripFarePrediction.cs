using Microsoft.ML.Data;

namespace CabFarePrediction
{
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
