using Microsoft.ML;
using System;
using System.IO;

namespace CabFarePrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext();
            var model = Train(mLContext, _trainDataPath);
            Evaluate(mLContext, model);
            TestSinglePrediction(mLContext, model);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        /// <summary>
        /// Creates a single comment of test data.
        /// Predicts fare amount based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 600,
                TripDistance = 1.52f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 8.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 8.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}