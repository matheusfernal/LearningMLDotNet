using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;

namespace LearningML
{
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 2: Create an environment and load your data
            var env = new LocalEnvironment();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            var dataPath = "iris.data.txt";
            var reader = new TextLoader(env,
                new TextLoader.Arguments 
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("SepalLength", DataKind.R4, 0),
                        new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                        new TextLoader.Column("PetalLength", DataKind.R4, 2),
                        new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                        new TextLoader.Column("Label", DataKind.Text, 4)
                    }
                });

            var trainingDataView = reader.Read(new MultiFileSource(dataPath));

            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = new TermEstimator(env, "Label", "Label")
                .Append(new ConcatEstimator(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments()))
                .Append(new KeyToValueEstimator(env, "PredictedLabel"));

            // STEP 4: Train your model based on the data set
            var model = pipeline.Fit(trainingDataView);

            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.MakePredictionFunction<IrisData, IrisPrediction>(env).Predict(new IrisData
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }
    }
}
