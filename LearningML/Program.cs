using System;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace LearningML
{
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            var dataPath = "iris.data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.Predict(new IrisData
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
