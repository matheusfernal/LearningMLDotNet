using System;
using Microsoft.ML.Runtime.Api;

namespace LearningML
{
    // IrisPrediction is the result returned from prediction operations
    public class IrisPrediction
    {
        [ColumnName("PredictionLabels")]
        public string PredictedLabels { get; set; }
    }
}
