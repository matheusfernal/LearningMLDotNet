using System;
using Microsoft.ML.Runtime.Api;

namespace LearningML
{
    // STEP 1: Define your data structure
    // IrisData is used to provide training data, and as 
    // input for prediction operations
    // - First 4 properties are inputs/features used to predict the label
    // - Label is what you are predicting, and is only set when training
    public class IrisData
    {
        [Column("0")]
        public float SepalLength { get; set; }

        [Column("1")]
        public float SepalWidth { get; set; }

        [Column("2")]
        public float PetalLength { get; set; }

        [Column("3")]
        public float PetalWidth { get; set; }

        [Column(ordinal: "4", name: "Label")]
        public string Label { get; set; }
    }
}
