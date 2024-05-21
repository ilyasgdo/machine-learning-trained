﻿// This file was auto-generated by ML.NET Model Builder. 

using MLModel1_ConsoleApp1;
using Microsoft.ML.Data;

// Create single instance of sample data from first line of dataset for model input.
var image = MLImage.CreateFromFile(@"C:\Users\ilyas\Pictures\ia\Stop-Signs\yannis-h-Sqez8_QTi8o-unsplash.jpg");
MLModel1.ModelInput sampleData = new MLModel1.ModelInput()
{
    Image = image,
};
// Make a single prediction on the sample data and print results.
var predictionResult = MLModel1.Predict(sampleData);
Console.WriteLine("\n\nPredicted Boxes:\n");
if (predictionResult.PredictedBoundingBoxes == null)
{
    Console.WriteLine("No Predicted Bounding Boxes");
    return;
}
var boxes =
    predictionResult.PredictedBoundingBoxes.Chunk(4)
        .Select(x => new { XTop = x[0], YTop = x[1], XBottom = x[2], YBottom = x[3] })
        .Zip(predictionResult.Score, (a, b) => new { Box = a, Score = b });

foreach (var item in boxes)
{
    Console.WriteLine($"XTop: {item.Box.XTop},YTop: {item.Box.YTop},XBottom: {item.Box.XBottom},YBottom: {item.Box.YBottom}, Score: {item.Score}");
}

