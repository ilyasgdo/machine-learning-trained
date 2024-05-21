// Load the image file
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using MLModel1_ConsoleApp1;
using Microsoft.ML.Data;

using Microsoft.ML.Data;
using SkiaSharp;
using System.Drawing;

// Load the image file
// Load the image file
var imagePath = @"C:\Users\ilyas\Pictures\ia\Stop-Signs\ron-mcclenny-EpHH_NKwKkE-unsplash.jpg";
var image = MLImage.CreateFromFile(imagePath);

// Create a single instance of sample data from the first line of dataset for model input.
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
    Console.WriteLine($"XTop: {item.Box.XTop}, YTop: {item.Box.YTop}, XBottom: {item.Box.XBottom}, YBottom: {item.Box.YBottom}, Score: {item.Score}");
}

// Load the image using System.Drawing
Bitmap bitmap = new Bitmap(imagePath);
using (Graphics graphics = Graphics.FromImage(bitmap))
{
    // Draw bounding boxes on the image
    foreach (var item in boxes)
    {
        var pen = new Pen(Color.Red, 2);
        graphics.DrawRectangle(pen, item.Box.XTop, item.Box.YTop, item.Box.XBottom - item.Box.XTop, item.Box.YBottom - item.Box.YTop);
    }
}

// Save the modified image to a file
string outputImagePath = @"D:\annotated_image.jpg";
bitmap.Save(outputImagePath);

