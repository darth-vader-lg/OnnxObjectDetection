using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxObjectDetection
{
   class Model
   {
      #region Fields
      /// <summary>
      /// Nomi delle classi
      /// </summary>
      private readonly string[] classesNames;
      /// <summary>
      /// Altezza immagine
      /// </summary>
      private readonly int imageHeight;
      /// <summary>
      /// Larghezza immagine
      /// </summary>
      private readonly int imageWidth;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      private readonly MLContext mlContext;
      /// <summary>
      /// Path del modello onnx
      /// </summary>
      private readonly string modelPath;
      /// <summary>
      /// Modello onnx
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Predictor
      /// </summary>
      private PredictionEngine<PredictionData, PredictionResult> predictor;
      #endregion
      #region Properties
      /// <summary>
      /// Predictor
      /// </summary>
      public PredictionEngine<PredictionData, PredictionResult> Predictor
      {
         get
         {
            model ??= LoadModel();
            predictor ??= mlContext.Model.CreatePredictionEngine<PredictionData, PredictionResult>(model);
            return predictor;
         }
      }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="modelPath">Posizione del modello onnx</param>
      /// <param name="classesNames">Nomi delle classi</param>
      /// <param name="imageWidth">Larghezza delle immagini del modello</param>
      /// <param name="imageHeight">Altezza delle immagini del modello</param>
      public Model(MLContext mlContext, string modelPath, string[] classesNames, int imageWidth = 640, int imageHeight = 640)
      {
         this.mlContext = mlContext;
         this.modelPath = modelPath;
         this.classesNames = classesNames;
         this.imageWidth = imageWidth;
         this.imageHeight = imageHeight;
      }
      /// <summary>
      /// Carica il modello onnx
      /// </summary>
      /// <returns>Il modello di trasformazione</returns>
      private ITransformer LoadModel()
      {
         // Crea una dataview per ottenere lo schema di dati di input
         var data = mlContext.Data.LoadFromEnumerable(new List<PredictionData>());
         // Definisce la pipeline
         var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: imageWidth, imageHeight: imageHeight, resizing: ResizingKind.Fill)
             .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
             .Append(mlContext.Transforms.ApplyOnnxModel(
                 shapeDictionary: new Dictionary<string, int[]>()
                 {
                     { "images", new[] { 1, 3, imageWidth, imageHeight } },
                     { "output1", new[] { 1, 3, 80, 80, 5 + classesNames.Length } },
                     { "output2", new[] { 1, 3, 40, 40, 5 + classesNames.Length } },
                     { "output3", new[] { 1, 3, 20, 20, 5 + classesNames.Length } },
                 },
                 inputColumnNames: new[]
                 {
                     "images"
                 },
                 outputColumnNames: new[]
                 {
                     "output1",
                     "output2",
                     "output3"
                 },
                 modelFile: modelPath));
         // Ottiene il modello di trasformazione
         var model = pipeline.Fit(data);
         return model;
      }
      #endregion
   }
}

