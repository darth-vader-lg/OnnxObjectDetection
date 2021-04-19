using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxObjectDetection
{
   class Model : ITransformer
   {
      #region Fields
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
      /// Indicatore di RowToRowMapper
      /// </summary>
      public bool IsRowToRowMapper => (model ??= LoadModel()).IsRowToRowMapper;
      /// <summary>
      /// Predictor
      /// </summary>
      public PredictionEngine<PredictionData, PredictionResult> Predictor
      {
         get
         {
            predictor ??= mlContext.Model.CreatePredictionEngine<PredictionData, PredictionResult>(model ??= LoadModel());
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
      /// <param name="imageWidth">Larghezza delle immagini del modello</param>
      /// <param name="imageHeight">Altezza delle immagini del modello</param>
      public Model(MLContext mlContext, string modelPath, int imageWidth = 640, int imageHeight = 640)
      {
         this.mlContext = mlContext;
         this.modelPath = modelPath;
         this.imageWidth = imageWidth;
         this.imageHeight = imageHeight;
      }
      /// <summary>
      /// Restituisce lo schema di output
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns></returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => (model ??= LoadModel()).GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mapper</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => (model ??= LoadModel()).GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Carica il modello onnx
      /// </summary>
      /// <returns>Il modello di trasformazione</returns>
      private ITransformer LoadModel()
      {
         // Crea una dataview per ottenere lo schema di dati di input
         var data = mlContext.Data.LoadFromEnumerable(new List<PredictionData>());
         // Definisce la pipeline
         var pipeline = mlContext.Transforms
            .LoadImages(outputColumnName: "bitmap", imageFolder: "", inputColumnName: "ImagePath")
            .Append(mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "image", imageWidth: imageWidth, imageHeight: imageHeight, resizing: ResizingKind.Fill))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", inputColumnName: "image", scaleImage: 1f / 255f, interleavePixelColors: false))
            .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { "images" }, outputColumnNames: new[] { "output1", "output2", "output3" }, modelFile: modelPath));
         // Ottiene il modello di trasformazione
         var model = pipeline.Fit(data);
         return model;
      }
      /// <summary>
      /// Effettua il salvataggio del modello
      /// </summary>
      /// <param name="ctx">Contesto di salvataggio</param>
      public void Save(ModelSaveContext ctx) => (model ??= LoadModel()).Save(ctx);
      /// <summary>
      /// Effettua la trasformazione di dati
      /// </summary>
      /// <param name="input">Dati di ingresso</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input) => (model ??= LoadModel()).Transform(input);
      #endregion
   }
}

