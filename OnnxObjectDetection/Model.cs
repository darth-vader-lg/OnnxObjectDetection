using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxObjectDetection
{
   class Model : ITransformer
   {
      #region Fields
      /// <summary>
      /// Configurazione del modello
      /// </summary>
      private readonly ModelConfiguration config;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      private readonly MLContext mlContext;
      /// <summary>
      /// Modello onnx
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Path del modello onnx
      /// </summary>
      private readonly string modelPath;
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
      /// <param name="config">Configurazione del modello</param>
      public Model(MLContext mlContext, string modelPath, ModelConfiguration config = null)
      {
         this.mlContext = mlContext;
         this.modelPath = modelPath;
         this.config = config ?? new ModelConfiguration();
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
         if (Path.GetExtension(modelPath).ToLower() == ".onnx") {
            // Definisce la pipeline
            var pipeline = mlContext.Transforms
               .LoadImages(inputColumnName: "ImagePath", outputColumnName: "Bitmap", imageFolder: "")
               .Append(mlContext.Transforms.Expression(inputColumnNames: new[] { "ImagePath" }, outputColumnName: "ModelWidth", expression: $"w => {config.ImageWidth}f"))
               .Append(mlContext.Transforms.Expression(inputColumnNames: new[] { "ImagePath" }, outputColumnName: "ModelHeight", expression: $"w => {config.ImageHeight}f"))
               .Append(mlContext.Transforms.ResizeImages(inputColumnName: "Bitmap", outputColumnName: "ResizedBitmap", imageWidth: config.ImageWidth, imageHeight: config.ImageHeight, resizing: ResizingKind.Fill))
               .Append(mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedBitmap", outputColumnName: config.InputName, scaleImage: 1f / 255f, interleavePixelColors: false))
               .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { config.InputName }, outputColumnNames: config?.OutputNames, modelFile: modelPath));
            // Ottiene il modello di trasformazione
            var data = mlContext.Data.LoadFromEnumerable(new List<PredictionData>());
            return pipeline.Fit(data);
         }
         else
            return mlContext.Model.Load(modelPath, out _);
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

