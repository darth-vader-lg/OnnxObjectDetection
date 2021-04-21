using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxObjectDetection
{
   partial class Model : ITransformer
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
            // Crea una dataview per ottenere lo schema di dati di input
            var data = mlContext.Data.LoadFromEnumerable(new List<PredictionData>());
            // Definisce la pipeline
            var pipeline = mlContext.Transforms
               .LoadImages(inputColumnName: "ImagePath", outputColumnName: "Bitmap", imageFolder: "")
               .Append(mlContext.Transforms.CustomMapping(new PredictionDataCustomMapping(this).GetMapping(), nameof(PredictionDataCustomMapping)))
               .Append(mlContext.Transforms.ResizeImages(inputColumnName: "Bitmap", outputColumnName: "ResizedBitmap", imageWidth: config.ImageWidth, imageHeight: config.ImageHeight, resizing: ResizingKind.Fill))
               .Append(mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedBitmap", outputColumnName: config.InputName, scaleImage: 1f / 255f, interleavePixelColors: false))
               .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { config.InputName }, outputColumnNames: config?.OutputNames, modelFile: modelPath));
            // Ottiene il modello di trasformazione
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

   partial class Model // PredictionDataExt
   {
      /// <summary>
      /// Mappatura per aggiunta informazioni ai dati di ingresso
      /// </summary>
      [CustomMappingFactoryAttribute(nameof(PredictionDataCustomMapping))]
      private class PredictionDataCustomMapping : CustomMappingFactory<PredictionData, PredictionDataExt>
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly Model owner;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner"></param>
         public PredictionDataCustomMapping(Model owner) => this.owner = owner;
         /// <summary>
         /// Azione di mappatura
         /// </summary>
         /// <returns>L'azione</returns>
         public override Action<PredictionData, PredictionDataExt> GetMapping() => new Action<PredictionData, PredictionDataExt>((input, output) =>
         {
            // Copia ed aggiunge informazioni
            output.ImagePath = input.ImagePath;
            output.ModelWidth = owner.config?.ImageWidth ?? 640;
            output.ModelHeight = owner.config?.ImageHeight ?? 640;
         });
         #endregion
      }

      /// <summary>
      /// Estensione della classe PredictionData
      /// </summary>
      class PredictionDataExt : PredictionData
      {
         #region Properties
         /// <summary>
         /// Larghezza immagine del modello
         /// </summary>
         [ColumnName("ModelWidth")]
         public float ModelWidth { get; set; }
         /// <summary>
         /// Altezza immagine del modello
         /// </summary>
         [ColumnName("ModelHeight")]
         public float ModelHeight { get; set; }
         #endregion
      }
   }
}

