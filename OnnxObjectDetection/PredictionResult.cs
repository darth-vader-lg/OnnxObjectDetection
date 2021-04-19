using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace OnnxObjectDetection
{
   partial class PredictionResult
   {
      #region Properties
      /// <summary>
      /// Identity
      /// </summary>
      [VectorType()] //@@@ Controllare se e' eliminabile
      [ColumnName("output1")]
      public float[] Output1 { get; set; }
      /// <summary>
      /// Identity1
      /// </summary>
      [VectorType()]
      [ColumnName("output2")]
      public float[] Output2 { get; set; }
      /// <summary>
      /// Identity2
      /// </summary>
      [VectorType()]
      [ColumnName("output3")]
      public float[] Output3 { get; set; }
      /// <summary>
      /// Bitmap ridimensionato
      /// </summary>
      [ColumnName("image")]
      [ImageType(640, 640)]
      public Bitmap Image { get; set; }
      /// <summary>
      /// Larghezza immagine
      /// </summary>
      [ColumnName("width")]
      public float ImageWidth { get; set; }
      /// <summary>
      /// Altezza immagine
      /// </summary>
      [ColumnName("height")]
      public float ImageHeight { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Parses net output to predictions.
      /// </summary>
      /// <param name="categories">Elenco di categorie</param>
      /// <param name="confidence">Accuratezza minima della previsione</param>
      /// <param name="perCategoryConfidence">Accuratezza minima per categoria</param>
      /// <param name="nmsOverlapRatio">Filtro per la rimozione delle previsioni sovrapposte (rapporto fra le aree di sovrapposizione)</param>
      public IReadOnlyList<Result> GetResults(float confidence = 0.2f, float perCategoryConfidence = 0.25f, float nmsOverlapRatio = 0.45f)
      {
         // Risultato
         var results = new List<Result>();
         // Fattori di scala
         var (xGain, yGain) = (Image.Width / ImageWidth, Image.Height / ImageHeight);
         // Elenco di output del modello
         var outputs = new[] { Output1, Output2, Output3 };
         // Shapes dei singoli otuput
         var outputShapes = new int[] { 80, 40, 20 };
         // Ancoraggi del modello per singoli output
         var anchors = new float[][][]
         {
             new float[][] { new float[] { 010f, 13f }, new float[] { 016f, 030f }, new float[] { 033f, 023f } },
             new float[][] { new float[] { 030f, 61f }, new float[] { 062f, 045f }, new float[] { 059f, 119f } },
             new float[][] { new float[] { 116f, 90f }, new float[] { 156f, 198f }, new float[] { 373f, 326f } }
         };
         // Dimensione di ogni previsione
         var dimension = Output1.Length / (80 * 80 * 3);
         // Stride per ciascuna shape
         var strides = new float[] { 8f, 16f, 32f };
         // Loop sugli output
         for (var i = 0; i < outputs.Length; i++) {
            // Shapes per output
            var shapesCount = outputShapes[i];
            var shapeOffset = shapesCount * shapesCount;
            // Loop sugli ancoraggi
            for (var a = 0; a < anchors.Length; a++) {
               var shapesOffsetA = shapeOffset * a;
               // Loop sulle righe
               for (var y = 0; y < shapesCount; y++) {
                  var shapeOffsetY = shapesOffsetA + shapesCount * y;
                  // Loop sulle colonne
                  for (int x = 0; x < shapesCount; x++) {
                     // Offset della previsione
                     var offset = (shapeOffsetY + x) * dimension;
                     // Buffer della previsione
                     var buffer = outputs[i].Skip(offset).Take(dimension).Select(Sigmoid).ToArray();
                     // Estrae la accuratezza della previsione
                     var objConfidence = buffer[4];
                     // Verifica se e' sopra la soglia di filtraggio
                     if (objConfidence < confidence)
                        continue;
                     // Punteggi per ogni categoria
                     var scores = buffer.Skip(5).Select(x => x * objConfidence).ToList();
                     // Punteggio massimo fra le categorie
                     var max = scores.Max();
                     // Verifica se il punteggio e' oltre la soglia del punteggio minimo per categoria
                     if (max <= perCategoryConfidence)
                        continue;
                     // Calcola il centro del bounding box in x e y
                     var rawX = (buffer[0] * 2f - 0.5f + x) * strides[i];
                     var rawY = (buffer[1] * 2f - 0.5f + y) * strides[i];
                     // Calcola la larghezza e altezza del bounding box
                     var rawW = MathF.Pow(buffer[2] * 2f, 2f) * anchors[i][a][0]; // predicted bbox width
                     var rawH = MathF.Pow(buffer[3] * 2f, 2f) * anchors[i][a][1]; // predicted bbox height
                     // Coordinate scalate del bounding box
                     var xMin = (rawX - rawW / 2f) / xGain;
                     var yMin = (rawY - rawH / 2f) / yGain;
                     var xMax = (rawX + rawW / 2f) / xGain;
                     var yMax = (rawY + rawH / 2f) / yGain;
                     // Crea il risultato
                     var result = new Result(new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin), scores.IndexOf(max), max);
                     results.Add(result);
                  }
               }
            }
         }
         if (nmsOverlapRatio < 1f)
            results = NMS(results, nmsOverlapRatio);
         return results;
      }
      /// <summary>
      /// Rimuove i duplicati sovrapposti (non max suppression).
      /// </summary>
      /// <param name="items">Elenco di risultati da filtrare</param>
      /// <param name="nmsOverlapRatio">Filtro di sovrapposizione area</param>
      private static List<Result> NMS(List<Result> items, float nmsOverlapRatio)
      {
         var results = new List<Result>(items);
         foreach (var item in items) {
            foreach (var current in results.ToList()) {
               if (current == item)
                  continue;
               var intersection = RectangleF.Intersect(item.Box, current.Box);
               var intArea = intersection.Width * intersection.Height;
               var unionArea = item.Box.Width * item.Box.Height + current.Box.Width * current.Box.Height - intArea;
               var overlap = intArea / unionArea;
               if (overlap > nmsOverlapRatio) {
                  if (item.Confidence > current.Confidence)
                     results.Remove(current);
               }
            }
         }
         return results;
      }
      /// <summary>
      /// Funzione sigmoide per output fra 0 e 1
      /// </summary>
      private float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));
      #endregion
   }

   partial class PredictionResult // Result
   {
      /// <summary>
      /// Risultato della previsione
      /// </summary>
      public class Result
      {
         #region Properties
         /// <summary>
         /// Box della previsione
         /// </summary>
         public RectangleF Box { get; }
         /// <summary>
         /// Accuratezza
         /// </summary>
         public float Confidence { get; }
         /// <summary>
         /// Categoria
         /// </summary>
         public int Category { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="box">Bounding box</param>
         /// <param name="category">Categoria</param>
         /// <param name="confidence">Punteggio</param>
         public Result(RectangleF box, int category, float confidence)
         {
            Box = box;
            Category = category;
            Confidence = confidence;
         }
         #endregion
      }
   }
}